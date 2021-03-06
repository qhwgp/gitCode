# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:26:07 2019

@author: WGP
"""

import xlrd, threading,datetime,os,pymssql,time
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from queue import Queue, Empty
from threading import Thread
import WindTDFAPI as w
from keras import models,backend
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

def calPercentile(xValue,arrPercentile,st=0): #len(arrPercentile)=100,upscane
    isfind=False
    abv=abs(xValue)
    for i in range(st,100):
        if abv<(arrPercentile[i]+0.00001):
            isfind=True
            break
    if isfind:
        result=i/100
    else:
        result=1
    return result*np.sign(xValue)

def getNormInduData(xData,pclMatrix):
    xShape=len(xData)
    normInduData=np.zeros(xShape)
    for j in range(xShape):
        arrPercentile=pclMatrix[:,j]
        normInduData[j]=calPercentile(xData[j],arrPercentile)
    return normInduData

def btstr(btpara):
    return str(btpara,encoding='utf-8')

def myLoss(y_true, y_pred):
    #return backend.mean(backend.square((y_pred - y_true)*y_true), axis=-1)
    return backend.mean(backend.abs((y_pred - y_true)*y_true), axis=-1)

def myMetric(y_true, y_pred):
    return backend.mean(y_pred*y_true, axis=-1)*10

def getCfgFareFactor(ffPath):
    cfgFile=os.path.join(ffPath,'cfgForeFactor.csv')
    cfgData=tuple(map(btstr,np.loadtxt(cfgFile,dtype=bytes)))
    return cfgData[:4],cfgData[4:]

def getTSAvgAmnt(pdAvgAmntFile):
    global dictTSAvgAmnt,timeSpan
    pdAvgAmnt=pd.read_csv(pdAvgAmntFile,header=0,index_col=0,engine='python')
    for code in pdAvgAmnt.index:
        dictTSAvgAmnt[code]=int(pdAvgAmnt.loc[code][0]/(14400/timeSpan))

def registerAllSymbol():
    global dataVendor,listForeFactor
    codelist={}
    for ff in listForeFactor:
        codelist=codelist|set(ff.dictCodeInfo.keys())
    dataVendor.RegisterSymbol(codelist)
    print('Register symbol, please wait...')
    
class EventManager:
    def __init__(self):
        self.__eventQueue = Queue()
        self.__active = False
        self.__thread = Thread(target = self.__Run)
        self.__handlers = {}

    def __Run(self):
        while self.__active == True:
            try:
                event = self.__eventQueue.get(block = True, timeout = 1)  
                self.__EventProcess(event)
            except Empty:
                pass

    def __EventProcess(self, event):
        if event.type_ in self.__handlers:
            for handler in self.__handlers[event.type_]:
                handler(event)

    def Start(self):
        self.__active = True 
        self.__thread.start()

    def AddEventListener(self, type_, handler):
        try:
            handlerList = self.__handlers[type_]
        except KeyError:
            handlerList = []

        self.__handlers[type_] = handlerList
        if handler not in handlerList:
            handlerList.append(handler)
    
    def SendEvent(self, event):
        self.__eventQueue.put(event)

class MyEvent:
    def __init__(self, Eventtype,Data):
        self.type_ = Eventtype      # 事件类型
        self.data = Data          # 字典用于保存具体的事件数据

class MSSQL:

    def __init__(self,host,user,pwd,db):
        self.host = host
        self.user = user
        self.pwd = pwd
        self.db = db

    def Connect(self):
        try:
            self.conn = pymssql.connect(host=self.host,user=self.user,password=self.pwd,database=self.db,charset="UTF-8")
            self.conn.autocommit(True)
            self.cur = self.conn.cursor()
            if not self.cur:
                return False
            else:
                return True
        except:
            return False
        
    def UpdateFF(self,sname,pm):
        sql="update tblFundPricingParam set ff_1m_v=("+str(pm[0])+"),ff_2m_v=("+str(pm[1])+"),ff_3m_v=("+str(pm[2])+") where strategyName='"+sname+"'"
        self.cur.execute(sql)
        
    def UpdateAllFF(self):
        global listForeFactor
        listUpdate=[]
        lsn=[]
        for i in range(3):
            listUpdate.append('ff_'+str(i+1)+'m_v=case strategyName')
        for ff in listForeFactor:
            for strategyName in ff.listStrategyName:
                lsn.append('\''+strategyName+'\'')
                for i in range(3):
                    listUpdate[i]+=' when \''+strategyName+'\' then '+str(ff.pm[i])
        for i in range(3):
            listUpdate[i]+=' end'
        sql='update tblFundPricingParam set '+','.join(listUpdate)+' where strategyName in ('+','.join(lsn)+')'
        self.cur.execute(sql)

def TDFCallBack(pMarketdata):
    eventManager.SendEvent(MyEvent("quote",pMarketdata))

def MyNormData(normEvent):
    global listForeFactor,lock,sql
    isPush=normEvent.data
    listPm=[]
    intNTime=int(datetime.datetime.now().strftime('%H%M%S'))
    if intNTime<91000 or intNTime>150000:
        print('not trading time.')
        return
    lock.acquire()
    try:
        for ff in listForeFactor:
            ff.CalPM()
            listPm.append(ff.pm)
        print(intNTime,*tuple(listPm))
        if isPush and ((intNTime>93100 and intNTime<113000) or (intNTime>130100 and intNTime<150000)):
            sql.UpdateAllFF()
    finally:
        lock.release()
        
def ReceiveQuote(quoteEvent):
    global dictQuote,lock
    dt =quoteEvent.data
    lock.acquire()
    try:
        code=bytes.decode(dt.szWindCode)
        dictQuote[code]=(dt.nTime/1000,dt.nMatch/10000,dt.iTurnover)
    finally:
        lock.release()
    
class ForeFactor:
    def __init__(self, workPath,cfgFile):
        self.workPath = workPath
        self.cfgFile = cfgFile
        self.dictCodeInfo = {}
        self.nIndu=0
        self.listModel=[]
        self.listStrategyName=[]
        self.pclMatrix=np.array([])
        #output
        self.lastInduData=np.array([])
        self.inputData=np.array([])
        self.pm=np.zeros(3)
        self._getCfg()
    
    def _getCfg(self):
        global nXData,timeSpan
        data = xlrd.open_workbook(os.path.join(self.workPath,self.cfgFile))
        sheetCodeInfo = data.sheets()[0]
        arrShares = sheetCodeInfo.col_values(1)[1:]
        arrCode = sheetCodeInfo.col_values(0)[1:]
        arrIndustry = sheetCodeInfo.col_values(2)[1:]
        self.nIndu=len(set(arrIndustry))
        self.inputData=np.zeros((nXData,self.nIndu*2))
        for i in range(len(arrCode)):
            self.dictCodeInfo[arrCode[i]]=[arrShares[i],arrIndustry[i]]
        arrCfg=data.sheets()[1].col_values(1)
        self.listStrategyName=arrCfg[10].split(',')
        (filepath,tempfilename) = os.path.split(self.cfgFile)
        (filename,extension) = os.path.splitext(tempfilename)
        modelPath=os.path.join(self.workPath,filename)
        testP=np.zeros((1,nXData,self.nIndu*2))
        for i in range(3):
            modelfile=os.path.join(modelPath,'model_'+filename+'_1min_'+str(i+1)+'min.h5')
            model=models.load_model(modelfile,custom_objects={'myLoss': myLoss,'myMetric':myMetric})
            model.predict(testP)
            self.listModel.append(model)
        self.pclMatrix=np.loadtxt(os.path.join(modelPath,'pclMatrix_'+filename+'.csv'),delimiter=',')
        getTSAvgAmnt(os.path.join(modelPath,'avgAmnt_'+filename+'.csv'))
    
    def CalPM(self):
        global dictQuote,dictTSAvgAmnt,nXData
        crow=np.zeros(self.nIndu*2)
        inputRow=np.zeros(self.nIndu*2)
        npAveTSpanAmnt=np.zeros(self.nIndu)
        for (symbol,weiIndu) in self.dictCodeInfo.items():
            if (symbol not in dictQuote):# or (symbol not in dictTSAvgAmnt):
                #print('np Symbol: '+ symbol)
                #return
                continue
            wei=weiIndu[0]
            intIndu=int(weiIndu[1]+0.1)
            lpri=dictQuote[symbol][1]
            lamt=dictQuote[symbol][2]
            crow[2*intIndu-2]+=wei*lpri
            crow[2*intIndu-1]+=lamt
            npAveTSpanAmnt[intIndu-1]+=dictTSAvgAmnt[symbol]
        
            #if lpri<0.01:
                #print('price 0: '+symbol)
                #continue
        if crow[0]<1:
            print('wait quote')
            return
        if self.lastInduData.size==0:
            self.lastInduData=crow
        for i in range(self.nIndu):
            inputRow[2*i]=(crow[2*i]/self.lastInduData[2*i]-1)*10000
            inputRow[2*i+1]=(crow[2*i+1]-self.lastInduData[2*i+1])/npAveTSpanAmnt[i]
        inputRow=getNormInduData(inputRow,self.pclMatrix)
        self.inputData=np.vstack((self.inputData[1:,:],inputRow))
        self.lastInduData=crow
        for i in range(3):
            self.pm[i]=self.listModel[i].predict(self.inputData.reshape(1,nXData,self.nIndu*2))[0,0]
            #backend.clear_session()
        self.pm=np.round(self.pm,2)
        
if __name__ == '__main__':
    #global
    eventManager = EventManager()
    lock = threading.Lock()
    listForeFactor=[]
    dictQuote={}
    dictTSAvgAmnt={}
    timeSpan=3
    nXData=20
    
    #config
    cfgPath='D:\\CalForeFactor\\HFI_Model'
    if not os.path.exists(cfgPath):
        cfgPath='C:\\Users\\WAP\\Documents\\HFI_Model'
    cfgSQL,listCfgForeFactor=getCfgFareFactor(cfgPath)
    fPath=os.path.join(cfgPath,'cfg')
    for cfgFF in listCfgForeFactor:
        listForeFactor.append(ForeFactor(fPath,cfgFF))
    
    #SQL
    sql=MSSQL(*cfgSQL)

    nConnect=0
    while not sql.Connect():
        print('SQL Connet Error: ',nConnect)
        nConnect+=1
        time.sleep(5)

    #Event
    eventManager.AddEventListener("quote",ReceiveQuote)
    eventManager.AddEventListener("normData",MyNormData)
    eventManager.Start()
    
    w.SetMarketDataCallBack(TDFCallBack)
    dataVendor = w.WindMarketVendor("TDFConfig.ini", "TDFAPI25.dll")
    nConnect=0
    while (dataVendor.Reconnect() is False):
        print("Error nConnect: ",nConnect)
        nConnect+=1
        time.sleep(5)
    dataVendor.RegisterSymbol(set(dictTSAvgAmnt.keys()))
    time.sleep(10)
    for i in range(30):
        eventManager.SendEvent(MyEvent("normData",False))
        time.sleep(timeSpan)
    while True:
        eventManager.SendEvent(MyEvent("normData",True))
        time.sleep(timeSpan)

    
