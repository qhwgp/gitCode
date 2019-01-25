# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:26:07 2019

@author: WGP
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 09:18:19 2017

@author: wap
"""

import xlrd, threading,copy,time,datetime,pymssql,sys,os
from queue import Queue, Empty
from threading import Thread
import WindTDFAPI as w
from keras import models,backend
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"



def myLoss(y_true, y_pred):
    return backend.mean(backend.square((y_pred - y_true)*y_true), axis=-1)

def getCfgFareFactor(ffPath):
    cfgFile=os.path.join(ffPath,'cfgForeFactor.csv')
    cfgData=tuple(map(str,np.loadtxt(cfgFile,dtype=str)))
    return cfgData[:4],cfgData[4:]

def registerAllSymbol():
    global dataVendor,listForeFactor
    codelist={}
    for ff in listForeFactor:
        codelist=codelist|set(ff.dictCodeInfo.keys())
    dataVendor.RegisterSymbol(codelist)
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
        
    def UpdateFF(self,sname,p1m,p2m,p3m):
        sql="update tblFundPricingParam set ff_1m_v=("+str(p1m)+"),ff_2m_v=("+str(p2m)+"),ff_3m_v=("+str(p3m)+") where strategyName='"+sname+"'"
        self.cur.execute(sql)
        #self.conn.commit()
    def CallProc(self,sname,p1m,p2m,p3m):
        self.cur.callproc('sp_update_params',(sname,p1m,p2m,p3m))
        #self.conn.commit()
    def GetFF(self):
        sql="select  ff_1m_v,ff_2m_v,ff_3m_v from tblFundPricingParam where strategyName='QUOTER_510500'"
        self.cur.execute(sql)
        return self.cur.fetchone()


def TDFCallBack(pMarketdata):
    eventManager.SendEvent(MyEvent("quote",pMarketdata))

def MyNormData(normEvent):
    global listForeFactor,lock,sql
    lock.acquire()
    try:
        for ff in listForeFactor:
            ff.CalPM()
            #pm=dictForeFactor[key].pm
            #sql.UpdateFF(dictForeFactor[key].StrategyName,pm[0],pm[1],pm[2])
    finally:
        lock.release()
        
def ReceiveQuote(quoteEvent):
    global dictQuote,lock
    dt =quoteEvent.data
    lock.acquire()
    try:
        code=bytes.decode(dt.szWindCode)
        dictQuote[code]=(dt.nTime/1000,dt.nMatch/10000,dt.iTurnover/1000000)
    finally:
        lock.release()
    
class ForeFactor:
    def __init__(self, workPath,cfgFile):
        self.workPath = workPath
        self.cfgFile = cfgFile
        self.dictCodeInfo = {}
        self.listModel=[]
        self.pclMatrix=np.array([])
        self.listStrategyName=[]
        #output
        self.lastInduData=np.array([])
        self.inputData=np.array([])
        self.pm=np.zeros(3)
        self._getCfg()
    
    def _getCfg(self):
        data = xlrd.open_workbook(self.cfgFile)
        sheetCodeInfo = data.sheets()[0]
        arrShares = sheetCodeInfo.col_values(1)[1:]
        arrCode = sheetCodeInfo.col_values(0)[1:]
        arrIndustry = sheetCodeInfo.col_values(2)[1:]
        nColumn=len(set(arrIndustry))*2
        nRow=20
        self.inputData=np.zeros((nRow,nColumn))
        self.lastInduData=np.zeros(nColumn)
        for i in range(len(arrCode)):
            self.dictCodeInfo[arrCode[i]]=[arrShares[i],arrIndustry[i]]
        arrCfg=data.sheets()[1].col_values(1)
        self.listStrategyName=arrCfg[10].split(',')
        (filepath,tempfilename) = os.path.split(self.cfgFile)
        (filename,extension) = os.path.splitext(tempfilename)
        modelPath=os.path.join(self.workPath,'cfg',filename)
        for i in range(3):
            modelfile=os.path.join(modelPath,'model_'+filename+'_'+str(i+1)+'min.h5')
            self.listModel.append(models.load_model(modelfile,custom_objects={'myLoss': myLoss}))
        self.pclMatrix=np.loadtxt(os.path.join(modelPath,'pclMatrix_'+filename+'.csv'),delimiter=',')
    
    def CalPM(self):
        crow=np.zeros_like(self.lastInduData)
        for (symbol,quote) in self.dictquote.items():
            if (symbol not in self.lastDictQuote) or (symbol not in self.dictCodeWeight):
                continue
            ncol=self.dictCodeWeight[symbol][1]-1
            wei=self.dictCodeWeight[symbol][0]
            lpri=self.lastDictQuote[symbol][1]
            lamt=self.lastDictQuote[symbol][2]
            
            pri=self.dictquote[symbol][1]
            amt=self.dictquote[symbol][2]
            if lpri<0.01 or lamt<0.01 or pri<0.01 or amt<0.01:
                continue
            ret=(pri/lpri-1)*wei*100
            damt=(amt-lamt)*wei
            crow[2*ncol]+=ret
            crow[2*ncol+1]+=damt
        self.lastDictQuote=copy.deepcopy(self.dictquote)
        for i in range(self.ncolumn):
            crow[i]=(crow[i]-self.nparams[0][i])/self.nparams[1][i]/2
        self.ndata=np.row_stack((self.ndata[1:,:],crow))
        p1m=self.model_1min.predict(self.ndata.reshape(1,self.nrow,self.ncolumn))[0,0]
        p2m=self.model_2min.predict(self.ndata.reshape(1,self.nrow,self.ncolumn))[0,0]
        p3m=self.model_3min.predict(self.ndata.reshape(1,self.nrow,self.ncolumn))[0,0]
        ffvn=round(p1m*0.7+p2m*0.2+p3m*0.1,3)
        self.forerate=round(self.forerate*0.7+ffvn*0.3,3)
        self.pm=[round(p1m,3),round(p2m,3),round(p3m,3)]
        print(datetime.datetime.now().strftime('%H:%M:%S'),self.StrategyName,round(p1m,3),round(p2m,3),round(p3m,3),ffvn,self.forerate)
        
 
        
if __name__ == '__main__':
    #global
    eventManager = EventManager()
    lock = threading.Lock()
    listForeFactor=[]
    dictQuote={}
    w.SetMarketDataCallBack(TDFCallBack)
    dataVendor = w.WindMarketVendor("TDFConfig.ini", "TDFAPI25.dll")
    
    #config
    cfgPath='F:\\草稿\\HFI_Model'
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
    
    nConnect=0
    while (dataVendor.Reconnect() is False):
        print("Error nConnect: ",nConnect)
        nConnect+=1
        time.sleep(5)
    registerAllSymbol()
    
    for i in range(20):
        eventManager.SendEvent(MyEvent("normData",False))
        time.sleep(5)
    while True:
        eventManager.SendEvent(MyEvent("normData",True))
        time.sleep(5)
    
    
