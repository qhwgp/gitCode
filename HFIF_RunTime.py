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

import xlrd, threading,copy,time,datetime,pymssql,sys,csv,os
from queue import Queue, Empty
from threading import Thread
import WindTDFAPI as w
from keras import models
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
lock = threading.Lock()

global dictForeFactor,sql,dataVendor,datapath

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
        
eventManager = EventManager()

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

def MyIniData(basicparams):
    global dictForeFactor
    prm=basicparams.data
    dictForeFactor[prm[0]]=ForeFactor(*prm)
    dataVendor.RegisterSymbol(dictForeFactor[prm[0]].codelist)

def MyNormData(normEvent):
    global dictForeFactor,sql

    lock.acquire()
    try:
        for key in dictForeFactor:
            dictForeFactor[key].CalPM()
            #pm=dictForeFactor[key].pm
            #sql.UpdateFF(dictForeFactor[key].StrategyName,pm[0],pm[1],pm[2])
    finally:
        lock.release()
        
def ReceiveQuote(pMarketdata):
    dt =pMarketdata.data
    lock.acquire()
    try:
        code=bytes.decode(dt.szWindCode)
        for key in dictForeFactor:
            if code in dictForeFactor[key].codelist:
                dictForeFactor[key].dictquote[code]=(dt.nTime/1000,dt.nMatch/10000,dt.iTurnover/1000000)
    finally:
        lock.release()
    
class ForeFactor:
    def __init__(self, StrategyName,PathWei,PathParams,Path1min,Path2min,Path3min,
                 nrow,ncolumn,ffw1m,ffw2m,ffw3m,ffwt):
        self.StrategyName = StrategyName
        self.nrow = nrow
        self.ncolumn = ncolumn
        self.ffw1m = ffw1m
        self.ffw2m = ffw2m
        self.ffw3m = ffw3m
        self.ffwt = ffwt
        self.forerate=0
        xsheet=xlrd.open_workbook(PathWei).sheets()[0]
        nrow=xsheet.nrows
        self.codelist=set()
        self.dictCodeWeight={}
        for i in range(nrow):
            dt=xsheet.row_values(i)
            self.codelist.add(dt[0])
            self.dictCodeWeight[dt[0]]=(float(dt[1]),int(dt[2]))
        self.codelist.add('399905.SZ')
        self.nparams=np.load(PathParams)[:,1:]
        self.model_1min=models.load_model(Path1min)
        self.model_2min=models.load_model(Path2min)
        self.model_3min=models.load_model(Path3min)
        self.ndata=np.zeros((self.nrow,self.ncolumn))
        self.pm=np.zeros(3)
        self.dictquote={}
        self.lastdictquote={}
        
    def CalPM(self):
        crow=np.zeros(self.ncolumn)
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
    dictForeFactor={}
    #os
    dttime=datetime.datetime.now().date().strftime('%Y%m%d')
    datapath='D:\\data\\'+dttime
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    #SQL
    sql=MSSQL(host='10.200',user='s',pwd='s',db='s')
    if not sql.Connect():
        print('Connet Error')
        sys.exit(0)

    #Event
    eventManager.AddEventListener("quote",ReceiveQuote)
    eventManager.AddEventListener("iniData",MyIniData)
    eventManager.AddEventListener("normData",MyNormData)
    eventManager.Start()
    
    #MarketData
    w.SetMarketDataCallBack(TDFCallBack)
    dataVendor = w.WindMarketVendor("TDFConfig.ini", "TDFAPI25.dll")
    nConnect=0
    while (dataVendor.Reconnect() is False):
        print("Error nConnect: ",nConnect)
        nConnect+=1
        time.sleep(5)
    
    #Initual Data
    path="D:\\CalForeFactor\\"
    StrategyName='QUOTER_510500'
    PathWei=path+"ZZ500wei.xlsx"
    PathParams=path+"nparams.npy"
    Path1min=path+"FF_1min_3level_36.h5"
    Path2min=path+"FF_2min_3level_36.h5"
    Path3min=path+"FF_3min_3level_36.h5"
    nrow=36
    ncolumn=16
    ffw1m=0.7
    ffw2m=0.3
    ffw3m=0.1
    ffwt=0.7
    prm=(StrategyName,PathWei,PathParams,Path1min,Path2min,Path3min,
                 nrow,ncolumn,ffw1m,ffw2m,ffw3m,ffwt)
    eventManager.SendEvent(MyEvent("iniData",prm))
    
    while True:
        eventManager.SendEvent(MyEvent("normData",""))
        time.sleep(5)
    
    
