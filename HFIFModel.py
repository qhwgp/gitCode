# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 14:15:04 2018
version git1.0
@author: wap
"""

import time,csv,datetime,xlrd,os
from WindPy import w
import pandas as pd
import numpy as np

#--------------Function Start-----------------

def getWindPastAveAmnt(nday,listCode,strSDay,strEDay=''):
    if w.isconnected()==False:
        msg=w.start(waitTime=15)
        if msg.ErrorCode!=0:
            return False
    msg=w.tdaysoffset(-nday, strSDay.strftime('%Y-%m-%d'), "")
    if msg.ErrorCode!=0:
        return False
    amntSDay=msg.Data[0][0].strftime('%Y-%m-%d')
    strCode=','.join(listCode)
    amntEDay=''
    if strEDay!='':
        amntEDay=strEDay.strftime('%Y-%m-%d')
    msg=w.wsd(strCode, 'amt', amntSDay, amntEDay, "")
    if msg.ErrorCode!=0:
        return False
    amntData=msg.Data
    amntTimes=msg.Times
    amntCodes=msg.Codes
    if len(amntTimes)<=nday:
        return False
    dictPastAveAmnt={}
    for iday in range(nday,len(amntTimes)):
        strDay=amntTimes[iday].strftime('%Y%m%d')
        dictdailyPastAveAmnt={}
        for icode in range(len(amntCodes)):
            code=amntCodes[icode]
            listamnt=amntData[icode][(iday-nday):iday]
            nCount=0
            sumAmnt=0
            aveAmnt=0
            for amnt in listamnt:
                if amnt>1:
                    nCount+=1
                    sumAmnt+=amnt
            if nCount>nday/2:
                aveAmnt=sumAmnt/nCount
            dictdailyPastAveAmnt[code]=aveAmnt
        dictPastAveAmnt[strDay]=dictdailyPastAveAmnt
    return dictPastAveAmnt

def getDailyInduData(dictPartStdData,dictCodeInfo,dictdailyPastAveAmnt,timeSpan):
    arrIndu=np.array(list(dictCodeInfo.values()))[:,1]
    maxNIndu=int(np.max(arrIndu)+0.1)
    minNIndu=int(np.min(arrIndu)+0.1)
    nIndu=maxNIndu-minNIndu+1
    for listStdData in dictPartStdData.values():
        lenStdData=len(listStdData)
        break
    npDailyInduData=np.zeros([lenStdData,nIndu*2])
    npAveTSpanAmnt=np.zeros(nIndu)
    for code,weiIndu in dictCodeInfo.items():
        if not code in dictPartStdData.keys():
            continue
        wei=weiIndu[0]
        intIndu=int(weiIndu[1]+0.1)
        arrStdData=np.array(dictPartStdData[code])
        aveAmnt=dictdailyPastAveAmnt[code]
        #industry index cal
        npDailyInduData[:,2*(intIndu-minNIndu)]=npDailyInduData[:,
            2*(intIndu-minNIndu)]+wei*arrStdData[:,0]
        #industry amnt cal
        npDailyInduData[:,2*(intIndu-minNIndu)+1]=npDailyInduData[:,
            2*(intIndu-minNIndu)+1]+arrStdData[:,1]
        #industry average amnt cal
        npAveTSpanAmnt[intIndu-minNIndu]=npAveTSpanAmnt[intIndu-minNIndu]+aveAmnt
    #index->return,norm amnt
    npAveTSpanAmnt=npAveTSpanAmnt/(14400/timeSpan)
    for i in range(nIndu):
        npDailyInduData[1:,2*i]=(npDailyInduData[1:,
                2*i]/npDailyInduData[:-1,2*i]-1)*10000
        npDailyInduData[:,2*i+1]=npDailyInduData[:,2*i+1]/npAveTSpanAmnt[i]
    npDailyInduData=npDailyInduData[1:,:]
    return npDailyInduData

def csvToList(csvFile):
    resultList=[]
    with open(csvFile, mode='r') as f:
        csvReader = csv.reader(f)
        for row in csvReader:
            resultList.append(list(map(eval,row)))
    return resultList

def getDictStdData(nStdDataPath,listCode):
    dictStdData={}
    lenCode=len(listCode[0])
    for stdDataFileName in os.listdir(nStdDataPath):
        code=stdDataFileName[:lenCode]
        if code in listCode:
            #needed data
            tFlag=os.path.splitext(stdDataFileName)[0].split('_')[1]
            if not tFlag in dictStdData:
                dictStdData[tFlag]={}
            csvFile=os.path.join(nStdDataPath,stdDataFileName)
            dictStdData[tFlag][code]=csvToList(csvFile)
    return dictStdData

def listToCsv(listData,csvFilePath):
    #try:
    with open(csvFilePath, mode='w', newline='') as f:
        csvwriter = csv.writer(f)
        for row in listData:
            csvwriter.writerow(row)
    #    return True
    #except:
    #    return False
def intToTime(longInt):
    if longInt>240000000:
        return datetime.datetime(1900, 1, 1, 23, 59, 59)
    elif longInt<1000000:
        return datetime.datetime(1900, 1, 1, 0, 0, 0)
    else:
        temp=longInt/10000000
        thour=int(temp)
        temp=(temp-thour)*100
        tmin=int(temp+0.1)
        temp=(temp-tmin)*100
        tsec=int(temp+0.1)
        return datetime.datetime(1900, 1, 1, thour, tmin, tsec)
    
def CheckRawData(rawData):
    if len(rawData)<100:
        return 1
    
    return 0
    
def tickToStdData(rawTickData,ListtimeSE,secTimeDiff):
    """
    morningStart = datetime.datetime(1900, 1, 1, 9, 40, 0)
    morningEnd = datetime.datetime(1900, 1, 1, 11, 20, 0)
    afternoonStart = datetime.datetime(1900, 1, 1, 13, 10, 0)
    afternoonEnd = datetime.datetime(1900, 1, 1, 14, 50, 0)
    timeSE=[[morningStart,morningEnd],[afternoonStart,afternoonEnd]]
    timeDiff = datetime.timedelta(0, 3, 0)
    """
    listStdData=[]
    timeDiff = datetime.timedelta(0, secTimeDiff, 0)
    if CheckRawData(rawTickData)!=0:
        return listStdData
    nRawData=0
    for timeSpan in ListtimeSE:
        stdData=[]
        STime=timeSpan[0]
        ETime=timeSpan[1]
        len_series = int((ETime - STime) / timeDiff)
        lprice=0
        
        while intToTime(rawTickData[nRawData][0])<STime:
            lprice=rawTickData[nRawData][1]
            nRawData+=1
        for i in range(len_series+1):
            #lastTime=STime+(i-1)*timeDiff
            nowTime=STime+i*timeDiff
            amnt=0
            while intToTime(rawTickData[nRawData][0])<=nowTime:
                lprice=rawTickData[nRawData][1]
                amnt+=rawTickData[nRawData][2]
                nRawData+=1
            stdData.append([lprice,amnt])
        listStdData.append(stdData)
            
    return listStdData
    
#--------------Function End-----------------

#--------------Build HFIF Model Class-------

class AIHFIF:
    def __init__(self,workPath,cfgFile):
        self.workPath=workPath
        self.cfgFile=cfgFile
        #default cfg
        self.timeSE=[[datetime.datetime(1900, 1, 1, 9, 40, 0),
                      datetime.datetime(1900, 1, 1, 11, 20, 0)],[
                              datetime.datetime(1900, 1, 1, 13, 10, 0),
                              datetime.datetime(1900, 1, 1, 14, 50, 0)]]
        #cfg page1
        self.dictCodeInfo={}#{code,[share,industry]}
        #cfg page2
        self.timeSpan=3
        self.indexCode=''
        self.windIndexCode=''
        self.rawDataPath=''
        self.nDayAverage=0
        self.minuteXData=0
        self.minuteYData=0
        self.log=''
        self._getCfg()
        
        
    def _getCfg(self):
        data = xlrd.open_workbook(cfgFile)
        #page1
        sheetCodeInfo = data.sheets()[0]
        arrShares = sheetCodeInfo.col_values(1)[1:]
        arrCode = sheetCodeInfo.col_values(0)[1:]
        arrIndustry = sheetCodeInfo.col_values(2)[1:]
        for i in range(len(arrCode)):
            self.dictCodeInfo[arrCode[i]]=[arrShares[i],arrIndustry[i]]
        #page2
        sheetCfg=data.sheets()[1]
        arrCfg=sheetCfg.col_values(1)
        self.timeSpan=arrCfg[0]
        self.indexCode=arrCfg[1]
        self.windIndexCode=arrCfg[2]
        self.rawDataPath=arrCfg[3]
        self.nDayAverage=int(arrCfg[4]+0.01)
        self.minuteXData=int(arrCfg[5]+0.01)
        self.minuteYData=int(arrCfg[6]+0.01)
        return
    
    def updateStdData(self,startDate=0):
        mainOutPath=os.path.join(self.workPath,'StandardData',str(int(self.timeSpan))+'_sec_span')
        pathList = os.listdir(self.rawDataPath)
        listCode=list(self.dictCodeInfo.keys())
        listCode.append(self.indexCode)
        for path in pathList:
            if int(path)<startDate:
                continue
            datePath=os.path.join(self.rawDataPath,path)
            for code in listCode:
                outPath=os.path.join(mainOutPath,path)
                if not os.path.exists(outPath):
                    os.makedirs(outPath)
                if os.path.exists(os.path.join(outPath,code+'_0.csv')):
                    continue
                rawDataPath=os.path.join(datePath,code+'.csv')
                rawData=csvToList(rawDataPath)
                if CheckRawData(rawData)!=0:
                    continue
                listStdData=tickToStdData(rawData,self.timeSE,self.timeSpan)
                for i in range(len(listStdData)):
                    OutFile=os.path.join(outPath,code+'_'+str(i)+'.csv')
                    if not os.path.exists(OutFile):
                        listToCsv(listStdData[i],OutFile)
        return
    
    def calInduData(self,strSDate='19000101',strEDate=''):
        #get or set outpath
        (filepath,tempfilename) = os.path.split(self.cfgFile)
        (filename,extension) = os.path.splitext(tempfilename)
        outPath=os.path.join(self.workPath,'induData',filename)
        if not os.path.exists(outPath):
                    os.makedirs(outPath)
        #start&end day
        stdDataPath=os.path.join(self.workPath,'StandardData',str(int(self.timeSpan))+'_sec_span')
        listDate=os.listdir(stdDataPath)
        sDate=datetime.datetime.strptime(listDate[0],'%Y%m%d')
        eDate=datetime.datetime.strptime(listDate[-1],'%Y%m%d')
        startDate=datetime.datetime.strptime(strSDate,'%Y%m%d')
        if sDate<startDate:
            sDate=startDate
        if strEDate!='':
            endDate=datetime.datetime.strptime(strEDate,'%Y%m%d')
            if eDate>endDate:
                eDate=endDate
        #past nday's average amount
        listCode=list(self.dictCodeInfo.keys())
        dictPastAveAmnt=getWindPastAveAmnt(self.nDayAverage,listCode,
                                           sDate,eDate)
        #indu data
        for strDate in listDate:
            nDate=datetime.datetime.strptime(strDate,'%Y%m%d')
            if nDate<sDate or nDate>eDate:
                continue
            #collect nDate's standard data list
            nStdDataPath=os.path.join(stdDataPath,strDate)
            dictStdData=getDictStdData(nStdDataPath,listCode)
            for nFlag in dictStdData.keys():
                induDataFile=os.path.join(outPath,strDate+'_'+nFlag+'.csv')
                if os.path.exists(induDataFile):
                    continue
                dictPartStdData=dictStdData[nFlag]
                npDInduData=getDailyInduData(dictPartStdData,
                      self.dictCodeInfo,dictPastAveAmnt[strDate],self.timeSpan)
                
                np.savetxt(induDataFile,npDInduData,delimiter=',')
        return True
        
#---------------Build HFIF Model End--------

if __name__=='__main__':
    gtime = time.time()
    print('Start Running...')
    
    """
    csvFilePath="F:\\文档\\data\\ListData\\tickData\\20181227\\600000.SH.csv"
    resultList=csvToList(csvFilePath)
    
    writeFilePath="F:\\文档\\data\\ListData\\test.csv"
    tf=listToCsv(resultList,writeFilePath)
    
    longInt=resultList[2][0]
    mytimt=intToTime(longInt)
    """
    workPath='C:\\Users\\WAP\\Documents\\HFI_Model'
    cfgFile='C:\\Users\\WAP\\Documents\\HFI_Model\\cfg\\cfg_sz50_v331atan.xlsx'
    HFIF_Model=AIHFIF(workPath,cfgFile)
    dictCodeInfo=HFIF_Model.dictCodeInfo
    #tf=HFIF_Model.updateStdData(20181127)
    startDate='20181220'#datetime.datetime(2018,12,20)
    endDate='20181221'#datetime.datetime(2018,12,21)
    tf=HFIF_Model.calInduData()
    
    print('Running Ok. Duration in minute: %0.2f minutes'%((time.time() - gtime)/60))