# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 14:15:04 2018
version git1.1
@author: wap
"""

import time,csv,datetime,xlrd,os,math
from keras.models import Sequential
from keras.layers import Activation,Dense,GRU
import pandas as pd
import numpy as np

#--------------Basic function start-----------

def csvToList(csvFile):
    resultList=[]
    with open(csvFile, mode='r') as f:
        csvReader = csv.reader(f)
        for row in csvReader:
            resultList.append(list(map(eval,row)))
    return resultList

def listToCsv(listData,csvFilePath):
    with open(csvFilePath, mode='w', newline='') as f:
        csvwriter = csv.writer(f)
        for row in listData:
            csvwriter.writerow(row)

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
    
def intToDate(intDate):
    return datetime.datetime.strptime(str(intDate),'%Y%m%d')

def strToDate(strDate):
    return datetime.datetime.strptime(strDate,'%Y%m%d')

def calPercentile(xValue,arrPercentile): #len(arrPercentile)=100,upscane
    isfind=False
    abv=abs(xValue)
    for i in range(100):
        if abv<(arrPercentile[i]+0.0001):
            isfind=True
            break
    if isfind:
        result=i/100
    else:
        result=1
    return result*np.sign(xValue)

def buildRNNModel(data_x,data_y,arrGRU=0,arrDense=0,actFlag='tanh'):
    model = Sequential()
    xShape=data_x[0].shape
    if arrGRU==0:
        arrGRU=int(math.sqrt(xShape[0]*xShape[1]))
    if arrDense==0:
        arrDense=int(arrGRU*0.6)
    if type(arrGRU)!=list:
        arrGRU=[arrGRU]
    if type(arrDense)!=list:
        arrDense=[arrDense]
    nGRU=len(arrGRU)
    isrese=False
    if nGRU>1:
        isrese=True
    model.add(GRU(arrGRU[0],input_shape=xShape,
                  return_sequences=isrese))
    model.add(Activation(actFlag))
    for n in range(1,nGRU):
        if n==nGRU-1:
            isrese=False
        model.add(GRU(arrGRU[n],return_sequences=isrese))
        model.add(Activation(actFlag))
    for n in range(len(arrDense)):
        model.add(Dense(arrDense[n]))
        model.add(Activation(actFlag))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
    r = np.random.permutation(len(data_x)) #shuffle
    sdata_x=data_x[r,:]
    sdata_y=data_y[r]
    print('Start fit RNN Model...')
    model.fit(sdata_x,sdata_y,batch_size=128,epochs=1,validation_split=0.05)
    return model

def RNNTest(model,x_test,y_test,testRatePercent=80,judgeRight=0.01):
    predicted = model.predict(x_test).reshape(-1)
    pcl=np.percentile(np.abs(predicted),range((100-testRatePercent),100))
    arrSumValue=np.zeros([testRatePercent,2])
    arrSumRight=np.zeros([testRatePercent,2])
    arrSumN=np.zeros([testRatePercent,2])
    for idata in range(len(predicted)):
        for itrp in range(testRatePercent):
            if predicted[idata]>pcl[itrp]:
                arrSumN[itrp,0]+=1
                if y_test[idata]>judgeRight:
                    arrSumRight[itrp,0]+=1
                    arrSumValue[itrp,0]+=y_test[idata]
            if predicted[idata]<-pcl[itrp]:
                arrSumN[itrp,1]+=1
                if y_test[idata]<-judgeRight:
                    arrSumRight[itrp,1]+=1
                    arrSumValue[itrp,1]+=y_test[idata]
    arrPredictValue=arrSumValue/arrSumN
    arrRightRate=arrSumRight/arrSumN
    return np.hstack((arrPredictValue,arrRightRate))
        

#--------------Basic function end-------------

#--------------Functionality Start------------

def getPdWIndAmnt(nday,listCode,strSDay,strEDay=''):
    from WindPy import w
    if w.isconnected()==False:
        msg=w.start(waitTime=15)
        if msg.ErrorCode!=0:
            return False
    msg=w.tdaysoffset(-nday, strSDay, "")
    if msg.ErrorCode!=0:
        return False
    amntSDay=msg.Data[0][0].strftime('%Y%m%d')
    strCode=','.join(listCode)
    amntEDay=''
    if strEDay!='':
        amntEDay=strEDay
    msg=w.wsd(strCode, 'amt', amntSDay, amntEDay, "")
    if msg.ErrorCode!=0:
        return False
    return pd.DataFrame(np.array(msg.Data).T,index=msg.Times,columns=msg.Codes)

def getPastAveAmnt(pdData,nday,listCode,strSDay,strEDay=''):
    amntData=list(pdData.T.values)
    amntTimes=list(pdData.index)
    amntCodes=list(pdData.columns)
    if len(amntTimes)<=nday:
        return False
    dictPastAveAmnt={}
    for iday in range(nday,len(amntTimes)):
        strDay=amntTimes[iday].replace('-','')
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

#--------------Functionality end--------------

#----------Model step function start----------

#step 0
def CheckRawData(rawData):
    if len(rawData)<100:
        return 1
    pass
    return 0

#step 1
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

#step 2
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

#step3
def getDailyTensorData(npDailyInduData,indexData,nx,ny):
    xData=[]
    yData=[]
    for i in range(nx,len(indexData)-ny):
        xData.append(npDailyInduData[(i-nx):(i-1),:])
        yData.append((indexData[i+ny]/indexData[i]-1)*10000/ny)
    xData=np.array(xData)
    yData=np.array(yData)
    return xData,yData

#step3
def getNormInduData(xData,pclMatrix):
    xShape=xData.shape
    normInduData=np.zeros(xShape)
    for j in range(xShape[1]):
        arrPercentile=pclMatrix[:,j]
        for i in range(xShape[0]):
            normInduData[i,j]=calPercentile(xData[i,j],arrPercentile)
    return normInduData

#----------Model step function end------------

#-----------Build HFIF Model Class------------

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
    
    def updateStdData(self,intStartDate=0):
        print('Start updateStdData...')
        mainOutPath=os.path.join(self.workPath,'StandardData',str(int(self.timeSpan))+'_sec_span')
        pathList = os.listdir(self.rawDataPath)
        listCode=list(self.dictCodeInfo.keys())
        listCode.append(self.indexCode)
        listStdData=[]
        for path in pathList:
            if int(path)<intStartDate:
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
        #return listStdData
    
    def updateWindDailyAmntData(self,strSDate='19000101'):
        print('Start updateWindDailyAmntData...')
        dataFile=os.path.join(self.workPath,'DailyAmntData','DailyAmntData.csv')
        listRawDataDate=os.listdir(self.rawDataPath)
        startDate=listRawDataDate[0]
        if int(strSDate)>int(startDate):
            startDate=strSDate
        listCode=list(self.dictCodeInfo.keys())
        nday=self.nDayAverage
        pdData=getPdWIndAmnt(nday,listCode,startDate)
        pdData.to_csv(dataFile)
        #return pdData
    
    def calInduData(self,strSDate='19000101',strEDate=''):
        print('Start calInduData...')
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
        pdDataFile=os.path.join(self.workPath,'DailyAmntData','DailyAmntData.csv')
        pdData=pd.read_csv(pdDataFile,header=0,index_col=0)
        dictPastAveAmnt=getPastAveAmnt(pdData,self.nDayAverage,listCode,
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
                
                np.savetxt(induDataFile,npDInduData,fmt="%.2f",delimiter=',')
        return npDInduData
    
    def calTensorData(self,pclMatrix=[],strStartDate='0',strEndDate='99999999'):
        print('Start calTensorData...')
        #get or set outpath
        (filepath,tempfilename) = os.path.split(self.cfgFile)
        (filename,extension) = os.path.splitext(tempfilename)
        induDataPath=os.path.join(self.workPath,'induData',filename)
        listInduDataFile=os.listdir(induDataPath)
        nx=int(self.minuteXData*60/self.timeSpan+0.1)
        ny=int(self.minuteYData*60/self.timeSpan+0.1)
        xData=np.array([])
        intSDate=int(strStartDate)
        intEDate=int(strEndDate)
        for induDataFile in listInduDataFile:
            fpath=os.path.join(induDataPath,induDataFile)
            (filename,extension) = os.path.splitext(induDataFile)
            nameInfo=filename.split('_')
            intDate=int(nameInfo[0])
            if intDate<intSDate or intDate>intEDate:
                continue
            npxData=np.loadtxt(fpath,delimiter=',')[:-ny]
            yFilePath=os.path.join(self.workPath,'StandardData',
                str(int(self.timeSpan))+'_sec_span',nameInfo[0],
                self.indexCode+'_'+nameInfo[1]+'.csv')
            npyData=np.loadtxt(yFilePath,delimiter=',')[1:,0]
            npyData=(npyData[ny:]/npyData[:-ny]-1)*10000
            npxData=np.hstack((npxData,npyData.reshape((-1,1))))
            if xData.size==0:
                xData=npxData
                nDailyData=npxData.shape[0]
            else:
                xData=np.vstack((xData,npxData))
        if len(pclMatrix)==0:
            pclMatrix=np.percentile(np.abs(xData),range(100),axis=0)
        xNormData=getNormInduData(xData,pclMatrix)
        ndday=int(xNormData.shape[0]/nDailyData)
        xData=[]
        yData=[]
        for idday in range(ndday):
            for i in range(nx,nDailyData):
                n=(idday-1)*nDailyData+i
                xData.append(xNormData[(n-nx):n,:-1])
                yData.append(xNormData[n,-1])
        return (np.array(xData),np.array(yData),pclMatrix)
    
    def collectAllData(self):
        self.updateStdData()
        self.updateWindDailyAmntData()
        self.calInduData()
        
#---------------Build HFIF Model End--------

if __name__=='__main__':
    gtime = time.time()
    np.seterr(divide='ignore',invalid='ignore')
    print('Start Running...')
    #build up
    workPath='C:\\Users\\WAP\\Documents\\HFI_Model'
    cfgFile='C:\\Users\\WAP\\Documents\\HFI_Model\\cfg\\cfg_sz50_v331atan.xlsx'
    HFIF_Model=AIHFIF(workPath,cfgFile)
    dictCodeInfo=HFIF_Model.dictCodeInfo
    #collect data
    #HFIF_Model.collectAllData
    (xTrain,yTrain,pcl)=HFIF_Model.calTensorData(strEndDate='20190102')
    RNNModel=buildRNNModel(xTrain,yTrain)
    (xTest,yTest,pcl)=HFIF_Model.calTensorData(pclMatrix=pcl,strStartDate='20190103')
    testResult=RNNTest(RNNModel,xTest,yTest)
    
    print('\nRunning Ok. Duration in minute: %0.2f minutes'%((time.time() - gtime)/60))