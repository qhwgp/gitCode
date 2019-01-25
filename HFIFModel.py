# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 14:15:04 2018
version HFIF_v2.0
@author: wap
"""

import time,csv,datetime,xlrd,os#,math
from keras import models,backend,metrics
from keras.layers import GRU
import pandas as pd
import numpy as np

#--------------Basic function start-----------

#Basic 1:
def toInt(fv):
    return int(fv+0.5)

def csvToList(csvFile):
    resultList=[]
    try:
        with open(csvFile, mode='r') as f:
            csvReader = csv.reader(f)
            for row in csvReader:
                resultList.append(list(map(eval,row)))
    except:
        pass
    return resultList

def listToCsv(listData,csvFilePath):
    with open(csvFilePath, mode='w', newline='') as f:
        csvwriter = csv.writer(f)
        for row in listData:
            csvwriter.writerow(row)

#Basic 2:
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

def strToInt(strDate):
    if '-' in strDate:
        lDate=strDate.split('-')
    else:
        lDate=strDate.split('/')
    return int(lDate[0])*10000+int(lDate[1])*100+int(lDate[2])

#Basic 3:
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

def calPcl(xValue,arrPercentile,rng=[0,99]):
    if xValue<0:
        return -calPcl(-xValue,arrPercentile)
    if rng[1]-rng[0]<=1 or xValue>arrPercentile[rng[1]]:
        return rng[1]
    else:
        md=int(np.mean(rng))
        if xValue<=arrPercentile[md]:
            return calPcl(xValue,arrPercentile,rng=[rng[0],md])
        else:
            return calPcl(xValue,arrPercentile,rng=[md,rng[1]])

#Basic 4:
def getSaveName(fStr,nGRU,nDense,actFlag):
    try:
        strnGRU='_'.join(list(map(str,nGRU)))
    except:
        strnGRU=str(nGRU)
    try:
        strnDense='_'.join(list(map(str,nDense)))
    except:
        strnDense=str(nDense)
    return fStr+strnGRU+'.'+strnDense+'.'+actFlag+'.'+\
            datetime.datetime.now().strftime("%Y%m%d")
    
def getTestResultName(nGRU,nDense,actFlag):
    return 'testResult.'+'_'.join(list(map(str,nGRU)))+'.'+\
    '_'.join(list(map(str,nDense)))+'.'+actFlag+'.'+datetime.datetime.now().strftime("%Y%m%d")

def myLoss(y_true, y_pred):
    return backend.mean(backend.square((y_pred - y_true)*y_true), axis=-1)

def buildRNNModel(xShape,actFlag='tanh'):
    model = models.Sequential()
    model.add(GRU(1,input_shape=xShape,activation=actFlag,recurrent_activation=actFlag,
                  dropout=0.1,recurrent_dropout=0.1,return_sequences=False))
    model.compile(loss=myLoss, optimizer="rmsprop",
                  metrics=[metrics.mean_squared_error])
    return model

#Basic 5:
def trainRNNModel(model,xNormData,nDailyData,nx,ny,iy,cRate=1,batchSize=100):
    print('Start fit RNN Model...')
    geneR=[]
    ndd=nDailyData-ny[-1]-1
    nday=int(xNormData.shape[0]/ndd)
    cyValue=np.percentile(np.abs(xNormData[:,-(len(ny)-iy)]),(1-cRate)*100)
    for i in range(nday):
        for j in range(nx,ndd):
            if xNormData[i*ndd+j,iy-len(ny)]>=cyValue or xNormData[i*ndd+j,iy-len(ny)]<=-cyValue:
                geneR.append(i*ndd+j)
    r = np.random.permutation(geneR) #shuffle
    spb=int(len(r)/batchSize)
    model.fit_generator(generateTrainData(xNormData,nDailyData,
                    nx,ny,iy,r,batchSize),steps_per_epoch=spb, epochs=1)
    
def generateTrainData(xNormData,nDailyData,nx,ny,iy,r,batchSize):
    xData=[]
    yData=[]
    for n in r:
        xData.append(xNormData[(n-nx):n,:-len(ny)])
        yData.append(xNormData[n,iy-len(ny)])
        if n%batchSize==batchSize-1:
            xData=np.array(xData)
            yData=np.array(yData)
            yield (xData,yData)
            xData=[]
            yData=[]
    
#Basic 6:
def RNNTest(listModel,x_test,y_test,testRatePercent=90,judgeRight=0.01):
    npPredict=y_test
    npTestResult=np.array([])
    for i in range(len(listModel)):
        model=listModel[i]
        predicted = model.predict(x_test).reshape(-1,1)
        testResult=testPredict(predicted,y_test[:,i],testRatePercent,judgeRight)
        npPredict=np.hstack((npPredict,predicted))
        if npTestResult.size==0:
            npTestResult=testResult
        else:
            npTestResult=np.hstack((npTestResult,testResult))
    return (npPredict,npTestResult)

def testPredict(predicted,y_test,testRatePercent=90,judgeRight=0.01):
    pcl=np.percentile(np.abs(predicted),range((100-testRatePercent),testRatePercent))
    nl=2*testRatePercent-100
    arrSumValue=np.zeros([nl,2])
    arrSumRight=np.zeros([nl,2])
    arrSumN=np.zeros([nl,2])
    for idata in range(len(predicted)):
        for itrp in range(nl):
            if predicted[idata]>pcl[itrp]:
                arrSumN[itrp,0]+=1
                arrSumValue[itrp,0]+=y_test[idata]
                if y_test[idata]>judgeRight:
                    arrSumRight[itrp,0]+=1
            if predicted[idata]<-pcl[itrp]:
                arrSumN[itrp,1]+=1
                if y_test[idata]<-judgeRight:
                    arrSumRight[itrp,1]+=1
                    arrSumValue[itrp,1]+=y_test[idata]
    arrPredictValue=arrSumValue/arrSumN
    arrRightRate=arrSumRight/arrSumN
    return np.hstack((arrPredictValue,arrRightRate,arrSumN,arrSumValue))

#--------------Basic function end-------------

#--------------Functionality Start------------

#Func 1:
def getPastAveAmnt(dictDailyAmnt,nday,listCode,strSDay='19000101',strEDay='99990101'):
    dictPastAveAmnt={}
    intSDay=int(strSDay)
    intEDay=int(strEDay)
    for aCode,pdAmntData in dictDailyAmnt.items():
        for intDate in pdAmntData.index:
            if intDate<intSDay or intDate>intEDay:
                continue
            pdData=pdAmntData[pdAmntData.index<intDate]
            if len(pdData)<nday:
                avgAmnt=0
            else:
                pdData=pdData.tail(nday)
                ma=pdData.Amnt.max()
                mi=pdData.Amnt.min()
                pdData=pdData[(pdData.Amnt<ma)&(pdData.Amnt>mi)]
                avgAmnt=pdData['Amnt'].mean()
            if not intDate in dictPastAveAmnt:
                dictPastAveAmnt[intDate]={}
            dictPastAveAmnt[intDate][aCode]=avgAmnt
    return dictPastAveAmnt

#Func 2:
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

#Func 3:
def getDictAmntData(amntDataPath,listCode):
    dictDailyAmnt={}
    for aCode in listCode:
        fileName=os.path.join(amntDataPath,aCode+'.csv')
        if os.path.exists(fileName):
            pdAmntData=pd.read_csv(fileName,header=0,index_col=0,engine='python')
        else:
            pdAmntData=pd.DataFrame(columns=['Amnt'])
        dictDailyAmnt[aCode]=pdAmntData
    return dictDailyAmnt

#--------------Functionality end--------------

#----------Model step function start----------

#step 0
def CheckRawData(rawData):
    if len(rawData)<500:
        return 1
    pass
    return 0

#step 1
def tickToStdData(rawTickData,ListtimeSE,secTimeDiff):
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
def getTensorData(xNormData,nDailyData,nx,ny):
    lenDData=nDailyData-ny[-1]-1
    ndday=int(xNormData.shape[0]/lenDData)
    xData=[]
    yData=[]
    for idday in range(ndday):
        for i in range(nx,lenDData):
            n=idday*lenDData+i
            xData.append(xNormData[(n-nx):n,:-len(ny)])
            yData.append(xNormData[n,-len(ny):])
    xData=np.array(xData)
    yData=np.array(yData)
    return (xData,yData)

#step3
def getNormInduData(xData,pclMatrix):
    xShape=xData.shape
    normInduData=np.zeros(xShape)
    for j in range(xShape[1]):
        arrPercentile=pclMatrix[:,j]
        for i in range(xShape[0]):
            normInduData[i,j]=calPercentile(xData[i,j],arrPercentile)
            #normInduData[i,j]=calPcl(xData[i,j],arrPercentile)
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
        self.nGRU=0
        self.nDense=0
        self.actFunction=''
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
        self.minuteYData=list(map(int,arrCfg[6].split(',')))
        if type(arrCfg[7])==float:
            self.nGRU=[int(arrCfg[7])]
        else:
            self.nGRU=list(map(int,arrCfg[7].split(',')))
        if type(arrCfg[8])==float:
            self.nDense=[int(arrCfg[8])]
        else:
            self.nDense=list(map(int,arrCfg[8].split(',')))
        self.actFunction=arrCfg[9]

#Build 1:  
    def updateStdData(self,strSDate='19000101',strEDate='99999999'):
        print('Start updateStdData...')
        mainOutPath=os.path.join(self.workPath,'StandardData',str(int(self.timeSpan))+'_sec_span')
        listCode=list(self.dictCodeInfo.keys())
        listCode.append(self.indexCode)
        listStdData=[]
        intStartDate=int(strSDate)
        intEndDate=int(strEDate)
        for pathDate in os.listdir(self.rawDataPath):
            intPathDate=int(pathDate)
            if intPathDate<intStartDate or intPathDate>intEndDate:
                continue
            datePath=os.path.join(self.rawDataPath,pathDate)
            for code in listCode:
                outPath=os.path.join(mainOutPath,pathDate)
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

#Build 2:
    def updateAmntByTick(self,strSDate='19000101'):
        print('Start updateAmntByTick...')
        pdDataFile=os.path.join(self.workPath,'DailyAmntData','DailyAmntData.csv')
        pdData=pd.read_csv(pdDataFile,header=0,index_col=0,engine='python')
        amntDays=list(pdData.index)
        amntCodes=list(pdData.columns)
        intSDate=int(strSDate)
        for datePath in os.listdir(self.rawDataPath):
            if int(datePath)<intSDate:
                continue
            pdDate=datetime.datetime.strptime(datePath,'%Y%m%d').strftime("%Y/%m/%d").replace('/0','/')
            if not pdDate in amntDays:
                listAmnt=[]
                for pdCode in amntCodes:
                    rawDataFile=os.path.join(self.rawDataPath,datePath,pdCode+'.csv')
                    amnt=0
                    if os.path.exists(rawDataFile):
                        amnt=np.sum(np.loadtxt(rawDataFile,delimiter=',')[:,2])
                    listAmnt.append(amnt)
                pdData.loc[pdDate] = listAmnt
        pdData.sort_index()
        pdData.to_csv(pdDataFile)           
    
    def updateAmntByRaw(self,strSDate='19000101',strEDate='99990101'):
        print('Start updateAmntByRaw...')
        outPath=os.path.join(self.workPath,'DailyAmntData')
        listCode=list(self.dictCodeInfo.keys())
        dictDailyAmnt=getDictAmntData(outPath,listCode)
        intStartDate=int(strSDate)
        intEndDate=int(strEDate)
        for pathDate in os.listdir(self.rawDataPath):
            intPathDate=int(pathDate)
            if intPathDate<intStartDate or intPathDate>intEndDate:
                continue
            dPath=os.path.join(self.rawDataPath,pathDate)
            for aCode in listCode:
                if not intPathDate in dictDailyAmnt[aCode].index:
                    rawDataFile=os.path.join(dPath,aCode+'.csv')
                    try:
                        amnt=np.sum(np.loadtxt(rawDataFile,delimiter=',')[:,2])
                    except:
                        amnt=0
                    if amnt>1:
                        dictDailyAmnt[aCode].loc[intPathDate]=amnt
        for aCode in listCode:
            pdData=dictDailyAmnt[aCode].sort_index()
            pdData.to_csv(os.path.join(outPath,aCode+'.csv'))  

    def updateAmntByWnd(self,wndFileName,strSDate='19000101',strEDate='99990101'):
        print('Start updateAmntByWnd...')
        outPath=os.path.join(self.workPath,'DailyAmntData')
        setCode=set(self.dictCodeInfo.keys())
        dictDailyAmnt=getDictAmntData(outPath,setCode)
        intStartDate=int(strSDate)
        intEndDate=int(strEDate)
        pdDataFile=os.path.join(self.workPath,wndFileName)
        pdData=pd.read_csv(pdDataFile,header=0,index_col=0,engine='python')
        amntDays=list(pdData.index)
        amntCodes=set(pdData.columns)
        for aDay in amntDays:
            intDate=strToInt(aDay)
            if intDate<intStartDate or intDate>intEndDate:
                continue
            for aCode in setCode&amntCodes:
                if not intDate in dictDailyAmnt[aCode].index and aCode in amntCodes:
                    amnt=pdData.loc[aDay,aCode]
                    if amnt>1:
                        dictDailyAmnt[aCode].loc[intDate]=amnt
        for aCode in setCode:
            pdData=dictDailyAmnt[aCode].sort_index()
            pdData.to_csv(os.path.join(outPath,aCode+'.csv')) 

#Build 3:  
    def calInduData(self,strSDate='19000101',strEDate='99990101'):
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
        intLSD=int(listDate[0])
        intLED=int(listDate[-1])
        intSDate=int(strSDate)
        if intSDate<intLSD:
            intSDate=intLSD
        intEDate=int(strEDate)
        if intEDate>intLED:
            intEDate=intLED
        #past nday's average amount
        listCode=list(self.dictCodeInfo.keys())
        #pdDataFile=os.path.join(self.workPath,'DailyAmntData','DailyAmntData.csv')
        #pdData=pd.read_csv(pdDataFile,header=0,index_col=0,engine='python')
        daPath=os.path.join(self.workPath,'DailyAmntData')
        dictDailyAmnt=getDictAmntData(daPath,listCode)
        dictPastAveAmnt={}
        #indu data
        for strDate in listDate:
            intDate=int(strDate)
            #nDate=datetime.datetime.strptime(strDate,'%Y%m%d')
            if intDate<intSDate or intDate>intEDate:
                continue
            #collect nDate's standard data list
            nStdDataPath=os.path.join(stdDataPath,strDate)
            induDataFile=os.path.join(outPath,strDate+'_0.csv')
            if os.path.exists(induDataFile):
                continue
            dictStdData=getDictStdData(nStdDataPath,listCode)
            if len(dictPastAveAmnt)==0:
                dictPastAveAmnt=getPastAveAmnt(dictDailyAmnt,self.nDayAverage,
                        listCode,strDate,strEDate)
            for nFlag in dictStdData.keys():
                induDataFile=os.path.join(outPath,strDate+'_'+nFlag+'.csv')
                dictPartStdData=dictStdData[nFlag]
                npDInduData=getDailyInduData(dictPartStdData,
                      self.dictCodeInfo,dictPastAveAmnt[intDate],self.timeSpan)
                np.savetxt(induDataFile,npDInduData,fmt="%.4f",delimiter=',')

#Build 4:
    def calTensorData(self,isTrain=True,strSDate='19000101',strEDate='99990101'):
        print('Start calTensorData...')
        #get or set outpath
        (filepath,tempfilename) = os.path.split(self.cfgFile)
        (filename,extension) = os.path.splitext(tempfilename)
        induDataPath=os.path.join(self.workPath,'induData',filename)
        normDataPath=os.path.join(self.workPath,'normData',filename)
        modelPath=os.path.join(self.workPath,'cfg',filename)
        if not os.path.exists(normDataPath):
            os.makedirs(normDataPath)
        if not os.path.exists(modelPath):
            os.makedirs(modelPath)
        listInduDataFile=os.listdir(induDataPath)
        nDailyData,nx,ny=self._getModelParam_()
        xData=np.array([])
        intSDate=int(strSDate)
        intEDate=int(strEDate)
        lenDData=nDailyData-ny[-1]-1
        for induDataFile in listInduDataFile:
            fpath=os.path.join(induDataPath,induDataFile)
            (ifilename,extension) = os.path.splitext(induDataFile)
            nameInfo=ifilename.split('_')
            intDate=int(nameInfo[0])
            if intDate<intSDate or intDate>intEDate:
                continue
            npxData=np.loadtxt(fpath,delimiter=',')[:lenDData]
            yFilePath=os.path.join(self.workPath,'StandardData',
                str(int(self.timeSpan))+'_sec_span',nameInfo[0],
                self.indexCode+'_'+nameInfo[1]+'.csv')
            npyData=np.loadtxt(yFilePath,delimiter=',')[2:,0]
            yData=np.array([])
            for iy in range(len(ny)):
                tempY=(npyData[ny[iy]:]/npyData[:-ny[iy]]-1)*10000
                tempY=tempY[:lenDData].reshape((-1,1))
                if yData.size==0:
                    yData=tempY
                else:
                    yData=np.hstack((yData,tempY))
            npxData=np.hstack((npxData,yData))
            if xData.size==0:
                xData=npxData
            else:
                try:
                    xData=np.vstack((xData,npxData))
                except:
                    print('induDataFile error: '+induDataFile)
        ttFlag='Test'
        if isTrain:
            pclMatrix=np.percentile(np.abs(xData),range(100),axis=0)
            np.savetxt(os.path.join(modelPath,'pclMatrix_'+filename+'.csv'),pclMatrix,fmt="%.4f",delimiter=',')
            ttFlag='Train'
        else:
            pclMatrix=np.loadtxt(os.path.join(modelPath,'pclMatrix_'+filename+'.csv'),delimiter=',')
        ttime = time.time()
        xNormData=getNormInduData(xData,pclMatrix)
        print('getNormInduData minute: %0.2f minutes'%((time.time() - ttime)/60))
        np.save(os.path.join(normDataPath,'norm'+ttFlag+'Data.npy'),xNormData)

    def _GetModelFile_(self):
        (filepath,tempfilename) = os.path.split(self.cfgFile)
        (filename,extension) = os.path.splitext(tempfilename)
        modelPath=os.path.join(self.workPath,'cfg',filename)
        if not os.path.exists(modelPath):
            os.makedirs(modelPath)
        modelName='model_'+filename+'_'+str(self.minuteXData)+'min_'
        listModelFile=[]
        ny=self.minuteYData
        for i in range(len(ny)):
            listModelFile.append(os.path.join(modelPath,
                            modelName+str(ny[i])+'min.h5'))
        return listModelFile
    
    def _GetTempDataPath_(self):
        (filepath,tempfilename) = os.path.split(self.cfgFile)
        (filename,extension) = os.path.splitext(tempfilename)
        tempDataPath=os.path.join(self.workPath,'tempData',filename)
        if not os.path.exists(tempDataPath):
            os.makedirs(tempDataPath)
        return tempDataPath
    
    def _GetNormDataPath_(self):
        (filepath,tempfilename) = os.path.split(self.cfgFile)
        (filename,extension) = os.path.splitext(tempfilename)
        normDataPath=os.path.join(self.workPath,'normData',filename)
        if not os.path.exists(normDataPath):
            os.makedirs(normDataPath)
        return normDataPath
        
    def _getModelParam_(self):
        ts=self.timeSE[0]
        nDailyData=int((ts[1]-ts[0]).seconds/self.timeSpan)
        nx=toInt(self.minuteXData*60/self.timeSpan)
        ny=np.array(self.minuteYData)*toInt(60/self.timeSpan)
        return (nDailyData,nx,ny)
    
    def TrainModel(self,crate=1):
        listModelFile=self._GetModelFile_()
        normDataPath=self._GetNormDataPath_()
        xNormData=np.load(os.path.join(normDataPath,'normTrainData.npy'))
        nDailyData,nx,ny=self._getModelParam_()
        for iy in range(len(ny)):
            modelfile=listModelFile[iy]
            if os.path.exists(modelfile):
                print('Load TrainModel...')
                model=models.load_model(modelfile,custom_objects={'myLoss': myLoss})
            else:
                print('Start TrainModel...')
                model=buildRNNModel((nx,xNormData.shape[1]-len(ny)),self.actFunction)
            trainRNNModel(model,xNormData,nDailyData,nx,ny,iy,cRate=crate)
            model.save(modelfile)
    
    def saveTempFile(self,fStr,npPredict,nGRU,nDense,actFunct):
        predictName=getSaveName(fStr,nGRU,nDense,actFunct)
        tempDataPath=self._GetTempDataPath_()
        mfileName=os.path.join(tempDataPath,predictName)
        i=0
        fileName=mfileName+'_'+str(i)+'.csv'
        while os.path.exists(fileName):
            i+=1
            fileName=mfileName+'_'+str(i)+'.csv'
        np.savetxt(os.path.join(tempDataPath,fileName),npPredict,fmt="%.4f",delimiter=',')
        
    def saveTestResultFile(self,listTestResult,nGRU,nDense,actFunct):
        testResultName=getTestResultName(nGRU,nDense,actFunct)
        tempDataPath=self._GetTempDataPath_()
        mfileName=os.path.join(tempDataPath,testResultName)
        i=0
        fileName=mfileName+'_'+str(i)+'.csv'
        while os.path.exists(fileName):
            i+=1
            fileName=mfileName+'_'+str(i)+'.csv'
        nptr=listTestResult[0]
        if len(listTestResult)>0:
            for i in range(1,len(listTestResult)):
                nptr=np.hstack((nptr,listTestResult[i]))
        np.savetxt(os.path.join(tempDataPath,fileName),nptr,fmt="%.4f",delimiter=',')
    
    def TestModel(self):
        print('Start TestModel...')
        listModelfile=self._GetModelFile_()
        listModel=[]
        for modelfile in listModelfile:
            listModel.append(models.load_model(modelfile,custom_objects={'myLoss': myLoss}))
        normDataPath=self._GetNormDataPath_()
        normTestData=np.load(os.path.join(normDataPath,'normTestData.npy'))
        xTest,yTest=getTensorData(normTestData,*self._getModelParam_())
        predict,testResult=RNNTest(listModel,xTest,yTest)
        self.saveTempFile('testResult.',testResult,self.minuteXData,self.minuteYData,self.actFunction)
        self.saveTempFile('predict.',predict,self.minuteXData,self.minuteYData,self.actFunction)
        
    def collectAllData(self,strSDate='19000101',strEDate='99990101'):
        self.updateStdData(strSDate,strEDate)
        self.updateAmntByRaw(strSDate,strEDate)
        self.calInduData(strSDate,strEDate)#minus one row
        
#---------------Build HFIF Model End--------

if __name__=='__main__':
    gtime = time.time()
    #np.seterr(divide='ignore',invalid='ignore')
    print('Start Running...')
    #build up
    workPath='F:\\草稿\\HFI_Model'
    #cfgFile='F:\\草稿\\HFI_Model\\cfg\\cfg_sz50_v331atan.xlsx'
    cfgFile='F:\\草稿\\HFI_Model\\cfg\\cfg_hs300_v22tan.xlsx'
    #cfgFile='F:\\草稿\\HFI_Model\\cfg\\cfg_zz500_v11tan.xlsx'
    if not os.path.exists(workPath):
        workPath='C:\\Users\\WAP\\Documents\\HFI_Model'
        #cfgFile='C:\\Users\\WAP\\Documents\\HFI_Model\\cfg\\cfg_hs300_v22tan.xlsx'
        cfgFile='C:\\Users\\WAP\\Documents\\HFI_Model\\cfg\\cfg_sz50_v331atan.xlsx'
    HFIF_Model=AIHFIF(workPath,cfgFile)
    dictCodeInfo=HFIF_Model.dictCodeInfo
    #collect data
    #HFIF_Model.collectAllData()
    #cal
    
    #HFIF_Model.calTensorData(strEDate='20190105')#Train Data,minus len(yTimes) rows
    #HFIF_Model.calTensorData(isTrain=False,strSDate='20190106')#Test Data
    for i in range(20):
        HFIF_Model.TrainModel()
        HFIF_Model.TestModel()
    print('\nRunning Ok. Duration in minute: %0.2f minutes'%((time.time() - gtime)/60))