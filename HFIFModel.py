# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 14:15:04 2018
version git1.1
@author: wap
"""

import time,csv,datetime,xlrd,os,math
from keras import models,backend
from keras.layers import Activation,Dense,GRU
import pandas as pd
import numpy as np

#--------------Basic function start-----------

#Basic 1:
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

#Basic 3:
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

#Basic 4:
def getModelName(nGRU,nDense,actFlag):
    return 'model.'+'_'.join(list(map(str,nGRU)))+'.'+\
    '_'.join(list(map(str,nDense)))+'.'+actFlag+'.h5'
    
def getPredictName(nGRU,nDense,actFlag):
    return 'predict.'+'_'.join(list(map(str,nGRU)))+'.'+\
    '_'.join(list(map(str,nDense)))+'.'+actFlag+'.'+datetime.datetime.now().strftime("%Y%m%d")

def buildRNNModel(xShape,arrGRU,arrDense,actFlag='tanh'):
    model = models.Sequential()
    #xShape=data_x[0].shape
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
    model.add(GRU(int(arrGRU[0]),input_shape=xShape,return_sequences=isrese))
    model.add(Activation(actFlag))
    for n in range(1,nGRU):
        if n==nGRU-1:
            isrese=False
        model.add(GRU(int(arrGRU[n]),return_sequences=isrese))
        model.add(Activation(actFlag))
    for n in range(len(arrDense)):
        model.add(Dense(int(arrDense[n])))
        model.add(Activation(actFlag))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
    return model

#Basic 5:
def trainRNNModel(model,xNormData,nDailyData,nx,ny,batchSize=100):
    print('Start fit RNN Model...')
    geneR=[]
    ndd=nDailyData-ny
    nday=int(xNormData.shape[0]/ndd)
    for i in range(nday):
        for j in range(nx,ndd):
            geneR.append(i*ndd+j)
    r = np.random.permutation(geneR) #shuffle
    spb=int(len(r)/batchSize)
    model.fit_generator(generateTrainData(xNormData,nDailyData,
                    nx,ny,r,batchSize),steps_per_epoch=spb, epochs=1)
    
def generateTrainData(xNormData,nDailyData,nx,ny,r,batchSize):
    xData=[]
    yData=[]
    for n in r:
        xData.append(xNormData[(n-nx):n,:-1])
        yData.append(xNormData[n,-1])
        if n%batchSize==batchSize-1:
            xData=np.array(xData)
            yData=np.array(yData)
            yield (xData,yData)
            xData=[]
            yData=[]
    
#Basic 6:
def RNNTest(model,x_test,y_test,testRatePercent=90,judgeRight=0.01):
    predicted = model.predict(x_test).reshape(-1)
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
    return (predicted,np.hstack((arrPredictValue,arrRightRate,arrSumValue)))
        

#--------------Basic function end-------------

#--------------Functionality Start------------

def getPastAveAmnt(pdData,nday,listCode,strSDay='19000101',strEDay='99999999'):
    amntData=pdData.T.values.tolist()
    amntTimes=list(pdData.index)
    amntCodes=list(pdData.columns)
    if len(amntTimes)<=nday:
        return False
    dictPastAveAmnt={}
    for iday in range(nday,len(amntTimes)):
        strDay=amntTimes[iday].replace('-','')
        if len(strDay.split('/'))>1:
            strDay=datetime.datetime.strptime(amntTimes[iday],'%Y/%m/%d').strftime("%Y%m%d")
        if int(strDay)<int(strSDay) or int(strDay)>int(strEDay):
            continue
        dictdailyPastAveAmnt={}
        for icode in range(len(amntCodes)):
            code=amntCodes[icode]
            listamnt=amntData[icode][(iday-nday):iday]
            nCount=0
            sumAmnt=0
            aveAmnt=0
            for amnt in listamnt:
                if type(amnt)==str:
                    amnt=int(amnt.replace(',','').split('.')[0])
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
    ndday=int(xNormData.shape[0]/nDailyData)
    xData=[]
    yData=[]
    for idday in range(ndday):
        for i in range(nx,nDailyData):
            n=(idday-1)*nDailyData+i
            xData.append(xNormData[(n-nx):n,:-1])
            yData.append(xNormData[n,-1])
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
        self.minuteYData=int(arrCfg[6]+0.01)
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
        pass

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
        pdData=pd.read_csv(pdDataFile,header=0,index_col=0,engine='python')
        dictPastAveAmnt={}
        #indu data
        for strDate in listDate:
            nDate=datetime.datetime.strptime(strDate,'%Y%m%d')
            if nDate<sDate or nDate>eDate:
                continue
            #collect nDate's standard data list
            nStdDataPath=os.path.join(stdDataPath,strDate)
            induDataFile=os.path.join(outPath,strDate+'_0.csv')
            if os.path.exists(induDataFile):
                continue
            dictStdData=getDictStdData(nStdDataPath,listCode)
            if len(dictPastAveAmnt)==0:
                dictPastAveAmnt=getPastAveAmnt(pdData,self.nDayAverage,
                        listCode,strDate,listDate[-1])
            for nFlag in dictStdData.keys():
                induDataFile=os.path.join(outPath,strDate+'_'+nFlag+'.csv')
                dictPartStdData=dictStdData[nFlag]
                npDInduData=getDailyInduData(dictPartStdData,
                      self.dictCodeInfo,dictPastAveAmnt[strDate],self.timeSpan)
                np.savetxt(induDataFile,npDInduData,fmt="%.2f",delimiter=',')

#Build 4:
    def calTensorData(self,isTrain=True,strSDate='19000101',strEDate='99990101'):
        print('Start calTensorData...')
        #get or set outpath
        (filepath,tempfilename) = os.path.split(self.cfgFile)
        (filename,extension) = os.path.splitext(tempfilename)
        induDataPath=os.path.join(self.workPath,'induData',filename)
        tempDataPath=os.path.join(self.workPath,'tempData',filename)
        if not os.path.exists(tempDataPath):
            os.makedirs(tempDataPath)
        listInduDataFile=os.listdir(induDataPath)
        ny=int(self.minuteYData*60/self.timeSpan+0.1)
        xData=np.array([])
        intSDate=int(strSDate)
        intEDate=int(strEDate)
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
            else:
                xData=np.vstack((xData,npxData))
        ttFlag='Test'
        if isTrain:
            pclMatrix=np.percentile(np.abs(xData),range(100),axis=0)
            np.savetxt(os.path.join(tempDataPath,'pclMatrix.csv'),pclMatrix,fmt="%.4f",delimiter=',')
            ttFlag='Train'
        else:
            pclMatrix=np.loadtxt(os.path.join(tempDataPath,'pclMatrix.csv'),delimiter=',')
        xNormData=getNormInduData(xData,pclMatrix)
        np.save(os.path.join(tempDataPath,'norm'+ttFlag+'Data.npy'),xNormData)
    
    def _GetModelPath_(self):
        (filepath,tempfilename) = os.path.split(self.cfgFile)
        (filename,extension) = os.path.splitext(tempfilename)
        modelPath=os.path.join(self.workPath,'model',filename)
        if not os.path.exists(modelPath):
            os.makedirs(modelPath)
        return modelPath
    
    def _GetModelFile_(self):
        modelPath=self._GetModelPath_()
        modelName=getModelName(self.nGRU,self.nDense,self.actFunction)
        return os.path.join(modelPath,modelName)
    
    def _GetTempDataPath_(self):
        (filepath,tempfilename) = os.path.split(self.cfgFile)
        (filename,extension) = os.path.splitext(tempfilename)
        tempDataPath=os.path.join(self.workPath,'tempData',filename)
        if not os.path.exists(tempDataPath):
            os.makedirs(tempDataPath)
        return tempDataPath
        
    def _getModelParam_(self):
        ts=self.timeSE[0]
        nDailyData=int((ts[1]-ts[0]).seconds/self.timeSpan)
        nx=int(self.minuteXData*60/self.timeSpan+0.1)
        ny=int(self.minuteYData*60/self.timeSpan+0.1)
        return (nDailyData,nx,ny)
    
    def TrainModel(self):
        print('Start TrainModel...')
        modelfile=self._GetModelFile_()
        tempDataPath=self._GetTempDataPath_()
        xNormData=np.load(os.path.join(tempDataPath,'normTrainData.npy'))
        nDailyData,nx,ny=self._getModelParam_()
        if os.path.exists(modelfile):
            model=models.load_model(modelfile)
        else:
            model=buildRNNModel((nx,xNormData.shape[1]-1),self.nGRU,self.nDense,self.actFunction)
        trainRNNModel(model,xNormData,nDailyData,nx,ny)
        model.save(modelfile)
    
    def savePredictFile(self,listPredict,nGRU,nDense,actFunct):
        predictName=getPredictName(nGRU,nDense,actFunct)
        tempDataPath=self._GetTempDataPath_()
        mfileName=os.path.join(tempDataPath,predictName)
        i=0
        fileName=mfileName+'_'+str(i)+'.csv'
        while os.path.exists(fileName):
            i+=1
            fileName=mfileName+'_'+str(i)+'.csv'
        np.savetxt(os.path.join(tempDataPath,fileName),np.array(listPredict).T,fmt="%.4f",delimiter=',')
    
    def TestModel(self):
        print('Start TestModel...')
        modelfile=self._GetModelFile_()
        model=models.load_model(modelfile)
        tempDataPath=self._GetTempDataPath_()
        normTestData=np.load(os.path.join(tempDataPath,'normTestData.npy'))
        xTest,yTest=getTensorData(normTestData,*self._getModelParam_())
        predict,testResult=RNNTest(model,xTest,yTest)
        fileName='testResult_'+datetime.datetime.now().strftime("%Y%m%d")+'.csv'
        np.savetxt(os.path.join(tempDataPath,fileName),testResult,fmt="%.4f",delimiter=',')
        self.savePredictFile([yTest,predict],self.nGRU,self.nDense,self.actFunction)
        
    def CompareModels(self,rRange=2,nNet=5,nRepeat=10):
        tempDataPath=self._GetTempDataPath_()
        normTestData=np.load(os.path.join(tempDataPath,'normTestData.npy'))
        xTest,yTest=getTensorData(normTestData,*self._getModelParam_())
        xNormData=np.load(os.path.join(tempDataPath,'normTrainData.npy'))
        nDailyData,nx,ny=self._getModelParam_()
        npGRU=np.array(self.nGRU)
        npDense=np.array(self.nDense)
        modelPath=self._GetModelPath_()
        if not os.path.exists(modelPath):
            os.makedirs(modelPath)
        listPredict=[yTest]
        for nn in range(nNet):
            nowGRU=list((npGRU*(3*nn/2/nNet+0.5)).astype(int))
            nowDense=list((npDense*(3*nn/2/nNet+0.5)).astype(int))
            modelName=getModelName(nowGRU,nowDense,self.actFunction)
            modelFile=os.path.join(modelPath,modelName)
            if not os.path.exists(modelFile):
                RNNModel=buildRNNModel((nx,xNormData.shape[1]-1),nowGRU,nowDense,self.actFunction)
            else:
                RNNModel=models.load_model(modelFile)
            print(nowGRU,nowDense)
            for nr in range(nRepeat):
                print(nn,nr)
                trainRNNModel(RNNModel,xNormData,nDailyData,nx,ny)
                listPredict.append(RNNModel.predict(xTest).reshape(-1))
            RNNModel.save(modelFile)
            backend.clear_session()
            self.savePredictFile(listPredict,nowGRU,nowDense,self.actFunction)
            listPredict=[yTest]
        
    def collectAllData(self,strSDate='19000101'):
        self.updateStdData(strSDate)
        self.updateAmntByTick(strSDate)
        self.calInduData(strSDate)
        
#---------------Build HFIF Model End--------

if __name__=='__main__':
    gtime = time.time()
    np.seterr(divide='ignore',invalid='ignore')
    print('Start Running...')
    #build up
    workPath='F:\\草稿\\HFI_Model'
    cfgFile='F:\\草稿\\HFI_Model\\cfg\\cfg_sz50_v331atan.xlsx'
    #cfgFile='F:\\草稿\\HFI_Model\\cfg\\cfg_hs300_v22tan.xlsx'
    if not os.path.exists(workPath):
        workPath='C:\\Users\\WAP\\Documents\\HFI_Model'
        cfgFile='C:\\Users\\WAP\\Documents\\HFI_Model\\cfg\\cfg_sz50_v331atan.xlsx'
    HFIF_Model=AIHFIF(workPath,cfgFile)
    dictCodeInfo=HFIF_Model.dictCodeInfo
    #collect data
    HFIF_Model.collectAllData('20180601')
    #cal
    HFIF_Model.calTensorData(strEDate='20190101')#Train Data
    HFIF_Model.calTensorData(isTrain=False,strSDate='20190101')#Test Data
    HFIF_Model.TrainModel()
    HFIF_Model.TestModel()
    HFIF_Model.CompareModels()
    
    print('\nRunning Ok. Duration in minute: %0.2f minutes'%((time.time() - gtime)/60))