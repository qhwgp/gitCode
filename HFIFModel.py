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
#from WindPy import w

#--------------Basic function start-----------

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

def getModelName(nGRU,nDense,actFlag):
    return 'model.'+'_'.join(list(map(str,nGRU)))+'.'+\
    '_'.join(list(map(str,nDense)))+'.'+actFlag+'.h5'

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

def trainRNNModel(model,xTrain,yTrain):
    r = np.random.permutation(len(xTrain)) #shuffle
    xTrain=xTrain[r,:]
    yTrain=yTrain[r]
    print('Start fit RNN Model...')
    model.fit(xTrain,yTrain,batch_size=128,epochs=1,validation_split=0.05)

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
"""
def getPdWIndAmnt(nday,listCode,strSDay,strEDay=''):
    
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
"""
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
        self.nGRU=list(map(int,arrCfg[7].split(',')))
        self.nDense=list(map(int,arrCfg[8].split(',')))
        self.actFunction=arrCfg[9]
    
    def updateStdData(self,intStartDate=0):
        print('Start updateStdData...')
        mainOutPath=os.path.join(self.workPath,'StandardData',str(int(self.timeSpan))+'_sec_span')
        listCode=list(self.dictCodeInfo.keys())
        listCode.append(self.indexCode)
        listStdData=[]
        for path in os.listdir(self.rawDataPath):
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
    """
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
    """ 
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
    
    def calTensorData(self,isTrain=True,strStartDate='0',strEndDate='99999999'):
        print('Start calTensorData...')
        #get or set outpath
        (filepath,tempfilename) = os.path.split(self.cfgFile)
        (filename,extension) = os.path.splitext(tempfilename)
        induDataPath=os.path.join(self.workPath,'induData',filename)
        tempDataPath=os.path.join(self.workPath,'tempData',filename)
        if not os.path.exists(tempDataPath):
            os.makedirs(tempDataPath)
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
        ttFlag='Test'
        if isTrain:
            pclMatrix=np.percentile(np.abs(xData),range(100),axis=0)
            np.savetxt(os.path.join(tempDataPath,'pclMatrix.csv'),pclMatrix,fmt="%.4f",delimiter=',')
            ttFlag='Train'
        else:
            pclMatrix=np.loadtxt(os.path.join(tempDataPath,'pclMatrix.csv'),delimiter=',')
        xNormData=getNormInduData(xData,pclMatrix)
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
        np.save(os.path.join(tempDataPath,'x'+ttFlag+'.npy'),xData)
        np.save(os.path.join(tempDataPath,'y'+ttFlag+'.npy'),yData)
        #return (np.array(xData),np.array(yData),pclMatrix)
    
    def _GetModelFile_(self):
        (filepath,tempfilename) = os.path.split(self.cfgFile)
        (filename,extension) = os.path.splitext(tempfilename)
        modelPath=os.path.join(self.workPath,'model')
        if not os.path.exists(modelPath):
            os.makedirs(modelPath)
        modelName=getModelName(self.nGRU,self.nDense,self.actFunction)
        return os.path.join(self.workPath,modelName)
    
    def _GetTempDataPath_(self):
        (filepath,tempfilename) = os.path.split(self.cfgFile)
        (filename,extension) = os.path.splitext(tempfilename)
        return os.path.join(self.workPath,'tempData',filename)
        
    def TrainModel(self):
        modelfile=self._GetModelFile_()
        tempDataPath=self._GetTempDataPath_()
        data_x=np.load(os.path.join(tempDataPath,'xTrain.npy'))
        data_y=np.load(os.path.join(tempDataPath,'yTrain.npy'))
        if os.path.exists(modelfile):
            model=models.load_model(modelfile)
        else:
            model=buildRNNModel(data_x[0].shape,self.nGRU,self.nDense,self.actFunction)
        trainRNNModel(model,data_x,data_y)
        model.save(modelfile)
        
    def TestModel(self):
        modelfile=self._GetModelFile_()
        model=models.load_model(modelfile)
        tempDataPath=self._GetTempDataPath_()
        xTest=np.load(os.path.join(tempDataPath,'xTest.npy'))
        yTest=np.load(os.path.join(tempDataPath,'yTest.npy'))
        testResult=RNNTest(model,xTest,yTest)
        fileName='testResult_'+datetime.datetime.now().strftime("%Y%m%d")+'.csv'
        np.savetxt(os.path.join(tempDataPath,fileName),testResult,fmt="%.4f",delimiter=',')
        
    def CompareModels(self,rRange=2,nNet=3,nRepeat=1):
        tempDataPath=self._GetTempDataPath_()
        xTest=np.load(os.path.join(tempDataPath,'xTest.npy'))
        yTest=np.load(os.path.join(tempDataPath,'yTest.npy'))
        xTrain=np.load(os.path.join(tempDataPath,'xTrain.npy'))
        yTrain=np.load(os.path.join(tempDataPath,'yTrain.npy'))
        xShape=xTrain[0].shape
        listPredict=[yTest]
        npGRU=np.array(self.nGRU)
        npDense=np.array(self.nDense)
        modelPath=os.path.join(self.workPath,'model')
        if not os.path.exists(modelPath):
            os.makedirs(modelPath)
        for nn in range(nNet):
            nowGRU=list((npGRU*(3*nn/2/nNet+0.5)).astype(int))
            nowDense=list((npDense*(3*nn/2/nNet+0.5)).astype(int))
            modelFile=os.path.join(modelPath,getModelName(nowGRU,nowDense,self.actFunction))
            if not os.path.exists(modelFile):
                RNNModel=buildRNNModel(xShape,nowGRU,nowDense,self.actFunction)
            else:
                RNNModel=models.load_model(modelFile)
            print(nowGRU,nowDense)
            for nr in range(nRepeat):
                print(nn,nr)
                trainRNNModel(RNNModel,xTrain,yTrain)
                listPredict.append(RNNModel.predict(xTest).reshape(-1))
                backend.clear_session()
        fileName='predicted_'+str(rRange)+'_'+str(nNet)+'_'+str(nRepeat)+datetime.datetime.now().strftime("%Y%m%d")+'.csv'
        np.savetxt(os.path.join(tempDataPath,fileName),np.array(listPredict).T,fmt="%.4f",delimiter=',')
        
        
    def collectAllData(self):
        self.updateStdData()
        #self.updateWindDailyAmntData()
        self.updateAmntByTick()
        self.calInduData()
        
#---------------Build HFIF Model End--------

if __name__=='__main__':
    gtime = time.time()
    np.seterr(divide='ignore',invalid='ignore')
    print('Start Running...')
    #build up
    workPath='F:\\草稿\\HFI_Model'
    cfgFile='F:\\草稿\\HFI_Model\\cfg\\cfg_sz50_v331atan.xlsx'
    if not os.path.exists(workPath):
        workPath='C:\\Users\\WAP\\Documents\\HFI_Model'
        cfgFile='C:\\Users\\WAP\\Documents\\HFI_Model\\cfg\\cfg_sz50_v331atan.xlsx'
    HFIF_Model=AIHFIF(workPath,cfgFile)
    dictCodeInfo=HFIF_Model.dictCodeInfo
    #collect data
    
    HFIF_Model.collectAllData()
    HFIF_Model.calTensorData(strEndDate='20190102')
    HFIF_Model.calTensorData(isTrain=False,strStartDate='20190103')
    HFIF_Model.TrainModel()
    HFIF_Model.TestModel()
    HFIF_Model.CompareModels()
    #testResult=RNNTest(RNNModel,xTest,yTest)
    
    print('\nRunning Ok. Duration in minute: %0.2f minutes'%((time.time() - gtime)/60))