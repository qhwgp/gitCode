# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 13:48:44 2019

@author: WAP
"""

import myModel,myFunc
import datetime,xlrd,os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import numpy as np

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
            dictStdData[tFlag][code]=myFunc.csvToList(csvFile)
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
        
        while myFunc.intToTime(rawTickData[nRawData][0])<STime:
            lprice=rawTickData[nRawData][1]
            nRawData+=1
        for i in range(len_series+1):
            #lastTime=STime+(i-1)*timeDiff
            nowTime=STime+i*timeDiff
            amnt=0
            while myFunc.intToTime(rawTickData[nRawData][0])<=nowTime:
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
        if aveAmnt<0.01:
            continue
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
def getTensorData(xNormData,nDailyData,nx,ny,nIndu):
    lenDData=nDailyData-ny[-1]-1
    ndday=int(xNormData.shape[0]/lenDData)
    xData=[]
    yData=[]
    for idday in range(ndday):
        for i in range(nx,lenDData):
            n=idday*lenDData+i
            xData.append(myFunc.toModelInput(xNormData[(n-nx):n,:-len(ny)-1],(2,nx,nIndu)))
            yData.append(xNormData[n,-len(ny)-1:])
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
            normInduData[i,j]=myFunc.calPercentile(xData[i,j],arrPercentile)
            #normInduData[i,j]=calPcl(xData[i,j],arrPercentile)
    return normInduData

#----------Model step function end------------
        
#-----------Build HFIF Model Class------------

class AIHFIF:
    def __init__(self,workPath,cfgFile,nMinuteX=0):
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
        self.minuteXData=nMinuteX
        self.minuteYData=0
        self.nGRU=0
        self.nDense=0
        self.actFunction=''
        self.log=''
        self._getCfg()
        
    def _getCfg(self):
        cfgFilePath=os.path.join(self.workPath,'cfg',self.cfgFile)
        data = xlrd.open_workbook(cfgFilePath)
        #page1
        sheetCodeInfo = data.sheets()[0]
        arrShares = sheetCodeInfo.col_values(1)[1:]
        arrCode = sheetCodeInfo.col_values(0)[1:]
        arrIndustry = sheetCodeInfo.col_values(2)[1:]
        dictIndu={}
        for i in range(len(arrCode)):
            self.dictCodeInfo[arrCode[i]]=[arrShares[i],arrIndustry[i]]
            dictIndu[arrIndustry[i]]=1
        #page2
        sheetCfg=data.sheets()[1]
        arrCfg=sheetCfg.col_values(1)
        self.nIndu=len(dictIndu)
        self.timeSpan=arrCfg[0]
        self.indexCode=arrCfg[1]
        self.windIndexCode=arrCfg[2]
        self.rawDataPath=arrCfg[3]
        self.nDayAverage=int(arrCfg[4]+0.01)
        self.minuteXData=list(map(int,arrCfg[5].split(',')))[self.minuteXData]
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
                rawData=myFunc.csvToList(rawDataPath)
                if CheckRawData(rawData)!=0:
                    continue
                listStdData=tickToStdData(rawData,self.timeSE,self.timeSpan)
                for i in range(len(listStdData)):
                    OutFile=os.path.join(outPath,code+'_'+str(i)+'.csv')
                    if not os.path.exists(OutFile):
                        myFunc.listToCsv(listStdData[i],OutFile)

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
            intDate=myFunc.strToInt(aDay)
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
                avgAmntFile=os.path.join(self.workPath,'cfg',filename)
                if not os.path.exists(avgAmntFile):
                    os.makedirs(avgAmntFile)
                avgAmntFile=os.path.join(avgAmntFile,'avgAmnt_'+filename+'.csv')
                pd.DataFrame.from_dict(dictPastAveAmnt[intLED], orient='index').to_csv(avgAmntFile)
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
        rawYData=np.array([])
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
            npyData=npyData[:lenDData]
            if xData.size==0:
                xData=npxData
                rawYData=npyData
            else:
                try:
                    xData=np.vstack((xData,npxData))
                    rawYData=np.hstack((rawYData,npyData))
                except:
                    print('induDataFile error: '+induDataFile)
        ttFlag='Test'
        if isTrain:
            pclMatrix=np.percentile(np.abs(xData),range(100),axis=0)
            np.savetxt(os.path.join(modelPath,'pclMatrix_'+filename+'.csv'),pclMatrix,fmt="%.4f",delimiter=',')
            ttFlag='Train'
        else:
            pclMatrix=np.loadtxt(os.path.join(modelPath,'pclMatrix_'+filename+'.csv'),delimiter=',')
        xNormData=getNormInduData(xData,pclMatrix)
        if not isTrain:
            xNormData=np.hstack((xNormData,rawYData.reshape((-1,1))))
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
        nDailyData=int((ts[1]-ts[0]).seconds/self.timeSpan)#2000
        nx=myFunc.toInt(self.minuteXData*60/self.timeSpan)
        ny=np.array(self.minuteYData)*myFunc.toInt(60/self.timeSpan)
        return (nDailyData,nx,ny)
    
    def TrainModel(self,config):#nRepeat=1,isNewTrain=False,batchSize=1024):
        listModelFile=self._GetModelFile_()
        normDataPath=self._GetNormDataPath_()
        xNormData=np.load(os.path.join(normDataPath,'normTrainData.npy'))
        normTestData=np.load(os.path.join(normDataPath,'normTestData.npy'))
        nDailyData,nx,ny=self._getModelParam_()
        xTest,yTest=getTensorData(normTestData,nDailyData,nx,ny,self.nIndu)
        listPScore=[]
        listModel=[]
        for iy in range(len(ny)):
            modelfile=listModelFile[iy]
            mf=modelfile.split('\\')[-1].replace('model_cfg_','').replace('.h5','')
            if os.path.exists(modelfile) and not config.isNewTrain:
                print('Load TrainModel...'+mf)
                model=myModel.loadModel(modelfile)
            else:
                print('Create TrainModel...'+mf)
                #model=buildRNNModel((nx,xNormData.shape[1]-len(ny)),self.nGRU,self.actFunction)
                model=myModel.buildCNNModel((2,nx,self.nIndu),nHiddenLayers=config.nHiddenLayers,
                                        filters=config.filters,kernel_size=config.kernel_size,opt=config.opt)
            npScore=np.zeros((config.nRepeat,2))
            ev=myModel.trainRNNModel(model,xNormData,nDailyData,nx,ny,iy,
                        xTest,yTest[:,iy],self.nIndu,config.batchSize,config.nRepeat)
            listModel.append(model)
            npScore[:,0]=ev['val_loss'][-config.nRepeat:]
            npScore[:,1]=ev['val_myMetric'][-config.nRepeat:]
            npScore=np.round(np.mean(npScore,axis=0),4)
            listPScore.append(npScore)
            self.saveLogFile(iy,npScore)
            model.save(modelfile)
            
        predict,pScore=myFunc.testPredict(listModel,normTestData,nDailyData,nx,ny)
        self.saveTempFile(pScore,predict)
        myModel.clearBackEnd()
        return np.array(listPScore)
    
    def saveLogFile(self,iy,npScore):
        tempDataPath=self._GetTempDataPath_()
        msg=[datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")]
        msg.append(str(self.nGRU))
        msg.append(self.minuteXData)
        msg.append(self.minuteYData[iy])
        msg+=list(npScore)
        myFunc.writeLog(tempDataPath,msg)
      
    def saveTempFile(self,pScore,npPredict):
        mx=self.minuteXData
        my=self.minuteYData
        actFunct=self.actFunction
        predictName=myFunc.getSaveName(pScore,mx,my,actFunct)
        tempDataPath=self._GetTempDataPath_()
        mfileName=os.path.join(tempDataPath,predictName)
        i=0
        fileName=mfileName+'.csv'
        while os.path.exists(fileName):
            i+=1
            fileName=mfileName+'_'+str(i)+'.csv'
        np.savetxt(os.path.join(tempDataPath,fileName),npPredict,fmt="%.4f",delimiter=',')

    def collectAllData(self,rCfg):
        if rCfg.isCollectAllData:
            self.updateStdData(rCfg.strSDate,rCfg.strEDate)
            self.updateAmntByRaw(rCfg.strSDate,rCfg.strEDate)
            self.calInduData(rCfg.strSDate,rCfg.strEDate)#minus one row
        if rCfg.isCalTensorData:
            self.calTensorData(isTrain=True,strEDate=rCfg.splitDay)#Train Data,minus len(yTimes) rows
            self.calTensorData(isTrain=False,strSDate=rCfg.splitDay)#Test Data
        return self.TrainModel(rCfg)#nRepeat=rCfg.nRepeat,isNewTrain=rCfg.isNewTrain,
           #        batchSize=rCfg.batchSize)
        
#---------------Build HFIF Model End--------