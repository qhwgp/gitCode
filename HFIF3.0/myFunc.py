# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 13:41:32 2019

@author: WAP
"""
import csv,datetime,xlrd,os
import numpy as np
#--------------Basic function start-----------

#Basic 1:
def toInt(fv):
    return int(fv+0.5)

def toIntStr(fv):
    return str(int(fv+0.5))

def btstr(btpara):
    return str(btpara,encoding='utf-8')

def getListCfgFile(ffPath):
    cfgFile=os.path.join(ffPath,'cfgForeFactor.csv')
    cfgData=list(map(btstr,np.loadtxt(cfgFile,dtype=bytes)))
    return cfgData[4:]

def getRunCfg(wPath):
    dictCfg={}
    cfgFilePath=os.path.join(wPath,'cfg','runCfg.xls')
    data = xlrd.open_workbook(cfgFilePath).sheets()[0]
    arrName=data.col_values(0)
    arrCfg=data.col_values(1)
    for i in range(len(arrName)):
        dictCfg[arrName[i]]=arrCfg[i]
    return dictCfg

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

def writeLog(logPath,msg):
    logFile=os.path.join(logPath,'log.csv')
    with open(logFile,'a',newline='') as f:
        wrt=csv.writer(f)
        wrt.writerow(msg)
        f.close()

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
def getSaveName(pScore,nGRU,nDense,actFlag):
    try:
        strpScore='_'.join(list(map(toIntStr,pScore)))
    except:
        strpScore=toIntStr(pScore)
    strpScore='ps.'+strpScore
    try:
        strnGRU='_'.join(list(map(str,nGRU)))
    except:
        strnGRU=str(nGRU)
    try:
        strnDense='_'.join(list(map(str,nDense)))
    except:
        strnDense=str(nDense)
    return strpScore+'.'+strnGRU+'.'+strnDense+'.'+\
            datetime.datetime.now().strftime("%m%d")
    
def getTestResultName(nGRU,nDense,actFlag):
    return 'testResult.'+'_'.join(list(map(str,nGRU)))+'.'+\
    '_'.join(list(map(str,nDense)))+'.'+actFlag+'.'+datetime.datetime.now().strftime("%Y%m%d")

#Basic 6:

def testPredict(listModel,normTestData,nDailyData,nx,ny):
    lenDData=nDailyData-ny[-1]-1
    ndday=int(normTestData.shape[0]/lenDData)
    lenyday=lenDData-nx-ny[-1]
    xData=[]
    yData=[]
    pScore=[]
    yRawData=np.array([])
    for idday in range(ndday):
        nStart=idday*lenDData
        yIndex=normTestData[(nStart+nx):(nStart+lenDData),-1]
        yday=np.zeros((len(ny),lenyday))
        for iy in range(len(ny)):
            yday[iy,:]=(yIndex[ny[iy]:]/yIndex[:-ny[iy]]-1)[:lenyday]*10000
        if yRawData.size==0:
            yRawData=yday.T
        else:
            yRawData=np.vstack((yRawData,yday.T))
        for i in range(nx,lenyday+nx):
            n=nStart+i
            xData.append(normTestData[(n-nx):n,:-len(ny)-1])
            yData.append(normTestData[n,-len(ny)-1:])
    xData=np.array(xData)
    yData=np.array(yData)
    ly=len(xData)
    npPredict=np.hstack((yData,yRawData))
    for i in range(len(listModel)):
        model=listModel[i]
        predicted = model.predict(xData).reshape(-1)
        pScore.append(np.dot(predicted,yRawData[:,i])/ly*240)
        npPredict=np.hstack((npPredict,predicted.reshape(-1,1)))
    #yData=np.array(yData)
    return (npPredict,pScore)

def toModelInput(xNormData,shape):#(2,20,16)
    dt=np.zeros(shape)
    for j in range(shape[2]):
        dt[0,:,j]=xNormData[:,2*j]
        dt[1,:,j]=xNormData[:,2*j+1]
    return dt

#--------------Basic function end-------------