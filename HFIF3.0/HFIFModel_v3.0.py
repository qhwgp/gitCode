# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 14:15:04 2018
version HFIF_v3.0
@author: wap
"""
import os,myConfig
if not myConfig.useGPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from clsHFIF import AIHFIF
import time,sys


def createWorkPath():
    listWorkPath=[sys.path[0]]
    listWorkPath.append('F:\\草稿\\HFI_Model')
    listWorkPath.append('C:\\Users\\WAP\\Documents\\HFI_Model')
    listWorkPath.append('D:\\ForeFactor\\HFI_Model')
    for wPath in listWorkPath:
        if os.path.exists(os.path.join(wPath,'cfg')):
            return wPath
    return None

#-------------------Run Process----------------
    
if __name__=='__main__':
    gtime = time.time()
    listCfgFile=myConfig.listCfgFile
    workPath=createWorkPath()
    dictPScore={}
    for ic in myConfig.calFile:
        cfgFile=listCfgFile[ic]
        for ix in range(myConfig.nXData):
            print('programming: '+cfgFile)
            HFIF_Model=AIHFIF(workPath,cfgFile,ix)
            dictPScore[cfgFile.replace('.xlsx','_mx'+str(ix))]=HFIF_Model.collectAllData(myConfig)
    print('\nOK.Duration in minute: %0.2f minutes'%((time.time() - gtime)/60))