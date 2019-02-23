# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 13:56:26 2019

@author: WAP
"""

#workPath='C:\\Users\\WAP\\Documents\\HFI_Model'
strSDate='19000101'
strEDate='99990101'
splitDay='20190112'
listCfgFile=['cfg_sz50_v331atan.xlsx',
             'cfg_hs300_v22tan.xlsx',
             'cfg_zz500_v11tan.xlsx']
calFile=[0]
isCollectAllData=False
isCalTensorData=False
nRepeat=3
isNewTrain=True
batchSize=1024

useGPU=False

#model
nHiddenLayers=3#5
filters=75
kernel_size=(4,4)
opt='nadam'