# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 09:02:36 2019

@author: WGP

push to git
"""

import git,datetime,os

strDT=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
strUpdateInfo='by office at '+strDT
gitPath='F:/草稿/gitCode/gitCode'
if not os.path.exists(gitPath):
    gitPath='C:/Users/WAP/gitCode'
    strUpdateInfo='by home at '+strDT
#-------------------------------------------
repo = git.Repo(gitPath)
fileList=[]
listStatus=repo.git.status().split('\n')
for strInfo in listStatus:
    if '\t' in strInfo:
        fileList.append(strInfo.replace('\t','').replace('modified:   ','').replace('deleted:   ',''))
if fileList:
    repo.index.add(fileList)
    repo.index.commit(strUpdateInfo)
    repo.remote().push()
    print('push to gitHub OK.')
else:
    print('Nothing happened.')