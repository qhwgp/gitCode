# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 09:02:36 2019

@author: WGP

push to git in office
"""

import git

strUpdateInfo='add home'

gitPath='C:/Users/WAP/gitCode'
#-------------------------------------------
repo = git.Repo(gitPath)
fileList=[]
listStatus=repo.git.status().split('\n')
for strInfo in listStatus:
    if '\t' in strInfo:
        fileList.append(strInfo.replace('\t','').replace('modified:   ',''))
if fileList:
    repo.index.add(fileList)
    repo.index.commit(strUpdateInfo)
    repo.remote().push()
    print('push to gitHub OK.')
else:
    print('Nothing happened.')