# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 09:02:36 2019

@author: WGP
"""

import git

def pullPath(gitPath):
    repo = git.Repo(gitPath)
    repo.remote().pull()

def pushPath(gitPath,strUpdateInfo):
    repo = git.Repo(gitPath)
    fileList=[]
    if repo.is_dirty():
        listStatus=repo.git.status().split('\n')
        for strInfo in listStatus:
            if '\t' in strInfo:
                fileList.append(strInfo.replace('\t','').replace('modified:   ',''))
        if fileList:
            repo.index.add(fileList)
            repo.index.commit(strUpdateInfo)
            repo.remote().push()
    return fileList
    
if __name__=='__main__':
    #Work Computer Path
    
    #gitPath='F:/草稿/gitCode/gitCode'
    gitPath='C:/Users/WAP/gitCode'
    
    strUpdateInfo='add model test'
    
    """
    fileList=pushPath(gitPath,strUpdateInfo)
    print('push to gitHub OK.')
    """
    pullPath(gitPath)
    print('pull from gitHub OK.')
    