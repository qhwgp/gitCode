# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 09:02:36 2019

@author: WGP
"""

import git



def pushPath(gitPath,strUpdateInfo):
    repo = git.Repo(gitPath)
    if repo.is_dirty():
        fileList=[]
        listStatus=repo.git.status().split('\n')
        for strInfo in listStatus:
            if '\t' in strInfo:
                fileList.append(strInfo.replace('\t','').replace('modified:   ',''))
    repo.index.add(fileList)
    repo.index.commit(strUpdateInfo)
    repo.remote().push()
    return fileList
    
if __name__=='__main__':
    gitPath='F:/草稿/gitCode/gitCode'
    fileList=pushPath(gitPath,'test for giPython')