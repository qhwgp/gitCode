# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 09:02:36 2019

@author: WGP
"""

import git

gitPath='F:/草稿/gitCode/gitCode'

repo = git.Repo(gitPath)

if repo.is_dirty():
    fileList=[]
    listStatus=repo.git.status().split('\n')
    for strInfo in listStatus:
        if '\tmodified:' in strInfo:
            fileList.append()
            