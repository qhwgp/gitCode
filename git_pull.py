# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 08:43:52 2019

@author: WGP
"""
import git,os
gitPath='F:/草稿/gitCode/gitCode'
if not os.path.exists(gitPath):
    gitPath='C:/Users/WAP/gitCode'
repo = git.Repo(gitPath)
try:
    repo.remote().pull()
    print('pull from gitHub OK.')
except:
    if input('change on local file.Overwrite?([n]/y):')=='y':
        repo.git.stash()
        repo.remote().pull()
        print('save work and pull from gitHub OK.')
    else:
        print('Keep local file.')