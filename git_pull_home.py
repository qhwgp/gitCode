# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 08:43:52 2019

@author: WGP
"""
import git
gitPath='C:/Users/WAP/gitCode'
repo = git.Repo(gitPath)
try:
    repo.remote().pull()
    print('pull from gitHub OK.')
except:
    repo.git.stash()
    repo.remote().pull()
    print('save work and pull from gitHub OK.')