# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 08:43:52 2019

@author: WGP
"""
import git
gitPath='F:/草稿/gitCode/gitCode'
repo = git.Repo(gitPath)
repo.remote().pull()
print('pull from gitHub OK.')