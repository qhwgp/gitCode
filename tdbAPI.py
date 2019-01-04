# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 13:54:08 2019

@author: WGP
"""

from tdb import tdbapi,tdbapi_struct

api = tdbapi.tdbapi("10.200.50.15", "10010", "Test", "Test")

ret = api.start()

for market in ret.marketList:
    codeList = api.getCodeTable(market.replace('\x00',''))