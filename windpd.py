# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
from WindPy import w
import numpy as np

w.isconnected()

w.start(waitTime=15)

strCode=['600000.SH','600999.SH']

amntEDay='20181220'

amntSDay=amntEDay

amntEDay=''

msg=w.wsd(strCode, 'amt', amntSDay, amntEDay, "")

pdd=pd.DataFrame(np.array(msg.Data).T,index=msg.Times,columns=msg.Codes)

pdd.to_csv('pdData.csv')

pdData=pd.read_csv('pdData.csv',header=0,index_col=0)

