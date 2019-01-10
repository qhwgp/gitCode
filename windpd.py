# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
from WindPy import w
import numpy as np

def updateWindDailyAmntData(self,strSDate='19000101'):
        print('Start updateWindDailyAmntData...')
        dataFile=os.path.join(self.workPath,'DailyAmntData','DailyAmntData.csv')
        listRawDataDate=os.listdir(self.rawDataPath)
        startDate=listRawDataDate[0]
        if int(strSDate)>int(startDate):
            startDate=strSDate
        listCode=list(self.dictCodeInfo.keys())
        nday=self.nDayAverage
        pdData=getPdWIndAmnt(nday,listCode,startDate)
        pdData.to_csv(dataFile)
        #return pdData

"""
def getPdWIndAmnt(nday,listCode,strSDay,strEDay=''):
    
    if w.isconnected()==False:
        msg=w.start(waitTime=15)
        if msg.ErrorCode!=0:
            return False
    msg=w.tdaysoffset(-nday, strSDay, "")
    if msg.ErrorCode!=0:
        return False
    amntSDay=msg.Data[0][0].strftime('%Y%m%d')
    strCode=','.join(listCode)
    amntEDay=''
    if strEDay!='':
        amntEDay=strEDay
    msg=w.wsd(strCode, 'amt', amntSDay, amntEDay, "")
    if msg.ErrorCode!=0:
        return False
    return pd.DataFrame(np.array(msg.Data).T,index=msg.Times,columns=msg.Codes)
"""

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

