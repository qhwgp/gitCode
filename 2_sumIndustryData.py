#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 15:46:57 2018

@author: apple
"""

import os
import csv
import xlrd
import datetime
import numpy as np
import pandas as pd

def weight_rate(workfile_path, data_path, weight_path, data_amount, day, period):
    Nshare=10#股数
    Ncode = 4#代码
    Nindustry = 17#行业
    """
    input area
    
    """
    for j in range(0, len(data_amount)):
        if data_amount[j, 0] == day:
            amount_list = data_amount[j - period: j, :]
    code_list = data_amount[0, :]
    
    data = xlrd.open_workbook(workfile_path)
    table = data.sheets()[0]

    shares = table.col_values(Nshare)[1:]
    code = table.col_values(Ncode)[1:]
    industry = table.col_values(Nindustry)[1:]
    while '' in shares:
        shares.remove('')
    while '' in code:
        code.remove('')
    while '' in industry:
        industry.remove('')

    for i in range(0, len(code)):
        code[i] = str(code[i])
        if code[i][0] == '6' or code[i][0] == '9':
            code[i] = code[i] + '.SH.csv'
        else:
            code[i] = code[i] + '.SZ.csv'
    filelist = os.listdir(data_path)
    miss_file = []
    
    data = pd.DataFrame([code, shares, industry]).T
    data.columns = ['code', 'shares', 'industry']
    data = data.groupby(data['industry'])
    counter = 0
    for name,group in data:
        day_a = []
        group = group.values.tolist()
        price_rate_am = np.zeros(shape = (1999,1))
        amount_am = np.zeros(shape = (1999,1))
        price_rate_pm = np.zeros(shape = (1999,1))
        amount_pm = np.zeros(shape = (1999,1))
        for i in range(0, len(group)):
            if not (group[i][0][:-4] + '.am.csv') in filelist:
                miss_file.append(group[i][0])
                continue
            for k in range(0, len(amount_list[0])):
                if code_list[k] == group[i][0][:-4]:
                    day_a.append(amount_list[:,k].sum(axis = 0))
            path_am = os.path.join(data_path, group[i][0][:-4] + '.am.csv')
            csv_file = csv.reader(open(path_am))
            rows = []
            j = []
            for row in csv_file:
                weight = float(row[0]) * float(group[i][1])
                j = [weight, float(row[1])]
                rows.append(j)
            rows = np.array(rows)
            price_rate_am = price_rate_am + rows[:,0].reshape(len(rows),1)
            amount_am = amount_am + rows[:,1].reshape(len(rows),1)
            
            path_pm = os.path.join(data_path, group[i][0][:-4] + '.pm.csv')
            csv_file = csv.reader(open(path_pm))
            rows = []
            j = []
            for row in csv_file:
                weight = float(row[0]) * float(group[i][1])
                j = [weight, float(row[1])]
                rows.append(j)
            rows = np.array(rows)
            price_rate_pm = price_rate_pm + rows[:,0].reshape(len(rows),1)
            amount_pm = amount_pm + rows[:,1].reshape(len(rows),1)
        day_amount = (np.array(day_a)).sum() / (period * 4800)
        price_rate_am = ((price_rate_am[1:1999,:] / price_rate_am[0:1998,:]) - 1) * 10000
        price_rate_pm = ((price_rate_pm[1:1999,:] / price_rate_pm[0:1998,:]) - 1) * 10000
        amount_am = (amount_am / day_amount)[1:1999,:]
        amount_pm = (amount_pm / day_amount)[1:1999,:]
        data_1_am = np.append(price_rate_am, amount_am, axis = 1)
        data_1_pm = np.append(price_rate_pm, amount_pm, axis = 1)
        if counter == 0:
            data_final_am = data_1_am
            data_final_pm = data_1_pm
            counter += 1
        else:
            data_final_am = np.append(data_final_am, data_1_am, axis = 1)
            data_final_pm = np.append(data_final_pm, data_1_pm, axis = 1)

    new_path_am = os.path.join(weight_path, data_path[-8:] + '.am.csv')
    data_final_am = pd.DataFrame(data_final_am)
    data_final_am.to_csv(new_path_am, header = 0, index = 0)
    new_path_pm = os.path.join(weight_path, data_path[-8:] + '.pm.csv')
    data_final_pm = pd.DataFrame(data_final_pm)
    data_final_pm.to_csv(new_path_pm, header = 0, index = 0)
    return(miss_file)
  
workfile_path = '/Users/apple/Downloads/000016closeweight20181018.xls'
#权重路径
data_path = '/Users/apple/Downloads/ListData/standardData'
#标准数据路径
path = '/Users/apple/Downloads/ListData'
#输出路径
amount_path = '/Users/apple/Downloads/ListData/DailyAmountData.xlsx'
#日数据
period = 20
#平均交易量时间
"""

input area


"""

file = os.listdir(data_path)
weight_path = os.path.join(path, 'weight_rate')
folder = os.path.exists(weight_path)
miss_file = []

data_amount = pd.read_excel(amount_path, sheet_name='Sheet1', header = None)
data_amount = np.array(data_amount)


if not folder:
    os.makedirs(weight_path)
for i in range(0, len(file)):
    if file[i][-3:] == 'ore':
        continue
    day = datetime.datetime.strptime(file[i], '%Y%m%d')
    day_path = os.path.join(data_path, file[i])
    miss_file_day = weight_rate(workfile_path, day_path, weight_path, data_amount, day, period)
    miss_file.append(miss_file_day)