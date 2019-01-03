#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 15:33:18 2018

@author: apple
"""

import os
import csv
import datetime
import numpy as np
import pandas as pd

def csvread(rootdir, file_new):
    
    morningStart = datetime.datetime(1900, 1, 1, 9, 40, 0)
    morningEnd = datetime.datetime(1900, 1, 1, 11, 20, 0)
    afternoonStart = datetime.datetime(1900, 1, 1, 13, 10, 0)
    afternoonEnd = datetime.datetime(1900, 1, 1, 14, 50, 0)
    delta_time = datetime.timedelta(0, 3, 0)
    #时间参数
    delta_time_max = datetime.timedelta(minutes = 5)
    len_series = int((morningEnd - morningStart) / delta_time)
    time_series_am = []
    time_series_pm = []
    file_wrong = []
    for i in range(0, len_series):
        time_series_am.append([morningStart + i * delta_time])
        time_series_pm.append([afternoonStart + i * delta_time])
    #生成时间序列
    
    filelist = os.listdir(rootdir)
    for k in range(0, len(filelist)):
        data_am = []
        data_pm = []
        flag = 0
        if filelist[k][-3:] != 'csv':
            continue
        path = os.path.join(rootdir, filelist[k])
        csv_file = csv.reader(open(path))
        flag_am = 0
        flag_pm = 0
        counter_am = 0
        counter_pm = 0
        for row in csv_file:
            dt = datetime.datetime.strptime(row[0], '%H%M%S%f')
            if dt < datetime.datetime(1900,1,1,9,36,0):
                continue
            if dt < datetime.datetime(1900,1,1,11,24,0):
                if flag_am == 0:
                    row[1] = int(row[1]) / 10000
                    row_old = row
                    dt_old = dt
                    flag_am += 1
                    continue
                else:
                    row[1] = int(row[1]) / 10000
                    if dt < dt_old:
                        flag = 1
                        break
                    if dt - dt_old > delta_time_max:
                        flag = 2
                        break
                    if float(row[1]) - float(row_old[1]) > 0.10 * float(row_old[1]) and dt > morningStart:
                        flag = 3
                        break
                    j_am = counter_am
                    for i in range(j_am, len_series):
                        if dt <= time_series_am[i][0] and dt >= morningStart:
                            if i > 0 and dt <= time_series_am[i - 1][0]:
                                data_am[i - 1][0] = row[1]
                                data_am[i - 1][1] = float(data_am[i - 1][1]) + float(row[2])
                            else:
                                data_am.append(row[1:])
                                counter_am += 1
                            row_old = row
                            dt_old = dt
                            break
                        elif dt > time_series_am[i][0] and dt_old <= morningEnd:
                            row_old[2] = 0
                            data_am.append(row_old[1:])
                            counter_am += 1
                            continue
            if dt > datetime.datetime(1900,1,1,13,6,0) and dt < datetime.datetime(1900,1,1,14,54,0):
                if flag_pm == 0:
                    row[1] = int(row[1]) / 10000
                    row_old = row
                    dt_old = dt
                    flag_pm += 1
                    continue
                else:
                    row[1] = int(row[1]) / 10000
                    if dt < dt_old:
                        flag = 4
                        break
                    if dt - dt_old > delta_time_max:
                        flag = 5
                        break
                    if float(row[1]) - float(row_old[1]) > 0.10 * float(row_old[1]):
                        flag = 6
                        break
                    j_pm = counter_pm
                    for i in range(j_pm, len_series):
                        if dt <= time_series_pm[i][0] and dt >= afternoonStart:
                            if i > 0 and dt <= time_series_pm[i - 1][0]:
                                data_pm[i - 1][0] = row[1]
                                data_pm[i - 1][1] = float(data_pm[i - 1][1]) + float(row[2])
                            else:
                                data_pm.append(row[1:])
                                counter_pm += 1
                            row_old = row
                            dt_old = dt
                            break
                        elif dt > time_series_pm[i][0] and dt_old <= afternoonEnd:
                            row_old[2] = 0
                            data_pm.append(row_old[1:])
                            counter_pm += 1
                            continue
        if data_am == [] or data_pm == []:
            file_wrong.append(rootdir[-8:] + filelist[k] + 'n' + str(flag))
            continue
        if len(data_am) != len_series or len(data_pm) != len_series:
            file_wrong.append(rootdir[-8:] + filelist[k] + 'l' + str(flag))
            continue
        if flag != 0:
            file_wrong.append(rootdir[-8:] + filelist[k] + str(flag))
            continue
        data_am = np.delete(data_am, 0, axis = 0)
        data_pm = np.delete(data_pm, 0, axis = 0)
        new_path_am = os.path.join(file_new, filelist[k][:-4]+'.am.csv')
        new_path_pm = os.path.join(file_new, filelist[k][:-4]+'.pm.csv')
        data_am = pd.DataFrame(data_am)
        data_pm = pd.DataFrame(data_pm)
        data_am.to_csv(new_path_am,header = 0, index = 0)
        data_pm.to_csv(new_path_pm,header = 0, index = 0)
    return file_wrong
        
        
rawData_path = '/Users/apple/Downloads/ListData'
standardData_path = '/Users/apple/Downloads/ListData'
"""

input_area

"""
file = os.listdir(rawData_path)
path_result = os.path.join(standardData_path, 'standardData')
folder = os.path.exists(path_result)
file_wrong = []
if not folder:
    os.makedirs(path_result)
for i in range(0, len(file)):
    if file[i][-3:] == 'ore':
        continue
    if file[i] == 'standardData':
        continue
    if file[i][-4:] == 'xlsx':
        continue
    if file[i] == 'weight_rate':
        continue
    path_data = os.path.join(rawData_path, file[i])
    path_final = os.path.join(path_result, file[i])
    final = os.path.exists(path_final)
    if not final:
        os.makedirs(path_final)
    file_wrong_day = csvread(path_data,path_final)
    file_wrong.append(file_wrong_day)