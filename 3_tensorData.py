#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 15:07:39 2018

@author: apple
"""

import os
import csv
import math
import numpy as np

def tensor_x(data_path, output_path, num_x, num_y, f):
    filelist = os.listdir(data_path)
    filelist.sort()
    data = []
    for i in range(0, len(filelist)):
        if filelist[i][-3:] != 'csv':
            continue
        path = os.path.join(data_path, filelist[i])
        csv_file = csv.reader(open(path))
        rows = []
        for row in csv_file:
            rows.append(list(map(float, row)))
        for j in range(0, len(rows)):
            if j + num_x + num_y > len(rows):
                break
            data1time = []
            for k in range(j, j + num_x):
                data1time.append(list(map(f, rows[k]))[2:])
            data.append(data1time)
            j += 1
    data = np.array(data)
    np.save(output_path, data)
    return(data)
    
def tensor_y(data_path, output_path, num_x, num_y, f):
    filelist = os.listdir(data_path)
    filelist.sort()
    data = []
    for i in range(0, len(filelist)):
        if filelist[i] == 'result_2' or filelist[i] == 'weight_rate':
            continue
        if filelist[i][-3:] == 'lsx' or filelist[i][-3:] == 'ore':
            continue
        path = os.path.join(data_path, filelist[i])
        file = os.listdir(path)
        file.sort()
        for j in range(0, len(file)):
            if file[j][0] == '9':
                path_csv = os.path.join(path, file[j])
                csv_file = csv.reader(open(path_csv))
                rows = [row for row in csv_file]
                for k in range(0, len(rows)):
                    if k + num_x + num_y >= len(rows):
                        break
                    data1time = f(float(rows[k + num_x + num_y][0]) / float(rows[k + num_x][0]))
                    data.append(data1time)
                    k += 1
    data = np.array(data)
    np.save(output_path, data)
    return(data)

def f(x):
    if x == 0:
        y = math.pi / 2
    else:
        y = math.atan(1 / x)
    return(y)

if __name__ == '__main__':    
    data_path_x= '/Users/apple/Downloads/ListData/weight_rate'
    #权重文件夹
    data_path_y = '/Users/apple/Downloads/ListData/standardData'
    #指数标准数据文件夹
    output_x_file = '/Users/apple/Downloads/ListData/tensor_x'
    output_y_file = '/Users/apple/Downloads/ListData/\tensor_y'
    num_x = 60
    num_y = 20
    data_x = tensor_x(data_path_x, output_x_file, num_x, num_y, f)
    data_y = tensor_y(data_path_y, output_y_file, num_x, num_y, f)