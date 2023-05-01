#!python
# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
import sys, os 
sys.path.insert(0, os.path.abspath('.'))
import pandas as pd
import glob2 as glob
import shutil

import argparse

def parse_args():
    """[This is a function used to parse command line arguments]

    Returns:
        args ([object]): [Parse parameter object to get parse object]
    """
    parse = argparse.ArgumentParser(description='Move the metric result files (do not include data files)')
    parse.add_argument('from-path', type=str, help='path for the results folder')
    parse.add_argument('to-path', type=str, help='path for the results folder')

    args = parse.parse_args()
    config = vars(args)
    return config
 
config = parse_args()
#print(config)

results_path    = config["from-path"]
to_path         = config["to-path"]

def getFiles(path):
    filesList = []
    print("Looking for result files in " + path)
    for files in glob.glob(path):
        fileName, fileExtension = os.path.splitext(files)
#         filelist.append(fileName) #filename without extension
        filesList.append(files) #filename with extension

    return filesList

filelist = ['*.txt', 'classification_times.csv', '*_report_Step5.csv', '*_report.csv', '*_history_Step5.csv', '*_history.csv', '*_results.csv', 'eval_*.csv']
filesList = []

for file in filelist:
    path = os.path.join(results_path, '**', file)
    filesList = filesList + getFiles(path)


results_path = os.path.abspath(results_path)

if not os.path.exists(to_path):
    print('Creating: ', to_path)
    os.makedirs(to_path)
to_path      = os.path.abspath(to_path)

filesList = list(set(filesList))

for original in filesList:
    target = original.replace(results_path, '')
    target = to_path + target #os.path.join(to_path, target)
    if not os.path.exists(os.path.dirname(target)):
        os.makedirs(os.path.dirname(target))
    print('CP:', original, '=>', target)
    shutil.copyfile(original, target)

print("Done.")
