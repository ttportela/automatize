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

# from main import importer
# importer(['S', 'glob', 'shutil'], globals())

if len(sys.argv) < 2:
    print('Please run as:')
    print('\tResultsTo.py', 'PATH TO RESULTS', 'DESTINATION')
    print('Example:')
    print('\tResultsTo.py', '"./results"', '"./simple_results"')
    exit()

results_path = sys.argv[1]
to_path    = sys.argv[2]

def getFiles(path):
    filesList = []
    print("Looking for result files in " + path)
    for files in glob.glob(path):
        fileName, fileExtension = os.path.splitext(files)
#         filelist.append(fileName) #filename without extension
        filesList.append(files) #filename with extension

    return filesList

filelist = ['*.txt', 'classification_times.csv', '*_history_Step5.csv', '*_history.csv', '*_results.csv']
filesList = []

for file in filelist:
    path = os.path.join(results_path, '**', file)
    filesList = filesList + getFiles(path)
    
# path = os.path.join(results_path, '**', '*.txt' )
# filesList.append( getFiles(path) )
# path = os.path.join(results_path, '**', 'model', '**')
# filesList.append( getFiles(path) )


# 1: Build up list of files:
# import glob2 as glob
# print("Looking for result files in " + path)
# for files in glob.glob(path):
#     fileName, fileExtension = os.path.splitext(files)
#     filelist.append(fileName) #filename without extension
#     filesList.append(files) #filename with extension


# path = os.path.join(results_path, '**', 'model', '**')
# print("Looking for result files in " + path)
# for files in glob.glob(path):
#     fileName, fileExtension = os.path.splitext(files)
#     filelist.append(fileName) #filename without extension
#     filesList.append(files) #filename with extension


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
