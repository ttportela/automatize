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
import tarfile

# from main import importer
# importer(['S', 'glob', 'shutil'], globals())

if len(sys.argv) < 2:
    print('Please run as:')
    print('\tExportResults.py', 'PATH TO RESULTS')
    print('Example:')
    print('\tExportResults.py', '"./results"')
    exit()

results_path = sys.argv[1]
to_file    = os.path.join(results_path, os.path.basename(os.path.normpath(results_path))+'.tgz')

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

filesList = list(set(filesList))

with tarfile.open(to_file, "w:gz") as tar:
    for source in filesList:
        target = source.replace(results_path, '')
        print('Add:', target)
        tar.add(source, arcname=target)

print("Done.")
