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

from automatize.results import containErrors, containWarnings

import argparse

def parse_args():
    """[This is a function used to parse command line arguments]

    Returns:
        args ([object]): [Parse parameter object to get parse object]
    """
    parse = argparse.ArgumentParser(description='Chek result files for erros, warnings, and count classes.')
    parse.add_argument('results-path', type=str, help='path for the results folder')

    args = parse.parse_args()
    config = vars(args)
    return config
 
config = parse_args()
#print(config)

results_path    = config["results-path"]

path = os.path.join(results_path, '**', '*.txt' )

# filelist = []
filesList = {}

# 1: Build up list of files:
print("Looking for result files in " + path)
for files in glob.glob(path):
    fileName, fileExtension = os.path.splitext(files)
#     filelist.append(os.path.dirname(fileName)) # parent
    filesList[os.path.dirname(fileName)] = files # filename with extension

for dirName, fileName in filesList.items(): 
    path = os.path.join(dirName, '*', '**', 'train.csv')
    ctc = 0
    for files in glob.glob(path):
        ctc += 1
    print('Classes: ', ctc, ' - Errors: ', containErrors(fileName), ' - Warnings: ', containWarnings(fileName), '-', dirName.replace(results_path, '...'))

print("Done.")
