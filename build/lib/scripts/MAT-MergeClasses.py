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

# from main import importer
# importer(['S'], globals())

if len(sys.argv) < 1:
    print('Please run as:')
    print('\tMergeClasses.py', 'PATH TO FOLDER')
    print('Example:')
    print('\tMergeClasses.py', '"./results/HiPerMovelets"')
    exit()

results_path = sys.argv[1]

# --------------------------------------------------------------------------------------
def mergeDatasets(dir_path, file='train.csv'):
    files = [i for i in glob.glob(os.path.join(dir_path, '*', '**', file))]

    print("Loading files - " + file)
    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f).drop('class', axis=1) for f in files[:len(files)-1]], axis=1)
    combined_csv = pd.concat([combined_csv, pd.read_csv(files[len(files)-1])], axis=1)
    #export to csv
    print("Writing "+file+" file")
    combined_csv.to_csv(os.path.join(dir_path, file), index=False)
    
    print("Done.")

mergeDatasets(results_path, 'train.csv')
mergeDatasets(results_path, 'test.csv')