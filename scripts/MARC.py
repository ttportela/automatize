# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
@author: Lucas May Petry
'''
import sys, os 
sys.path.insert(0, os.path.abspath('.'))
from automatize.methods.marc.marc_nn import marc

if len(sys.argv) < 8:
    print('Please run as:')
    print('\tpython MARC.py', 'TRAIN_FILE', 'TEST_FILE', 'RESULTS_FILE', 'DATASET_NAME', 'EMBEDDING_SIZE', 'MERGE_TYPE', 'RNN_CELL')
    exit()
    
METHOD = 'OURS'
TRAIN_FILE = sys.argv[1]
TEST_FILE = sys.argv[2]
METRICS_FILE = sys.argv[3]
DATASET = sys.argv[4]
EMBEDDER_SIZE = int(sys.argv[5])
MERGE_TYPE = sys.argv[6].lower()
RNN_CELL = sys.argv[7].lower()

marc(METHOD, TRAIN_FILE, TEST_FILE, METRICS_FILE, DATASET, EMBEDDER_SIZE, MERGE_TYPE, RNN_CELL)
