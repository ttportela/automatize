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
from datetime import datetime

# from main import importer
# importer(['S', 'datetime', 'ClassifierEnsemble'], globals())
from automatize.methods.tec.tec import TEC

if len(sys.argv) < 4:
    print('Please run as:')
    print('\tMAT-TEC.py', 'PATH TO DATASET', 'PATH TO RESULTS_DIR', 'ENSEMBLES', 'DATASET')
    print('Example:')
    print('\tMAT-TEC.py', '"./data"', '"./results"', '"{\'movelets\': \'./movelets-res\', \'marc\': \'./data\', \'poifreq\': \'./poifreq-res\'}"', 'specific')
    exit()

data_path = sys.argv[1]
results_path = sys.argv[2]
ensembles = eval(sys.argv[3])
dataset = sys.argv[4]

modelfolder='model-ensemble'
if len(sys.argv) >= 5:
    modelfolder = sys.argv[5]
    
time = datetime.now()

TEC(data_path, results_path, ensembles, dataset, save_results=True, modelfolder=modelfolder)

time_ext = (datetime.now()-time).total_seconds() * 1000

print("Done. Processing time: " + str(time_ext) + " milliseconds")
print("# ---------------------------------------------------------------------------------")