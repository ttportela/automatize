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

from automatise.main import display
# from main import importer, display
# importer(['S', 'results2df'], globals())
from automatise.results import results2df

# print(len(sys.argv))
if len(sys.argv) < 2:
    print('Please run as:')
    print('\tPrintResults.py', '"PATH TO FOLDER"', '"METHOD"', '"MODEL_FOLDER"')
    print('Example:')
    print('\tPrintResults.py', '"./results/method"', '"hiper*"', '"model"')
    exit()

results_path = sys.argv[1]
method = "*"

if len(sys.argv) > 2:
    method = sys.argv[2]
    
modelfolder = 'model'
if len(sys.argv) > 3:
    modelfolder = sys.argv[3]

dirr = os.path.join(results_path)
#coringa = ""

df = results2df(dirr, '', method, modelfolder=modelfolder)
display(df)

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(df)
