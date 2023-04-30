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

from automatize.main import display
from automatize.analysis import ACC4All
from automatize.results import results2df

if len(sys.argv) < 2:
    print('Please run as:')
    print('\tpython MAT-MC_RF.py', 'res_path', 'prefix', '[OPTIONAL:', 'save_results', 'modelfolder]')
    print('OR as:')
    print('\tpython MAT-MC_RF.py', 'res_path', 'prefix', 'True', 'modelfolder')
    exit()
    
    
res_path = sys.argv[1]
prefix   = sys.argv[2]

save_results = True
modelfolder='model'

if len(sys.argv) > 3:
    save_results = bool(sys.argv[3])
    modelfolder  = sys.argv[4]

print('Starting analysis in: ', res_path, prefix)
ACC4All(res_path, prefix, save_results, modelfolder, classifiers=['RF'])