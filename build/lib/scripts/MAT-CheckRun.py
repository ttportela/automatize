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
from automatise.results import check_run

if len(sys.argv) < 1:
    print('Please run as:')
    print('\tResultsTo.py', 'PATH TO RESULTS')
    print('Example:')
    print('\tResultsTo.py', '"./results"')
    exit()

res_path = sys.argv[1]

check_run(res_path)
print("Done.")
