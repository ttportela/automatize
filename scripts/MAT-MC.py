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

# def_random_seed(random_num=1, seed_num=1)

if len(sys.argv) < 2:
    print('Please run as:')
    print('\tpython MAT-MC.py', 'res_path', 'prefix', '[OPTIONAL:', 'classifiers_list', 'modelfolder]')
    print('OR as:')
    print('\tpython MAT-MC.py', 'res_path', 'prefix', '"MLP,RF,SVM"', '/modelfolder')
    exit()
    
    
res_path = sys.argv[1]
prefix   = sys.argv[2]
classifiers=['MLP', 'RF', 'SVM']

save_results = True
modelfolder='model'

if len(sys.argv) > 3:
    classifiers = sys.argv[3].split(',')
    
if len(sys.argv) > 4:
    modelfolder  = sys.argv[4]

print('Starting analysis in: ', res_path, prefix)
ACC4All(res_path, prefix, save_results, modelfolder, classifiers=classifiers)

# print('Results for: ', res_path, prefix)

# df = results2df(res_path, '**', prefix)
# display(df)