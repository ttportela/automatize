# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
import pandas as pd
import sys, os 
sys.path.insert(0, os.path.abspath('.'))
# from main import importer, display
# importer(['S', 'results2df'], globals())

from automatize.main import display
from automatize.results import results2df

import argparse

def parse_args():
    """[This is a function used to parse command line arguments]

    Returns:
        args ([object]): [Parse parameter object to get parse object]
    """
    parse = argparse.ArgumentParser(description='Check and print results as human readable')
    parse.add_argument('results-path', type=str, help='path for the results folder')
    
    parse.add_argument('-f', '--method', type=str, default='*', help='method folder')
    parse.add_argument('-m', '--modelfolder', type=str, default='model', help='model folder')

    args = parse.parse_args()
    config = vars(args)
    return config
 
config = parse_args()
#print(config)

results_path    = config["results-path"]
method          = config["method"]
modelfolder     = config["modelfolder"]

dirr = os.path.join(results_path)
#coringa = ""

df = results2df(dirr, '', method, modelfolder=modelfolder)
display(df)

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(df)