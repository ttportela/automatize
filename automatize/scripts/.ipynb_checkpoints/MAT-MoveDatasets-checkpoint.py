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
import glob2 as glob

# from main import importer
# importer(['S', 'mergeDatasets'], globals())
from automatize.run import moveResults

import argparse

def parse_args():
    """[This is a function used to parse command line arguments]

    Returns:
        args ([object]): [Parse parameter object to get parse object]
    """
    parse = argparse.ArgumentParser(description='Move train.csv/test.csv from subdirecoty to the path')
    parse.add_argument('path', type=str, help='path for the results folder')

    args = parse.parse_args()
    config = vars(args)
    return config
 
config = parse_args()
#print(config)

results_path    = config["path"]

dir_from = os.path.dirname(glob.glob(os.path.join(results_path, '**', 'train.csv'))[0])
# print(os.path.dirname(glob.glob(os.path.join(results_path, '**', 'train.csv'))[0]))

moveResults(dir_from, results_path)
