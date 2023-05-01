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
from automatize.methods.pois.model_poifreq import model_poifreq

import argparse

def parse_args():
    """[This is a function used to parse command line arguments]

    Returns:
        args ([object]): [Parse parameter object to get parse object]
    """
    parse = argparse.ArgumentParser(description='POIS Trajectory Classification')
    parse.add_argument('path', type=str, help='path for the POIS result files')
    parse.add_argument('-m', '--method', type=str, default='npoi', help='POIF method name (poi, [npoi], wnpoi)')
    parse.add_argument('-p', '--prefix', type=str, default='npoi_1_2_3_specific', help='files prefix name')
    parse.add_argument('-d', '--dataset', type=str, default='specific', help='the dataset prefix name')
    parse.add_argument('-f', '--result-folder', type=str, default='NPOI_1_2_3_specific', help='folder where to find the POIS processed files')
    
    parse.add_argument('-r', '--seed', type=int, default=1, help='random seed')
    
    #parse.add_argument('--gpu', action=argparse.BooleanOptionalAction, default=True, help='Use GPU devices, or False for CPU')    
    #parse.add_argument('-M', '--ram', type=int, default=-1, help='Limit RAM memory GB (not implemented)')
    #parse.add_argument('-T', '--njobs', type=int, default=-1, help='Limit number of threads, and no GPU (not implemented)')

    args = parse.parse_args()
    config = vars(args)
    return config
 
config = parse_args()
#print(config)

path      = config["path"]
METHOD    = config["method"]
DATASET   = config["dataset"]
PREFIX    = config["prefix"]

folder    = config["result_folder"]

random_seed   = config["seed"]

path_name = os.path.join(path, METHOD+'_'+PREFIX) #METHOD+'_'+PREFIX)

RESULTS_FILE = os.path.join(path, folder, 'poifreq_results.txt')
METRICS_FILE = os.path.join(path, folder, METHOD+'_'+PREFIX+'.csv')

model_poifreq(path_name, METHOD=METHOD, METRICS_FILE=METRICS_FILE, RESULTS_FILE=RESULTS_FILE, random_seed=random_seed)

# -----------------------------------------------------------------------------------
from datetime import datetime
print(datetime.now().isoformat())