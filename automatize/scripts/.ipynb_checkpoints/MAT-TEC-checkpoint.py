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
from datetime import datetime

# from main import importer
# importer(['S', 'datetime', 'ClassifierEnsemble'], globals())
from automatize.methods.tec.tec import TEC

import argparse

def parse_args():
    """[This is a function used to parse command line arguments]

    Returns:
        args ([object]): [Parse parameter object to get parse object]
    """
    parse = argparse.ArgumentParser(description='TEC Trajectory Ensemble Classification')
    parse.add_argument('data-path', type=str, help='path for the dataset files')
    parse.add_argument('result-path', type=str, help='path for the results files')
    parse.add_argument('ensembles', type=str, help='dictionary of the ensembles')
    parse.add_argument('-d', '--dataset', type=str, default='specific', help='dataset prefix name')
    parse.add_argument('-m', '--model-folder', type=str, default='model-ensemble', help='the folder where to put classification results')
    
    parse.add_argument('-r', '--seed', type=int, default=1, help='random seed')
    
    args = parse.parse_args()
    config = vars(args)
    return config
 
config = parse_args()
#print(config)

data_path    = config["data-path"]
results_path = config["result-path"]
ensembles    = eval(config["ensembles"])
dataset      = config["dataset"]

modelfolder  = config["model_folder"]
random_seed  = config["seed"]
    
time = datetime.now()

TEC(data_path, results_path, ensembles, dataset, save_results=True, modelfolder=modelfolder, random_seed=random_seed)

time_ext = (datetime.now()-time).total_seconds() * 1000

print("Done. Processing time: " + str(time_ext) + " milliseconds")
print("# ---------------------------------------------------------------------------------")