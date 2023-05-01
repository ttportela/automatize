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

from automatize.main import display
from automatize.analysis import ACC4All
#from automatize.results import results2df

# def_random_seed(random_num=1, seed_num=1)

import argparse

def parse_args():
    """[This is a function used to parse command line arguments]

    Returns:
        args ([object]): [Parse parameter object to get parse object]
    """
    parse = argparse.ArgumentParser(description='MAT Movelets Classification')
    parse.add_argument('results-path', type=str, help='path for the results folder')
    parse.add_argument('folder', type=str, help='dataset name')
    parse.add_argument('-c', '--classifiers', type=str, default='MLP,RF,SVM', help='classifiers methods')
    parse.add_argument('-m', '--modelfolder', type=str, default='model', help='model folder')
    
    parse.add_argument('-r', '--random', type=int, default=1, help='random seed')
    parse.add_argument('--save', action='store_true', default=True, help='save results')

    args = parse.parse_args()
    config = vars(args)
    return config
 
config = parse_args()
#print(config)

res_path  = config["results-path"]
folder    = config["folder"]

save_results = config["save"]
modelfolder  = config["modelfolder"]
random       = config["random"] # TODO

classifiers  = config["classifiers"].split(',')

ACC4All(res_path, folder, save_results, modelfolder, classifiers=classifiers, random_seed=random)