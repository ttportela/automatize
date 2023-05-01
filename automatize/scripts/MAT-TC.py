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
from automatize.analysis import ClassifyByTrajectory

import argparse

def parse_args():
    """[This is a function used to parse command line arguments]

    Returns:
        args ([object]): [Parse parameter object to get parse object]
    """
    parse = argparse.ArgumentParser(description='MAT Trajectory Classification')
    parse.add_argument('data-path', type=str, help='path for the dataset folder')
    parse.add_argument('results-path', type=str, help='path for the results folder')
    parse.add_argument('-ds', '--dataset', type=str, default='specific', help='dataset name')
    parse.add_argument('-c', '--classifiers', type=str, default='MARC,TRF,TXGB,DEEPEST', help='classifiers methods')
    
    parse.add_argument('-r', '--random', type=int, default=1, help='random seed')
    parse.add_argument('--save', action='store_true', default=True, help='save results') 
    
    parse.add_argument('--geohash', action='store_true', default=False, 
                       help='use GeoHash encoding for spatial aspects (not implemented)')   
    parse.add_argument('-g', '--geo-precision', type=int, default=30, help='Space precision for GeoHash/GridIndex encoding') 
    
    parse.add_argument('-of', '--one-feature', type=str, default='poi', help='[BITULER,TULVAE] Single feature classification (sets attribute name)')

    args = parse.parse_args()
    config = vars(args)
    return config
 
config = parse_args()
#print(config)
    
data_path = config["data-path"]
res_path  = config["results-path"]
prefix    = config["dataset"]

save_results  = config["save"]
random        = config["random"]
geohash       = config["geohash"]
geo_precision = config["geo_precision"]

one_feature = config["one_feature"]

classifiers  = config["classifiers"].split(',')

print('Starting analysis in: ', res_path, prefix)
ClassifyByTrajectory(res_path, data_path, prefix, save_results, classifiers=classifiers, 
                     random_seed=random, geohash=geohash, geo_precision=geo_precision, one_feature=one_feature)
