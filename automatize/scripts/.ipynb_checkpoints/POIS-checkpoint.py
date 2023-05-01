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

from automatize.main import importer #, display
importer(['S', 'datetime','poifreq'], globals())

import argparse

def parse_args():
    """[This is a function used to parse command line arguments]

    Returns:
        args ([object]): [Parse parameter object to get parse object]
    """
    parse = argparse.ArgumentParser(description='POIS Trajectory Classification')
    parse.add_argument('path', type=str, help='path for saving the POIS result files')
    parse.add_argument('-m', '--method', type=str, default='npoi', help='POIF method name (poi, [npoi], wnpoi)')
    parse.add_argument('-s', '--sequences', type=str, default='1,2,3', help='sequences sizes to concatenate')
    parse.add_argument('-a', '--attributes', type=str, default='poi,hour', help='attribute names to concatenate')
    parse.add_argument('-d', '--dataset', type=str, default='specific', help='the dataset prefix name')
    parse.add_argument('-f', '--result-folder', type=str, default='NPOI_1_2_3_specific', help='folder where to find the POIS processed files')
    
    parse.add_argument('-r', '--seed', type=int, default=1, help='random seed')
    
    parse.add_argument('--geohash', action='store_true', default=False, 
                       help='use GeoHash encoding for spatial aspects (not implemented)')   
    parse.add_argument('-g', '--geo-precision', type=int, default=30, help='Space precision for GeoHash/GridIndex encoding') 
    
    parse.add_argument('--classify', action='store_true', default=False, help='Do also classification?')
    
    args = parse.parse_args()
    config = vars(args)
    return config
 
config = parse_args()
#print(config)

#if len(sys.argv) < 6:
#    print('Please run as:')
#    print('\tPOIS.py', 'METHOD', 'SEQUENCES', 'FEATURES', 'DATASET', 'PATH TO DATASET', 'PATH TO RESULTS_DIR')
#    print('Example:')
#    print('\tPOIS.py', 'npoi', '"1,2,3"', '"poi,hour"', 'specific', '"./data"', '"./results"')
#    exit()

path_name   = config["path"]

METHOD      = config["method"]
SEQUENCES   = [int(x) for x in config["sequences"].split(',')]
FEATURES    = config["attributes"].split(',')
DATASET     = config["dataset"]
RESULTS_DIR = config["result_folder"]

random_seed   = config["seed"]
classify      = config["classify"]

# TODO:
geohash       = config["geohash"]
geo_precision = config["geo_precision"]

time = datetime.now()
poifreq(SEQUENCES, DATASET, FEATURES, path_name, RESULTS_DIR, method=METHOD, save_all=True, doclass=classify)
time_ext = (datetime.now()-time).total_seconds() * 1000

print("Done. Processing time: " + str(time_ext) + " milliseconds")
print("# ---------------------------------------------------------------------------------")