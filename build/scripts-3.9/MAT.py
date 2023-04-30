#!python
# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2023
Copyright (C) 2023, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
import sys, os 
sys.path.insert(0, os.path.abspath('.'))

import argparse

def parse_args():
    """[This is a function used to parse command line arguments]

    Returns:
        args ([object]): [Parse parameter object to get parse object]
    """
    parse = argparse.ArgumentParser(description='AutoMATize Web (MAT)')
    parse.add_argument('-d', '--data', '--data-path', type=str, default='../datasets', help='path for the datasets folder (for listing)')
#    parse.add_argument('experiment-path', type=str, , default='../results' help='path for the results folder (for scripting)')

    args = parse.parse_args()
    config = vars(args)
    return config
 
config = parse_args()
#print(config)

data_path  = config["data"]
#experiment_path  = config["experiment-path"]

# UNDER CONSTRUCTION
from automatize.app import app
from automatize.assets.app_base import HOST, PORT, DEBUG

os.environ['AUTOMATIZE_DATA_PATH'] = data_path

app.run_server(host=HOST, port=PORT, debug=DEBUG)

