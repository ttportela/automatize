#!python
# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
@author: Lucas May Petry
'''
import sys, os 
sys.path.insert(0, os.path.abspath('.'))

import argparse

def parse_args():
    """[This is a function used to parse command line arguments]

    Returns:
        args ([object]): [Parse parameter object to get parse object]
    """
    parse = argparse.ArgumentParser(
        description='MARC Trajectory Classification',
        epilog="Created on Dec, 2021. Copyright (C) 2022, GPLv3. Author: Tarlis Tortelli Portela"
    )
    parse.add_argument('train-file', type=str, help='path for the train dataset file')
    parse.add_argument('test-file', type=str, help='path for the train dataset file')
    parse.add_argument('metrics-file', type=str, help='path for the resulting metrics file')
    parse.add_argument('-d', '--dataset', type=str, default='specific', help='dataset name (for output purposes)')
    parse.add_argument('-e', '--embedding-size', type=int, default=100, help='the embedding size (default 100)')
    parse.add_argument('-m', '--merge-tipe', type=str, default='concatenate', help='the merge type (add, average, [concatenate])')
    parse.add_argument('-c', '--rnn-cell', type=str, default='lstm', help='the RNN cell type ([lstm], gru)')
    
    parse.add_argument('-r', '--seed', type=int, default=1, help='random seed')
    parse.add_argument('-g', '--geo-precision', type=int, default=8, help='Space precision for GeoHash encoding')
    
    parse.add_argument('--no-gpu', action='store_false', help='Don´t use GPU devices.')    
    #parse.add_argument('-M', '--ram', type=int, default=-1, help='Limit RAM memory GB (not implemented)')
    #parse.add_argument('-T', '--njobs', type=int, default=-1, help='Limit number of threads, and no GPU (not implemented)')

    args = parse.parse_args()
    config = vars(args)
    return config
 
config = parse_args()
#print(config)

METHOD        = 'OURS'
TRAIN_FILE    = config["train-file"]
TEST_FILE     = config["test-file"]
METRICS_FILE  = config["metrics-file"]
DATASET       = config["dataset"]
EMBEDDER_SIZE = config["embedding_size"]
MERGE_TYPE    = config["merge_tipe"].lower()
RNN_CELL      = config["rnn_cell"].lower()

random_seed   = config["seed"]
geo_precision = config["geo_precision"]

GPU           = config["no_gpu"]
#GIG           = config["ram"]
#THR           = config["njobs"]

# ---
from automatize.methods.marc.marc_nn import marc
# ---

if not GPU:
    from tensorflow.python.client import device_lib
    gp = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
    cp = [x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU']

    print('Use GPU set as: ' + str(GPU), '- Device list:')
    print(gp)
    print(cp)

#if THR > 0:
#    tf.config.threading.set_intra_op_parallelism_threads(THR)
#    tf.config.threading.set_inter_op_parallelism_threads(THR)
#
    import tensorflow as tf
    with tf.device(gp[0] if GPU else cp[0]):
        marc(METHOD, TRAIN_FILE, TEST_FILE, METRICS_FILE, 
             DATASET, EMBEDDER_SIZE, MERGE_TYPE, RNN_CELL, 
             random_seed=random_seed, geo_precision=geo_precision)
else:
    marc(METHOD, TRAIN_FILE, TEST_FILE, METRICS_FILE, 
         DATASET, EMBEDDER_SIZE, MERGE_TYPE, RNN_CELL, 
         random_seed=random_seed, geo_precision=geo_precision)

from datetime import datetime
print(datetime.now().isoformat())


    
# TODO: Configuring environment
# -----------------------------------------------------------------------------------
# # Limit Threads?:
# if THR > 0:
#     print('Limiting MARC to', str(THR), ' Threads') #  and no GPU

#     import tensorflow
#     os.environ['CUDA_VISIBLE_DEVICES']='-1'
#     os.environ["OMP_NUM_THREADS"] = str(THR)
#     tensorflow.config.threading.set_inter_op_parallelism_threads(THR)

#     GPU = False

# # Limit MEMORY?:
# if GIG > 0:
#     print('Limiting MARC to', str(GIG), 'G')

#     GIG = GIG*1073741824
#     import resource
#     soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
#     resource.setrlimit(resource.RLIMIT_DATA, (GIG, hard))


# NO GPU, run on CPU:
#if gpu == False:
#    print('> GPU set to disable')
#    os.environ['CUDA_VISIBLE_DEVICES']='-1'
#print(os.environ['CUDA_VISIBLE_DEVICES'], tf.test.is_gpu_available())
#import tensorflow as tf
#if tf.test.gpu_device_name():
#    print('> GPU found!')
#    if gpu == False:
#        exit()
#else:
#    print("> GPU not found.")
# ----------------------------------------------------------------------------------- 