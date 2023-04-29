# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Jun, 2022
Copyright (C) 2022, License GPL Version 3 or superior (this portion of code is subject to licensing from source project distribution)

@author: Tarlis Portela (adapted)

# Original source:
# Author: Nicksson C. A. de Freitas, 
          Ticiana L. Coelho da Silva, 
          Jose António Fernandes de Macêdo, 
          Leopoldo Melo Junior, 
          Matheus Gomes Cordeiro
# Adapted from: https://github.com/nickssonfreitas/ICAART2021
'''
# --------------------------------------------------------------------------------
import sys, os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#import pandas as pd
#import numpy as np
#import mplleaflet as mpl
#import traceback
#import time
#import gc
#import os
#import itertools
#import collections
#import itertools
#from os import path
#from tqdm.auto import tqdm
#from glob import glob
#from joblib import load, dump
#from methods._lib.pymove.models.classification import RandomForest as rf
#from methods._lib.pymove.models import datautils
#from methods._lib.pymove.core import utils
#import json

from main import importer
#importer(['S'], globals())

def TrajectoryRF(dir_path, res_path, prefix='', save_results=True, n_jobs=-1, random_state=42, geohash=False, geo_precision=30):
    
    #import time
    #import mplleaflet as mpl
    #import traceback
    #import gc
    #from joblib import load, dump
    
    # TODO - replace for pymove package version when implemented
    from methods._lib.pymove.models.classification import RandomForest as rf
##    from methods._lib.pymove.models import datautils
    from methods._lib.pymove.core import utils
    
    importer(['S', 'TCM', 'sys', 'json', 'tqdm', 'datetime'], globals())
    from methods._lib.datahandler import loadTrajectories
    from methods._lib.utils import update_report, print_params, concat_params

#    paper = 'SAC'
#    dataset = 'brightkite' #['fousquare_nyc', 'brightkite', 'foursquare_global', gowalla,'criminal_id', 'criminal_activity']
#    file_train = 'data/{}/train.csv.gz'.format(dataset)
#    file_val = 'data/{}/val.csv.gz'.format(dataset)
#    file_test = 'data/{}/test.csv.gz'.format(dataset)
#    dir_validation = '{}/{}/randomforest/validation/'.format(paper, dataset)
#    dir_evaluation = '{}/{}/randomforest/'.format(paper, dataset)

    dir_validation = os.path.join(res_path, 'TRF-'+prefix, 'validation')
    dir_evaluation = os.path.join(res_path, 'TRF-'+prefix)

    #space_geohash=False # True: Geohash, False: indexgrid
    #geo_precision=30 #meters
    
    # Load Data - Tarlis:
    X, y, features, num_classes, space, dic_parameters = loadTrajectories(dir_path, prefix+'_', 
                                                                          split_test_validation=True,
                                                                          features_encoding=True, 
                                                                          y_one_hot_encodding=False,
                                                                          space_geohash=geohash,
                                                                          geo_precision=geo_precision)
    
#    df_train = pd.read_csv(file_train)
#    df_val = pd.read_csv(file_val)
#    df_test = pd.read_csv(file_test)
#    df = pd.concat([df_train, df_val, df_test])


    # ## GET TRAJECTORIES
#    features = ['tid','label','hour','day','poi','indexgrid30']

#    data = [df_train[features], df_val[features], df_test[features]]
#    X, y, dic_parameters = datautils.generate_X_y_machine_learning(data= data,
#                                            features_encoding=True,       
#                                            y_one_hot_encodding=False)

    assert (len(X) > 2), "[TRF:] ERR: data is not set or < 3"
#   if len(X) == 2:
#       X_train = X[0] 
#       X_test = X[1]
#       y_train = y[0] 
#       y_test = y[1]
#       validate = False
    if len(X) > 2:
        X_train = X[0] 
        X_val = X[1]
        X_test = X[2]
        y_train = y[0] 
        y_val = y[1]
        y_test = y[2]
            
    print("\n[TRF:] Building Random Forrest Model")
    start_time = datetime.now()
    
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]# Number of trees in random forest
    max_depth = [int(x) for x in np.linspace(20, 40, num = 3)] # Maximum number of levels in tree
    min_samples_split =  [2, 5, 10] # Minimum number of samples required to split a node
    min_samples_leaf =  [1, 2, 4] # Minimum number of samples required at each leaf node
    max_features= ['sqrt', 'log2'] #['auto', 'sqrt'] # Number of features to consider at every split 
    # Tarlis: max_features 'auto' is deprecated, replaced ['auto', 'sqrt'] with: ['sqrt', 'log2'] ?
    bootstrap =  [True, False] # Method of selecting samples for training each tree
    
    #verbose = 1
    verbose = False
    
    total = len(n_estimators) * len(max_depth) * len(min_samples_split) * len(min_samples_leaf) * len(max_features) *        len(bootstrap)
    print('[TRF:] Starting model training, {} iterations'.format(total))

    if save_results and not os.path.exists(dir_validation):
        os.makedirs(dir_validation)
    
    # Hiper-param data:
    data = []
    def getParamData(f):
        marksplit = '-'
        df_ = pd.read_csv(f)
        f = f.split('randomforest')[-1]
        df_['ne']= f.split(marksplit)[1]
        df_['md']= f.split(marksplit)[2]
        df_['mss']= f.split(marksplit)[3]
        df_['msl']= f.split(marksplit)[4]
        df_['mf']= f.split(marksplit)[5]
        df_['bs']= f.split(marksplit)[6]
        df_['feature']= f.split(marksplit)[7].split('.csv')[0]
        data.append(df_)
    
    pbar = tqdm(itertools.product(n_estimators, max_depth, min_samples_split, 
                                  min_samples_leaf, max_features, bootstrap), 
                total=total, desc="[TRF:] Model Training")
    
    for c in pbar:
        ne=c[0]
        md=c[1]
        mss=c[2]
        msl=c[3]
        mf=c[4]
        bs=c[5]

        filename = os.path.join(dir_validation, 'randomforest-'+concat_params(ne, md, mss, msl, mf, bs, features)+'.csv')

#            print('this directory {} does not exist'.format(dir_validation))
#            break
        if os.path.exists(filename):
            pbar.set_postfix_str('Skip ---> {}\n'.format(filename))
            getParamData(filename)
        else:
#            print('Creating model...')
#            print(filename)
            pbar.set_postfix_str(print_params('ne, md, mss, msl, mf, bs', ne, md, mss, msl, mf, bs))

            RF = rf.RFClassifier(n_estimators=ne,
                                 max_depth=md,
                                 max_features=mf,
                                 min_samples_split=mss,
                                 min_samples_leaf=msl,
                                 bootstrap=bs,
                                 random_state=random_state,
                                 verbose=verbose,
                                 n_jobs=n_jobs)

            RF.fit(X_train, y_train)
            
            validation_report, y_pred = RF.predict(X_val, y_val)

            if save_results:
                validation_report.to_csv(filename, index=False)
            
            #validation_report['ne']= ne
            #validation_report['md']= md
            #validation_report['mss']= mss
            #validation_report['msl']= msl
            #validation_report['mf']= mf
            #validation_report['bs']= str(bs)
            #validation_report['feature']= str(features)
            #data.append(validation_report)
            data.append( update_report(validation_report, 'ne, md, mss, msl, mf, bs, features', 
                                       ne, md, mss, msl, mf, str(bs), str(features)) )

            RF.free()


#    files = utils.get_filenames_subdirectories(dir_validation)

#    marksplit = '-'
#    for f in files:
#        df_ = pd.read_csv(f)
#        f = f.split('randomforest')[-1]
#        df_['ne']= f.split(marksplit)[1]
#        df_['md']= f.split(marksplit)[2]
#        df_['mss']= f.split(marksplit)[3]
#        df_['msl']= f.split(marksplit)[4]
#        df_['mf']= f.split(marksplit)[5]
#        df_['bs']= f.split(marksplit)[6]
#        df_['feature']= f.split(marksplit)[7].split('.csv')[0]
#        data.append(df_)

    df_result = pd.concat(data)
    df_result.reset_index(drop=True, inplace=True)

    df_result.sort_values('acc', ascending=False, inplace=True)
    df_result.head(5)

    model = 0
    ne = int(df_result.iloc[model]['ne'])
    md = int(df_result.iloc[model]['md'])
    mss = int(df_result.iloc[model]['mss'])
    msl = int(df_result.iloc[model]['msl'])
    mf = df_result.iloc[model]['mf']
    bs = utils.str_to_bool(df_result.iloc[0]['bs'])
#   else:
#       model = 0
#       ne = n_estimators[0]
#       md = max_depth[0]
#       mss = min_samples_split[0]
#       msl = min_samples_leaf[0]
#       mf = max_features[0]
#       bs = bootstrap[0]
        

    filename = os.path.join(dir_evaluation, 'eval_randomforest-'+
                            concat_params(ne, md, mss, msl, mf, bs, features)+'.csv')

    print("[TRF:] Filename: {}.".format(filename))

    if not os.path.exists(filename):
        print('[TRF:] Creating a model to test set')
        evaluate_report = []
        rounds = 10
        
        print("[TRF:] Parameters: \n\tn_estimators: {}. \n\tmax_depth: {}. \n\tmin_samples_split: {}. \n\tmin_samples_leaf: {}. \n\tmax_features: {}. \n\tbootstrap: {}. \n\tfeatures: {}.".format(ne, md, mss, msl, mf, bs, features))

        pbar = tqdm(range(rounds), desc="Model Testing")

        for e in pbar:
            pbar.set_postfix_str('Round {} of {}'.format(e, rounds))
            RF = rf.RFClassifier(n_estimators=ne,
                            max_depth=md,
                            min_samples_split=mss,
                            min_samples_leaf=msl,
                            max_features=mf,
                            bootstrap=bs,
                            random_state=(random_state+e),
                            verbose=verbose,
                            n_jobs=n_jobs)


            #RF.fit(X_train, y_train)
            RF.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
            #RF.fit(X_val, y_val)
            eval_report, y_pred = RF.predict(X_test, y_test)
            #eval_report, y_pred = RF.predict(X_val, y_val)
            evaluate_report.append(eval_report)

            RF.free()

        if save_results:
            if not os.path.exists(dir_evaluation):
                os.makedirs(dir_evaluation)
                
            evaluate_report = pd.concat(evaluate_report)
            evaluate_report.to_csv(filename, index=False)
            
        end_time = (datetime.now()-start_time).total_seconds() * 1000
        print('[TRF:] Processing time: {} milliseconds. Done.'.format(end_time))
    else:
        print('[TRF:] Model previoulsy built.')
        
    print('\n--------------------------------------\n')
