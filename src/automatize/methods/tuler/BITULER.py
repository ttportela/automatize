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
#from tqdm.notebook import tqdm
#from glob import glob
#from joblib import load, dump
#from pymove.models.classification import Tuler as tul
#from pymove.models import datautils
#from pymove.core import utils
#import json

import sys, os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from main import importer

def TrajectoryBITULER(dir_path, res_path, prefix='', save_results=True, n_jobs=-1, random_state=42, label_poi='poi', geohash=False, geo_precision=30):
    
    importer(['S', 'TCM', 'sys', 'json', 'tqdm', 'datetime'], globals())
    from methods._lib.pymove.core import utils
    from methods._lib.pymove.models.classification import Tuler as tul
    from methods._lib.datahandler import loadTrajectories
    from methods._lib.utils import update_report, print_params, concat_params
    
    dir_validation = os.path.join(res_path, 'BITULER-'+prefix, 'validation')
    dir_evaluation = os.path.join(res_path, 'BITULER-'+prefix)
    
    # Load Data - Tarlis:
    X, y, features, num_classes, space, dic_parameters = loadTrajectories(dir_path, prefix+'_', 
                                                                          split_test_validation=True,
                                                                          features_encoding=True, 
                                                                          y_one_hot_encodding=False,
                                                                          data_preparation=2,
                                                                          features=[label_poi],
                                                                          space_geohash=geohash,
                                                                          geo_precision=geo_precision)
    assert (len(X) > 2), "[BITULER:] ERR: data is not set or < 3"
    if len(X) > 2:
        X_train = X[0] 
        X_val = X[1]
        X_test = X[2]
        y_train = y[0] 
        y_val = y[1]
        y_test = y[2]

    #features = ['poi', 'label', 'tid']

    num_classes = dic_parameters['num_classes']
    max_lenght = dic_parameters['max_lenght']
    vocab_size = dic_parameters['vocab_size'][label_poi]
    rnn= ['bilstm']
    units = [100, 200, 250, 300]
    stack = [1]
    dropout =[0.5]
    embedding_size = [100, 200, 300, 400]
    batch_size = [64]
    epochs = [1000]
    patience = [20]
    monitor = ['val_acc']
    optimizer = ['ada']
    learning_rate = [0.001]
            
    print("\n[BITULER:] Building BITULER Model")
    start_time = datetime.now()

    total = len(rnn)*len(units)*len(stack)* len(dropout)* len(embedding_size)* len(batch_size)*len(epochs) * len(patience) *len(monitor) * len(learning_rate)
    print('[BITULER:] Starting model training, {} iterations'.format(total))
    
    if save_results and not os.path.exists(dir_validation):
        os.makedirs(dir_validation)
    
    # Hiper-param data:
    data = []
    def getParamData(f):
        marksplit = '-'
        df_ = pd.read_csv(f)
        f = f[f.find('bituler-'):]
        df_['nn']=   f.split(marksplit)[1]
        df_['un']=     f.split(marksplit)[2]
        df_['st']=     f.split(marksplit)[3]
        df_['dp'] = f.split(marksplit)[4]
        df_['es'] = f.split(marksplit)[5]
        df_['bs'] = f.split(marksplit)[6]
        df_['epoch'] = f.split(marksplit)[7]
        df_['pat'] = f.split(marksplit)[8]
        df_['mon'] = f.split(marksplit)[9]
        df_['lr'] = f.split(marksplit)[10]
        df_['fet'] = f.split(marksplit)[11].split('.csv')[0]
        data.append(df_)

    pbar = tqdm(itertools.product(rnn, units, stack, dropout, embedding_size, 
                                  batch_size, epochs, patience, monitor, learning_rate), 
                total=total, desc="[BITULER:] Model Training")
    for c in pbar:
        nn=c[0]
        un=c[1]
        st=c[2]
        dp=c[3]
        es=c[4]
        bs=c[5]
        epoch=c[6]
        pat=c[7]
        mon=c[8]
        lr=c[9]

        filename = os.path.join(dir_validation, 'bituler-'+
                                concat_params(nn, un, st, dp, es, bs, epoch, pat, mon, lr, features)+'.csv')

        if os.path.exists(filename):
            pbar.set_postfix_str('Skip ---> {}\n'.format(filename))
            getParamData(filename)
        else:

            pbar.set_postfix_str(print_params('nn, un, st, dp, es, bs, epoch, pat, mon, lr',
                                             nn, un, st, dp, es, bs, epoch, pat, mon, lr))

            bituler = tul.BiTulerLSTM(max_lenght=max_lenght,    
                        num_classes=num_classes,
                        vocab_size=vocab_size,
                        rnn_units=un,
                        dropout=dp,
                        embedding_size=es,
                        stack=st)

            bituler.fit(X_train, y_train,
                        X_val, y_val,
                        batch_size=bs,
                        epochs=epoch,
                        learning_rate=lr,
                        save_model=False,
                        save_best_only=False,
                        save_weights_only=False)

            validation_report, y_pred = bituler.predict(X_val, y_val)
                
            if save_results:
                validation_report.to_csv(filename, index=False)
                
            
            data.append( update_report(validation_report, 'nn, un, st, dp, es, bs, epoch, pat, mon, lr, features',
                                       nn, un, st, dp, es, bs, epoch, pat, mon, lr, features) )
            
            bituler.free()
    
    
    df_result = pd.concat(data)
    df_result.reset_index(drop=True, inplace=True)
    df_result.sort_values('acc', ascending=False, inplace=True)
    df_result.iloc[:10:]
    

    model = 0
    nn = df_result.iloc[model]['nn']
    un = int(df_result.iloc[model]['un'])
    st = int(df_result.iloc[model]['st'])
    dp = float(df_result.iloc[model]['dp'])
    es = int(df_result.iloc[model]['es'])
    bs = int(df_result.iloc[model]['bs'])
    epoch = int(df_result.iloc[model]['epoch'])
    pat = int(df_result.iloc[model]['pat'])
    mon = df_result.iloc[model]['mon']
    lr = float(df_result.iloc[model]['lr'])
    features

    filename = os.path.join(dir_evaluation, 'eval_bituler-'+concat_params(nn, un, st, dp, es, bs, epoch, pat, mon, lr, features)+'.csv')

    print("[BITULER:] Filename: {}.".format(filename))

    if not os.path.exists(filename):
        print('[BITULER:] Creating a model to test set')
        print("[BITULER:] Parameters: " + print_params('nn, un, st, dp, es, bs, epoch, pat, mon, lr, features',
                                              nn, un, st, dp, es, bs, epoch, pat, mon, lr, features) )

        evaluate_report = []
        rounds = 10

        pbar = tqdm(range(rounds), desc="Model Testing")
        for e in pbar:
            pbar.set_postfix_str('Rounds {} of {}'.format(e, rounds))
            
            bituler = tul.BiTulerLSTM(max_lenght=max_lenght,    
                num_classes=num_classes,
                vocab_size=vocab_size,
                rnn_units=un,
                dropout=dp,
                embedding_size=es,
                stack=st)

            bituler.fit(X_train, y_train,
                    X_val, y_val,
                    batch_size=bs,
                    epochs=epoch,
                    learning_rate=lr,
                    save_model=False,
                    save_best_only=False,
                    save_weights_only=False)

            #evaluate_report.append(bituler.predict(X_test, y_test))
            eval_report, y_pred = bituler.predict(X_test, y_test)
            evaluate_report.append(eval_report)
            bituler.free()

            
        if save_results:
            if not os.path.exists(dir_evaluation):
                os.makedirs(dir_evaluation)
                
            evaluate_report = pd.concat(evaluate_report)
            evaluate_report.to_csv(filename, index=False)
            
        end_time = (datetime.now()-start_time).total_seconds() * 1000
        print('[BITULER:] Processing time: {} milliseconds. Done.'.format(end_time))
    else:
        print('[BITULER:] Model previoulsy built.')
        
    print('\n--------------------------------------\n')


