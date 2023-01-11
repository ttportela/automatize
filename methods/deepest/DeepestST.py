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
#from tqdm.auto import tqdm
#from glob import glob
#from joblib import load, dump
#import json
#
## From commom
#from _lib.pymove.models.classification import DeepestST as DST
#from _lib.pymove.models import datautils
#from _lib.pymove.core import utils

import sys, os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from main import importer

def TrajectoryDeepestST(dir_path, res_path, prefix='', save_results=True, n_jobs=-1, random_state=42, 
                        label_poi = 'poi', y_one_hot_encodding=True, geohash=False, geo_precision=30):
    
    importer(['S', 'TCM', 'sys', 'json', 'tqdm', 'datetime'], globals())
    from methods._lib.pymove.core import utils
    from methods._lib.pymove.models.classification import DeepestST as DST
    from methods._lib.datahandler import loadTrajectories
    from methods._lib.utils import update_report, print_params, concat_params
    
    dir_validation = os.path.join(res_path, 'DEEPEST-'+prefix, 'validation')
    dir_evaluation = os.path.join(res_path, 'DEEPEST-'+prefix)
    
    # Load Data - Tarlis:
    X, y, features, num_classes, space, dic_parameters = loadTrajectories(dir_path, prefix+'_', 
                                                                          split_test_validation=True,
                                                                          features_encoding=True, 
                                                                          y_one_hot_encodding=y_one_hot_encodding,
                                                                          data_preparation=2,
                                                                          space_geohash=geohash,
                                                                          geo_precision=geo_precision)
    assert (len(X) > 2), "[DEEPEST:] ERR: data is not set or < 3"
    if len(X) > 2:
        X_train = X[0] 
        X_val = X[1]
        X_test = X[2]
        y_train = y[0] 
        y_val = y[1]
        y_test = y[2]


    max_lenght = dic_parameters['max_lenght']
    num_classes = dic_parameters['num_classes']
    vocab_size = dic_parameters['vocab_size']
    features = dic_parameters['features']
    encode_features = dic_parameters['encode_features']
    encode_y = dic_parameters['encode_y']

    ## GRID SEARCH PARAMETERS
    rnn = ['bilstm', 'lstm']
    units = [100, 200, 300, 400, 500]
    merge_type = ['concat']
    dropout_before_rnn=[0, 0.5]
    dropout_after_rnn=[0.5]

    embedding_size = [50, 100, 200, 300, 400]
    batch_size = [64]
    epochs = [1000]
    patience = [20]
    monitor = ['val_acc']

    optimizer = ['ada']
    learning_rate = [0.001]
    loss = ['CCE']
    loss_parameters = [{}] # TODO unfix, it´s fixed for now, but if you add parameters, change all configs.

    y_ohe = y_one_hot_encodding
            
    print("\n[DEEPEST:] Building DeepestST Model")
    start_time = datetime.now()

    total = len(rnn)*len(units)*len(merge_type)*len(dropout_before_rnn)* len(dropout_after_rnn)*        len(embedding_size)* len(batch_size) * len(epochs) * len(patience) * len(monitor) *        len(optimizer) * len(learning_rate) * len(loss) #* len(loss_parameters) ## By Tarlis

    print('[DEEPEST:] Starting model training, {} iterations'.format(total))

    count = 0

    if save_results and not os.path.exists(dir_validation):
        os.makedirs(dir_validation)
    
    # Hiper-param data:
    data = []
    def getParamData(f):
        marksplit = '-'
        df_ = pd.read_csv(f)
        f = f[f.find('deepest-'):]
        df_['nn']= f.split(marksplit)[1]
        df_['un']= f.split(marksplit)[2]
        df_['mt']= f.split(marksplit)[3]
        df_['dp_bf']= f.split(marksplit)[4]
        df_['dp_af']= f.split(marksplit)[5]
        df_['em_s']= f.split(marksplit)[6]
        df_['bs']= f.split(marksplit)[7]
        df_['epoch']= f.split(marksplit)[8]
        df_['pat']= f.split(marksplit)[9]
        df_['mon']= f.split(marksplit)[10]
        df_['opt']= f.split(marksplit)[11]
        df_['lr']= f.split(marksplit)[12]
        df_['ls']= f.split(marksplit)[13]
#        df_['ls_p']= f.split(marksplit)[14]
        df_['ohe'] = y_one_hot_encodding
#        df_['feature']= f.split(marksplit)[16].split('.csv')[0]
        df_['feature']= f.split(marksplit)[15].split('.csv')[0]

        data.append(df_)

    pbar = tqdm(itertools.product(rnn, units, merge_type, dropout_before_rnn, dropout_after_rnn, 
                                  embedding_size, batch_size, epochs, patience, monitor, optimizer, 
                                  learning_rate, loss, loss_parameters), 
                total=total, desc="[DEEPEST:] Model Training")
    for c in pbar:
        nn=c[0]
        un=c[1]
        mt=c[2]
        dp_bf=c[3]
        dp_af=c[4]
        em_s=c[5]
        bs=c[6]
        epoch=c[7] 
        pat=c[8] 
        mon=c[9] 
        opt=c[10] 
        lr=c[11]
        ls=c[12]
        ls_p=loss_parameters[0] # TODO unfix # c[13]

        filename = os.path.join(dir_validation, 'deepest-'+
                                concat_params(nn, un, mt, dp_bf, dp_af, em_s, bs, epoch, 
                                              pat, mon, opt, lr, ls, y_ohe, features)+'.csv')
        count += 1

        if os.path.exists(filename):
            pbar.set_postfix_str('Skip ---> {}\n'.format(filename))
            getParamData(filename)
        else:

            pbar.set_postfix_str(print_params('nn, un, mt, dp_bf, dp_af, em_s, bs, epoch, pat, mon, opt, lr, ls, y_ohe, features',
                                             nn, un, mt, dp_bf, dp_af, em_s, bs, epoch, pat, mon, opt, lr, ls, y_ohe, features))



            deepest = DST.DeepeST(max_lenght=max_lenght,
                        num_classes=num_classes,
                        vocab_size = vocab_size,
                        rnn=nn,
                        rnn_units=un,
                        merge_type = mt,
                        dropout_before_rnn=dp_bf,
                        dropout_after_rnn=dp_af,
                        embedding_size = em_s)

            deepest.fit(X_train,
                        y_train,
                        X_val,
                        y_val,
                        batch_size=bs,
                        epochs=epoch,
                        monitor=mon,
                        min_delta=0,
                        patience=pat,
                        verbose=0,
#                        baseline=0.5,
                        baseline=None, # By Tarlis
                        optimizer=opt,
                        learning_rate=lr,
                        mode='auto',
                        new_metrics=None,
                        save_model=False,
                        modelname='',
                        save_best_only=True,
                        save_weights_only=False,
                        log_dir=None,
                        loss=ls,
                        loss_parameters=ls_p)

            validation_report, y_pred = deepest.predict(X_val, y_val)
                
            if save_results:
                validation_report.to_csv(filename, index=False)

            data.append( update_report(validation_report, 
                                       'nn, un, mt, dp_bf, dp_af, em_s, bs, epoch, pat, mon, opt, lr, ls, y_ohe, features',
                                       nn, un, mt, dp_bf, dp_af, em_s, bs, epoch, pat, mon, opt, lr, ls, y_ohe, features) )
            
            deepest.free()

    
    df_result = pd.concat(data)
    df_result = df_result[df_result['nn'] == 'lstm']
    df_result.reset_index(drop=True, inplace=True)
    df_result.sort_values('acc', ascending=False, inplace=True)
    
    model = 0
    nn =  df_result.iloc[model]['nn']
    un =  int(df_result.iloc[model]['un'])
    mt =  df_result.iloc[model]['mt']
    dp_bf = float(df_result.iloc[model]['dp_bf'])
    dp_af = float(df_result.iloc[model]['dp_af'])

    em_s = int(df_result.iloc[model]['em_s'])

    bs = int(df_result.iloc[0]['bs'])
    epoch = int(df_result.iloc[model]['epoch'])
    pat = float(df_result.iloc[model]['pat'])
    mon = df_result.iloc[model]['mon']

    opt = df_result.iloc[model]['opt']
    lr = float(df_result.iloc[0]['lr'])
    ls = df_result.iloc[model]['ls']
#    ls_p = json.loads(df_result.iloc[model]['ls_p'].replace("'", "\""))
    ls_p = loss_parameters[0] # TODO unfix

    y_ohe = y_one_hot_encodding

    filename = os.path.join(dir_evaluation, 'eval_deepest-'+ \
        concat_params(nn, un, mt, dp_bf, dp_af, em_s, bs, epoch, pat, mon, opt, lr, ls, y_ohe, features)+'.csv')
    
    print("[DEEPEST:] Filename: {}.".format(filename))

    if not os.path.exists(filename):
        print('[DEEPEST:] Creating a model to test set')
        print("[DEEPEST:] Parameters: " + print_params(
            'nn, un, mt, dp_bf, dp_af, em_s, bs, epoch, pat, mon, opt, lr, ls, y_ohe, features',
            nn, un, mt, dp_bf, dp_af, em_s, bs, epoch, pat, mon, opt, lr, ls, y_ohe, features) )
        
        evaluate_report = []
        rounds = 10

        pbar = tqdm(range(rounds), desc="Model Testing")
        for e in pbar:
            pbar.set_postfix_str('Rounds {} of {}'.format(e, rounds))

            deepest = DST.DeepeST(max_lenght=max_lenght,
                    num_classes=num_classes,
                    vocab_size = vocab_size,
                    rnn=nn,
                    rnn_units=un,
                    merge_type = mt,
                    dropout_before_rnn=dp_bf,
                    dropout_after_rnn=dp_af,
                    embedding_size = em_s)

            deepest.fit(X_train,
                    y_train,
                    X_val,
                    y_val,
                    batch_size=bs,
                    epochs=epoch,
                    monitor=mon,
                    min_delta=0,
                    patience=pat,
                    verbose=0,
                    baseline=None,
                    optimizer=opt,
                    learning_rate=lr,
                    mode='auto',
                    new_metrics=None,
                    save_model=False,
                    modelname='',
                    save_best_only=True,
                    save_weights_only=False,
                    log_dir=None,
                    loss=ls,
                    loss_parameters=ls_p)

            #evaluate_report.append(deepest.predict(X_test, y_test))
            eval_report, y_pred = deepest.predict(X_test, y_test)
            evaluate_report.append(eval_report)

            deepest.free()

        if save_results:
            if not os.path.exists(dir_evaluation):
                os.makedirs(dir_evaluation)
                
            evaluate_report = pd.concat(evaluate_report)
            evaluate_report.to_csv(filename, index=False)
            
        end_time = (datetime.now()-start_time).total_seconds() * 1000
        print('[DEEPEST:] Processing time: {} milliseconds. Done.'.format(end_time))
    else:
        print('[DEEPEST:] Model previoulsy built.')
        
    print('\n--------------------------------------\n')

