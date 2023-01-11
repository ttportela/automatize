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
from main import importer

def TrajectoryXGBoost(dir_path, res_path, prefix='', save_results=True, n_jobs=-1, random_state=42, geohash=False, geo_precision=30):
    
    importer(['S', 'TCM', 'sys', 'json', 'tqdm', 'datetime'], globals())
    from methods._lib.pymove.core import utils
    from methods._lib.pymove.models.classification import XGBoost as xg
    from methods._lib.datahandler import loadTrajectories
    from methods._lib.utils import update_report, print_params, concat_params
    
    dir_validation = os.path.join(res_path, 'TXGB-'+prefix, 'validation')
    dir_evaluation = os.path.join(res_path, 'TXGB-'+prefix)
    
    # Load Data - Tarlis:
    X, y, features, num_classes, space, dic_parameters = loadTrajectories(dir_path, prefix+'_', 
                                                                          split_test_validation=True,
                                                                          features_encoding=True, 
                                                                          y_one_hot_encodding=False,
                                                                          space_geohash=geohash,
                                                                          geo_precision=geo_precision)
    assert (len(X) > 2), "[TXGB:] ERR: data is not set or < 3"
    if len(X) > 2:
        X_train = X[0] 
        X_val = X[1]
        X_test = X[2]
        y_train = y[0] 
        y_val = y[1]
        y_test = y[2]

    n_estimators = [2000]
    max_depth = [3, 5]
    learning_rate = [0.01]
    gamma = [0.0, 1, 5]
    subsample = [0.1, 0.2, 0.5, 0.8]
    colsample_bytree = [0.5 , 0.7]
    reg_alpha_l1 = [1.0]#[0.0, 0.01, 1.0]
    reg_lambda_l2 = [100]#[0.0, 1.0, 100]
    eval_metric = ['merror', 'mlogloss'] #merror #(wrong cases)/#(all cases) Multiclass classification error // mlogloss:
    tree_method = 'auto' #   
    esr = [20]
            
    print("\n[TXGB:] Building XGBoost Model")
    start_time = datetime.now()

    total = len(n_estimators) * len(max_depth) * len(learning_rate) * len(gamma) * len(subsample) * len(colsample_bytree) * len(reg_alpha_l1) * len(reg_lambda_l2) * len(eval_metric) * len(esr) 
    print('[TXGB:] Starting model training, {} iterations'.format(total))
    
    if save_results and not os.path.exists(dir_validation):
        os.makedirs(dir_validation)
    
    # Hiper-param data:
    data = []
    def getParamData(f):
        marksplit = '-'
        df_ = pd.read_csv(f)
        f = f[f.find('xgboost-'):]
        df_['ne']=   f.split(marksplit)[1]
        df_['md']=     f.split(marksplit)[2]
        df_['lr']=     f.split(marksplit)[3]
        df_['gm'] = f.split(marksplit)[4]
        df_['ss'] = f.split(marksplit)[5]
        df_['cst'] = f.split(marksplit)[6]
        df_['l1'] = f.split(marksplit)[7]
        df_['l2'] = f.split(marksplit)[8]
        df_['loss']  = f.split(marksplit)[9]
        df_['epoch'] = f.split(marksplit)[10]
        df_['features'] = f.split(marksplit)[11].split('.csv')[0]
        data.append(df_)

    pbar = tqdm(itertools.product(n_estimators, max_depth, learning_rate, gamma, subsample, colsample_bytree, 
                                  reg_alpha_l1, reg_lambda_l2, eval_metric, esr), 
                total=total, desc="[TXGB:] Model Training")
    for c in pbar:
        ne=c[0]
        md=c[1]
        lr=c[2]
        gm=c[3]
        ss=c[4]
        cst=c[5]
        l1=c[6]
        l2=c[7]
        loss=c[8]
        epch=c[9] 

        filename = os.path.join(dir_validation, 'xgboost-'+
                                concat_params(ne, md, lr, gm, ss, cst, l1, l2, loss, epch, features)+'.csv')

        if os.path.exists(filename):
            pbar.set_postfix_str('Skip ---> {}\n'.format(filename))
            getParamData(filename)
        else:

            pbar.set_postfix_str(print_params('ne, md, lr, gm, ss, cst, l1, l2, loss, epch',
                                              ne, md, lr, gm, ss, cst, l1, l2, loss, epch))

            xgboost = xg.XGBoostClassifier(n_estimators=ne,
                                           max_depth=md,
                                           lr=lr,
                                           gamma=gm,
                                           colsample_bytree=cst,
                                           subsample=ss,
                                           l1=l1,
                                           l2=l2,
                                           random_state=42,
                                           tree_method=tree_method,
                                           eval_metric=loss,
                                           early_stopping_rounds=epch,
                                           num_classes=num_classes)

            xgboost.fit(X_train, 
                        y_train, 
                        X_val,
                        y_val,
                        verbose=False)#,
                        #loss=loss, 
                        #early_stopping_rounds=epch)

            validation_report, y_pred = xgboost.predict(X_val, y_val)

            if save_results:
                validation_report.to_csv(filename, index=False)
            
            #validation_report['ne']= ne
            #validation_report['md']= md
            #validation_report['lr']= lr
            #validation_report['gm']= gm
            #validation_report['ss']= ss
            #validation_report['cst']= cst
            #validation_report['l1']= l1
            #validation_report['l2']= l2
            #validation_report['loss']= loss
            #validation_report['epoch']= epoch
            #validation_report['features']= str(features)
            #data.append(validation_report)
            data.append( update_report(validation_report, 'ne, md, lr, gm, ss, cst, l1, l2, loss, epoch, features',
                                       ne, md, lr, gm, ss, cst, l1, l2, loss, epch, features) )
                                       

    df_result = pd.concat(data)
    df_result.reset_index(drop=True, inplace=True)
    df_result.sort_values('acc', ascending=False, inplace=True)

    model = 0
    ne = int(df_result.iloc[model]['ne'])
    md = int(df_result.iloc[model]['md'])
    lr = float(df_result.iloc[model]['lr'])
    gm = float(df_result.iloc[model]['gm'])
    ss = float(df_result.iloc[model]['ss'])
    cst = float(df_result.iloc[model]['cst'])
    l1 = float(df_result.iloc[model]['l1'])
    l2 = int(df_result.iloc[model]['l2']) 
    loss = df_result.iloc[model]['loss']
    epch = int(df_result.iloc[model]['epoch'])

    filename = os.path.join(dir_evaluation, 'eval_xgboost-'+\
        concat_params(ne, md, lr, gm, ss, cst, l1, l2, loss, epch, features)+'.csv')

    print("[TXGB:] Filename: {}.".format(filename))

    if not os.path.exists(filename):
        print('[TXGB:] Creating a model to test set')
        print("[TXGB:] Parameters: " + print_params('ne, md, lr, gm, ss, cst, l1, l2, loss, epch, features',
                                              ne, md, lr, gm, ss, cst, l1, l2, loss, epch, features) )

        evaluate_report = []
        rounds = 10

        pbar = tqdm(range(rounds), desc="Model Testing")
        for e in pbar:
            pbar.set_postfix_str('Rounds {} of {}'.format(e, rounds))
            xgboost = xg.XGBoostClassifier(n_estimators=ne,
                                        max_depth=md,
                                        lr=lr,
                                        gamma=gm,
                                        subsample=ss,
                                        l1=l1,
                                        l2=l2,
                                        random_state=e,
                                        tree_method=tree_method,
                                        eval_metric=loss,
                                        early_stopping_rounds=epch,
                                        num_classes=num_classes)

            xgboost.fit(X_train, 
                        y_train, 
                        X_val,
                        y_val)#,
                        #loss=loss)#, 
                        #early_stopping_rounds=epch)

            #evaluate_report.append(xgboost.predict(X_test, y_test))
            eval_report, y_pred = xgboost.predict(X_test, y_test)
            evaluate_report.append(eval_report)
            
        if save_results:
            if not os.path.exists(dir_evaluation):
                os.makedirs(dir_evaluation)
                
            evaluate_report = pd.concat(evaluate_report)
            evaluate_report.to_csv(filename, index=False)
            
        end_time = (datetime.now()-start_time).total_seconds() * 1000
        print('[TXGB:] Processing time: {} milliseconds. Done.'.format(end_time))
    else:
        print('[TXGB:] Model previoulsy built.')
        
    print('\n--------------------------------------\n')
