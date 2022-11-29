# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
import sys, os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from automatize.main import importer
importer(['S'], globals())

# --------------------------------------------------------------------------------
# ANALYSIS By Ensemble Learning Models
def TEC(data_path, results_path, ensembles, dataset='specific', save_results=True, modelfolder='model_ensemble', random_seed=1):
#     from ..main import importer
    importer(['S', 'datetime', 'tf', 'TEC.report', 'KerasClassifier', 'readDataset'], globals())
    
#     import os
#     import pandas as pd
#     import numpy as np
#     from datetime import datetime
#     import tensorflow
    tf.keras.backend.clear_session()
    
    np.random.seed(seed=random_seed)
    tf.random.set_seed(random_seed)
    
    print('[Ensemble]: Loading base data description.')
    if dataset == '':
        TRAIN_FILE = os.path.join(data_path, 'train.csv')
        TEST_FILE  = os.path.join(data_path, 'test.csv')
    else:
        TRAIN_FILE = os.path.join(data_path, dataset+'_train.csv')
        TEST_FILE  = os.path.join(data_path, dataset+'_test.csv')
        
#     print(TRAIN_FILE)
    df_test = readDataset(TEST_FILE, missing='-999') #pd.read_csv(TEST_FILE)
    y_test = df_test.drop_duplicates(subset=['tid', 'label'],
                                       inplace=False)['label'].values

    keys = list(pd.unique(y_test))
    keys.sort()
    y_labels = [keys.index(x) for x in y_test]

#     from keras.wrappers.scikit_learn import KerasClassifier
    
    time = datetime.now()
    # create the sub models
    models = dict()
    estimators = []
    print('[Ensemble]: '+', '.join(ensembles.keys()))
    for method, folder in ensembles.items():
        ctime = datetime.now()
        y_pred = []
        model = []
        if method == 'movelets_mlp':
#             from ..main import importer
            importer(['TEC.MLP'], globals())
#             from ensemble_models.movelets import model_movelets_mlp
#             from numpy import argmax
            model, x_test, y_test_pred, y_test_true = model_movelets_mlp(folder)
#             y_test_pred_dec = argmax( y_test_pred_dec , axis = 1)
#             y_pred = [keys.index(x) for x in y_test_pred_dec]
#             return y_labels, x_test, y_test_pred, y_test_true
#             from Methods import f1
            models[method] = KerasClassifier(model)
            model = y_test_pred
#             model = model.predict(x_test)
#             y_pred = [np.argmax(f) for f in y_test_pred_dec] 

        if method == 'movelets_nn' or method == 'movelets':
#             from ..main import importer
            importer(['TEC.NN'], globals())
#             from ensemble_models.movelets import model_movelets_nn
            model, x_test = model_movelets_nn(folder)
#             from Methods import f1
            models[method] = KerasClassifier(model)
            model = model.predict(x_test)

        if method == 'marc':
#             from ..main import importer
            importer(['TEC.MARC'], globals())
#             from ensemble_models.marc2 import model_marc
            model, x_test = model_marc(folder, results_path, dataset)
            models[method] = KerasClassifier(model)
            model = model.predict(x_test)
            
        if method == 'poi' or method == 'npoi' or method == 'wnpoi':
#             from ..main import importer
            importer(['TEC.POIS'], globals())
#             from ensemble_models.poifreq import model_poifreq
            model, x_test = model_poifreq(folder)
            models[method] = KerasClassifier(model)
            model = model.predict(x_test)
            
#        if method == 'rf': # TODO
##             from ..main import importer
#            importer(['TEC.RF'], globals())
##             from ensemble_models.randomforrest import model_rf
#            model, x_test = model_rf(folder, dataset)
#            models[method] = model
#            model = model.predict(x_test)
#            
#        if method == 'rfhp': # TODO
##             from ..main import importer
#            importer(['TEC.RFHP'], globals())
##             from ensemble_models.randomforresthp import model_rfhp
#            model, x_test = model_rfhp(folder, dataset)
#            models[method] = model
#            model = model.predict(x_test)
        
#         print(method, 'ESTIMATORS:', model)
        y_pred = [np.argmax(f) for f in model]  
        ctime = (datetime.now()-ctime).total_seconds() * 1000
        
#         return model, y_pred
#         print(method, 'PRED:', y_pred)
        estimators.append(model) 
        ensembles[method] = get_line(y_labels, y_pred, model) + [ctime]
        print(method+': ', ensembles[method])
        print("---------------------------------------------------------------------------------")
    
#     print(estimators)
    final_pred = estimators[0]
    for i in range(1, len(estimators)):
        final_pred = final_pred + estimators[i]
    y_pred = [np.argmax(f) for f in final_pred]
    # ----------------------------------------------------------------------------------
    time = (datetime.now()-time).total_seconds() * 1000
    
    print('[Ensemble]: Final results.')
    print(ensembles)
    line=get_line(y_labels, y_pred, final_pred) + [time]
    ensembles['TEC'] = line
    print('[Ensemble]:', line)
    
    # Correct Class names:
    y_labels = [keys[f] for f in y_labels]
    y_pred = [keys[f] for f in y_pred]
    ensembles = [[key] + ensembles[key] for key in ensembles.keys()]
    # ---------------------------------------------------------------------------------
    if (save_results) :
        if not os.path.exists(os.path.join(results_path, modelfolder)):
            os.makedirs(os.path.join(results_path, modelfolder))
#         from sklearn.metrics import classification_report
#         from Methods import classification_report_csv
        report = classification_report(y_labels, y_pred, output_dict=True, zero_division=False)
        classification_report_dict2csv(report, os.path.join(results_path, modelfolder, "model_approachEnsemble_report.csv"),"Ensemble") 
        pd.DataFrame(ensembles, columns=['classifier', 'accuracy', 'f1_score', 'precision', 'recall', 'accTop5', 'time']).to_csv(os.path.join(results_path, modelfolder, "model_approachEnsemble_history.csv")) 

    # ---------------------------------------------------------------------------------
    print("Done. " + str(time) + " milliseconds")
    print("---------------------------------------------------------------------------------")
    return time

def TEC2(data_path, results_path, ensembles, dataset='specific', save_results=True, modelfolder='model_ensemble', random_seed=1):
#     from ..main import importer
    importer(['S', 'datetime', 'tf', 'KerasClassifier', 'A2'], globals())
    # V2.0 - Concatenate data from POI-S and Movelets (pois, movelets)
#     import os
#     import pandas as pd
#     import numpy as np
#     from datetime import datetime
#     import tensorflow
    tf.keras.backend.clear_session()
    
    np.random.seed(seed=random_seed)
    tf.random.set_seed(random_seed)

#     from keras.wrappers.scikit_learn import KerasClassifier
    
    time = datetime.now()
    # Concatenate:
    dir_path = ensembles['pois']
    print("Loading train and test data from... " + dir_path)
    x_train_1 = pd.read_csv(dir_path+'-x_train.csv', header=None)
    y_train = pd.read_csv(dir_path+'-y_train.csv')

    x_test_1 = pd.read_csv(dir_path+'-x_test.csv', header=None)
    y_test = pd.read_csv(dir_path+'-y_test.csv')
    
    dir_path = ensembles['movelets']
    print("Loading train and test data from... " + dir_path)
    dataset_train = pd.read_csv(os.path.join(dir_path, "train.csv"))
    dataset_test  = pd.read_csv(os.path.join(dir_path, "test.csv"))

    nattr = len(dataset_train.iloc[1,:])
    print("Number of attributes: " + str(nattr))

    # Separating attribute data (X) than class attribute (y)
    x_train_2 = dataset_train.iloc[:, 0:(nattr-1)].values
#     y_train_2 = dataset_train.iloc[:, (nattr-1)].values
    x_test_2 = dataset_test.iloc[:, 0:(nattr-1)].values
#     y_test_2 = dataset_test.iloc[:, (nattr-1)].values

    # Replace distance 0 for presence 1
    # and distance 2 to non presence 0
    x_train_2[x_train_2 == 0] = 1
    x_train_2[x_train_2 == 2] = 0
    x_test_2[x_test_2 == 0] = 1
    x_test_2[x_test_2 == 2] = 0
    
    x_train = pd.concat([x_train_1, pd.DataFrame(x_train_2)], axis=1).values
    x_test = pd.concat([x_test_1, pd.DataFrame(x_test_2)], axis=1).values
    print("Done.")
    
    # ---------------------------------------------------------------------------
    # Neural Network - Definitions:
    par_droupout = 0.5
    par_batch_size = 200
    par_epochs = 80
    par_lr = 0.00095
    
    # Building the neural network-
    print("Building neural network")
    lst_par_epochs = [80,50,50,30,20]
    lst_par_lr = [0.00095,0.00075,0.00055,0.00025,0.00015]
    
#     from Methods import Approach2
    Approach2(x_train, y_train, x_test, y_test, par_batch_size, lst_par_epochs, lst_par_lr, par_droupout, save_results, results_path, modelfolder)
    # ----------------------------------------------------------------------------------
    time = (datetime.now()-time).total_seconds() * 1000
    # ---------------------------------------------------------------------------------
#     if (save_results) :
#         if not os.path.exists(os.path.join(results_path, modelfolder)):
#             os.makedirs(os.path.join(results_path, modelfolder))
#         from sklearn.metrics import classification_report
#         from Methods import classification_report_csv
#         report = classification_report(y_labels, y_pred)
#         classification_report_csv(report, os.path.join(results_path, modelfolder, "model_approachEnsemble_report.csv"),"Ensemble") 
#         pd.DataFrame(line).to_csv(os.path.join(results_path, modelfolder, "model_approachEnsemble_history.csv")) 

    # ---------------------------------------------------------------------------------
    print("Done. " + str(time) + " milliseconds")
    print("---------------------------------------------------------------------------------")
    return time

# # --------------------------------------------------------------------------------
# importer(['precision_score', 'recall_score', 'f1_score', 'accuracy_score'], locals())
importer(['metrics'], globals())

# Statistics:
def get_line(y_true, y_pred, y_test_pred=None):
    acc = accuracy(y_true, y_pred)
    f1  = f1_macro(y_true, y_pred)
    prec= precision_macro(y_true, y_pred)
    rec = recall_macro(y_true, y_pred)
    if y_test_pred is not None:
        accTop5 = calculateAccTop5(y_test_pred, y_pred, 5)
    else:
        accTop5 = 0
    line=[acc, f1, prec, rec, accTop5]
    return line

def precision_macro(y_true, y_pred):
#     from ..main import importer
#     importer(['precision_score'], locals())
#     from sklearn.metrics import precision_score
    return precision_score(y_true, y_pred, average='macro')
def recall_macro(y_true, y_pred):
#     from ..main import importer
#     importer(['recall_score'], locals())
#     from sklearn.metrics import recall_score
    return recall_score(y_true, y_pred, average='macro')
def f1_macro(y_true, y_pred):
#     from ..main import importer
#     importer(['f1_score'], locals())
#     from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average='macro')
def accuracy(y_true, y_pred):
#     from ..main import importer
#     importer(['accuracy_score'], locals())
#     from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred, normalize=True)
# --------------------------------------------------------------------------------
def calculateAccTop5(y_test_pred, y_true, K):
#     from ..main import importer
#     importer(['np'], locals())
#     import numpy as np
    
    K = K if len(y_true) > K else len(y_true)
    
#     y_test_pred = classifier.predict_proba(X_test)
    order=np.argsort(y_test_pred, axis=1)
#     n=classifier.classes_[order[:, -K:]]
    n=order[:, -K:]
    soma = 0;
    for i in range(0,len(y_true)) :
        if ( y_true[i] in n[i,:] ) :
            soma = soma + 1
    accTopK = soma / len(y_true)
    
    return accTopK