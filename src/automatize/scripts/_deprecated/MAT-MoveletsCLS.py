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
import pandas as pd
import glob2 as glob

# from main import importer
# importer(['S'], globals())

if len(sys.argv) < 2:
    print('Please run as:')
    print('\tMoveletsCLS.py', 'PATH TO FOLDER', 'FOLDER')
    print('Example:')
    print('\tMoveletsCLS.py', '"./results"', '"HiPerMovelets"')
    exit()

results_path = sys.argv[1]
prefix   = sys.argv[2]

# --------------------------------------------------------------------------------------
def mergeDatasets(dir_path, file='train.csv'):
    files = [i for i in glob.glob(os.path.join(dir_path, '*', '**', file))]

    print("Loading files - " + file)
    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f).drop('class', axis=1) for f in files[:len(files)-1]], axis=1)
    combined_csv = pd.concat([combined_csv, pd.read_csv(files[len(files)-1])], axis=1)
    #export to csv
    print("Writing "+file+" file")
    combined_csv.to_csv(os.path.join(dir_path, file), index=False)
    
    print("Done.")

# 1 - JOIN RESULTS BY CLASS:
mergeDatasets(os.path.join(results_path, prefix), 'train.csv')
mergeDatasets(os.path.join(results_path, prefix), 'test.csv')

# --------------------------------------------------------------------------------------
def loadData(dir_path):
    import os
#     import sys
#     import numpy as np
    import pandas as pd
#     import glob2 as glob
#     from datetime import datetime

    from sklearn import preprocessing
    
    print("Loading train and test data from... " + dir_path)
    dataset_train = pd.read_csv(os.path.join(dir_path, "train.csv"))
    dataset_test  = pd.read_csv(os.path.join(dir_path, "test.csv"))
#     n_jobs = N_THREADS
    print("Done.")

    nattr = len(dataset_train.iloc[1,:])
    print("Number of attributes: " + str(nattr))
    
    if (len( set(dataset_test.columns).symmetric_difference(set(dataset_train.columns)) ) > 0):
        print('*ERROR* Divergence in train and test columns:', 
              len(dataset_train.columns), 'train and', len(dataset_test.columns), 'test')

    # Separating attribute data (X) than class attribute (y)
    X_train = dataset_train.iloc[:, 0:(nattr-1)].values
    y_train = dataset_train.iloc[:, (nattr-1)].values

    X_test = dataset_test.iloc[:, 0:(nattr-1)].values
    y_test = dataset_test.iloc[:, (nattr-1)].values

    # Replace distance 0 for presence 1
    # and distance 2 to non presence 0
    X_train[X_train == 0] = 1
    X_train[X_train == 2] = 0
    X_test[X_test == 0] = 1
    X_test[X_test == 2] = 0
    
    # Scaling data
    min_max_scaler = preprocessing.MinMaxScaler()
#     min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test

def Classifier_MLP(dir_path, save_results = True, modelfolder='model', X_train = None, y_train = None, X_test = None, y_test = None):
    from datetime import datetime
    # --------------------------------------------------------------------------------
    if X_train is None:
        X_train, y_train, X_test, y_test = loadData(dir_path)
     
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
    
    time = datetime.now()
    Approach2(X_train, y_train, X_test, y_test, par_batch_size, lst_par_epochs, lst_par_lr, par_droupout, save_results, dir_path, modelfolder)
    time = (datetime.now()-time).total_seconds() * 1000
    # ---------------------------------------------------------------------------------
    print("Done. " + str(time) + " milliseconds")
    print("---------------------------------------------------------------------------------")
    return time

def Approach2(X_train, y_train, X_test, y_test, par_batch_size, lst_par_epochs, lst_par_lr, par_dropout, save_results, dir_path, modelfolder='model') :
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    import pandas as pd
    import os
        
    nattr = len(X_train[1,:])    

    # Scaling y and transforming to keras format
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train) 
    y_test = le.transform(y_test)
    from tensorflow.keras.utils import to_categorical
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    nclasses = len(le.classes_)
    
    #Initializing Neural Network
    model = Sequential()
    # Adding the input layer and the first hidden layer
    model.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'linear', input_dim = (nattr)))
    model.add(Dropout( par_dropout ))
    # Adding the output layer
    model.add(Dense(units = nclasses, kernel_initializer = 'uniform', activation = 'softmax'))
    # Compiling Neural Network
    
    k = len(lst_par_epochs)
    
    for k in range(0,k) :
           
#         adam = Adam(lr=lst_par_lr[k]) # TODO: check for old versions...
        adam = Adam(learning_rate=lst_par_lr[k])
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy','top_k_categorical_accuracy',f1])
        history = model.fit(X_train, y_train1, validation_data = (X_test, y_test1), epochs=lst_par_epochs[k], batch_size=par_batch_size)
    
        # ---------------------------------------------------------------------------------
        if (save_results) :
            if not os.path.exists(os.path.join(dir_path, modelfolder)):
                os.makedirs(os.path.join(dir_path, modelfolder))
            model.save(os.path.join(dir_path, modelfolder, 'model_approach2_Step'+str(k+1)+'.h5'))
            from numpy import argmax
            from sklearn.metrics import classification_report
            y_test_true_dec = le.inverse_transform(argmax(y_test1, axis = 1))
            y_test_pred_dec =  le.inverse_transform(argmax( model.predict(X_test) , axis = 1)) 
            report = classification_report(y_test_true_dec, y_test_pred_dec )
            classification_report_csv(report, os.path.join(dir_path, modelfolder, 'model_approach2_report_Step'+str(k+1)+'.csv'), "Approach2_Step"+str(k+1)) 
            pd.DataFrame(history.history).to_csv(os.path.join(dir_path, modelfolder, 'model_approach2_history_Step'+str(k+1)+'.csv'))
            pd.DataFrame(y_test_true_dec,y_test_pred_dec).to_csv(os.path.join(dir_path, modelfolder, 'model_approach2_prediction_Step'+str(k+1)+'.csv'), header = True)  

# Importing the Keras libraries and packages (Verificar de onde foi pego este codigo
from tensorflow.keras import metrics
from tensorflow.keras import backend as K
def f1(y_true, y_pred):
    
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# by Tarlis ---------------------------------------------------------------------------------
def calculateAccTop5(classifier, X_test, y_test, K ):
    import numpy as np
    K = K if len(y_test) > K else len(y_test)
    
    y_test_pred = classifier.predict_proba(X_test)
    order=np.argsort(y_test_pred, axis=1)
    n=classifier.classes_[order[:, -K:]]
    soma = 0;
    for i in range(0,len(y_test)) :
        if ( y_test[i] in n[i,:] ) :
            soma = soma + 1
    accTopK = soma / len(y_test)
    
    return accTopK

def classification_report_csv(report, reportfile, classifier):
    report_data = []
    lines = report.split('\n')   
    for line in lines[2:(len(lines)-3)]:
        row_data = line.split()
        row = {}  
        
        if row_data == []:
            break
            
        row["class"] = row_data[0]
        row["classifier"] = classifier
        row["precision"] = float(row_data[1])
        row["recall"] = float(row_data[2])
        row["f1_score"] = float(row_data[3])
        row["support"] = float(row_data[4])
        print(row)
        report_data.append(row)
    import pandas as pd
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(reportfile, index = False)            
            
# 2 - DO CLASSIFICATION:
save_results = True
modelfolder='model'

print('Starting analysis in: ', results_path, prefix)
Classifier_MLP(os.path.join(results_path, prefix), save_results, modelfolder)