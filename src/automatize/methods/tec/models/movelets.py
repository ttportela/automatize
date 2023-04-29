# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
@author: Carlos Andres Ferreira (adapted)
'''
import sys, os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from main import importer #, display
importer(['S'], globals())

def model_movelets_mlp(folder):
    importer(['S', 'report', 'loadData', 'MLP'], globals())
    
    
    X_train, y_train, X_test, y_test = loadData(folder) # temp
    
    labels = y_test

    # ---------------------------------------------------------------------------
    # Neural Network - Definitions:
    par_dropout = 0.5
    par_batch_size = 200
    par_epochs = 80
    par_lr = 0.00095
    
    # Building the neural network-
    print("[Movelets:] Building neural network")
    lst_par_epochs = [80,50,50,30,20]
    lst_par_lr = [0.00095,0.00075,0.00055,0.00025,0.00015]

    nattr = len(X_train[1,:])       

    # Scaling y and transforming to keras format
#     from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train) 
    y_test = le.transform(y_test)
#     from keras.utils import to_categorical
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
#         adam = Adam(lr=lst_par_lr[k])
        adam = Adam(learning_rate=lst_par_lr[k])
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy','top_k_categorical_accuracy',f1])
        history = model.fit(X_train, y_train1, validation_data = (X_test, y_test1), epochs=lst_par_epochs[k], batch_size=par_batch_size, verbose=0)

    print("[Movelets:] OK")

#     from numpy import argmax
#     y_test_pred_dec =  le.inverse_transform(argmax( model.predict(X_test) , axis = 1))
    y_test_pred_dec = model.predict(X_test)
#     print(y_test_pred_dec)
    y_test_true_dec = le.inverse_transform(argmax(y_test1, axis = 1))
#     print(y_test_true_dec)
#     y_test_true_dec = [le.inverse_transform(x) for x in y_test1]
#     y_test_pred_dec = [le.inverse_transform(x) for x in y_test_pred_dec]
    return model, X_test, y_test_pred_dec, y_test_true_dec

def model_movelets_nn(folder):

#     from ..main import importer
    importer(['S', 'report', 'loadData', 'NN'], globals())
#     from keras.models import Sequential
#     from keras.layers import Dense, Dropout
#     from keras.optimizers import Adam
#     import os
#     import pandas as pd
#     from analysis import loadData
#     from Methods import classification_report, classification_report_csv, calculateAccTop5, f1
        
    X_train, y_train, X_test, y_test = loadData(folder) # temp
    
    labels = y_test

    # ---------------------------------------------------------------------------
    # Neural Network - Definitions:
    par_dropout = 0.5
    par_batch_size = 200
    par_epochs = 80
    par_lr = 0.00095

    # Building the neural network-
    print("[Movelets:] Building neural network")
    lst_par_epochs = [80,50,50,30,20]
    lst_par_lr = [0.00095,0.00075,0.00055,0.00025,0.00015]

    nattr = len(X_train[1,:])    

    # Scaling y and transforming to keras format
#     from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train) 
    y_test = le.transform(y_test)
#     from keras.utils import to_categorical
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    nclasses = len(le.classes_)
#     from keras import regularizers

    #Initializing Neural Network
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 100, kernel_initializer = 'uniform', kernel_regularizer= regularizers.l2(0.02), activation = 'relu', input_dim = (nattr)))
    #classifier.add(BatchNormalization())
    classifier.add(Dropout( par_dropout )) 
    # Adding the output layer       
    classifier.add(Dense(units = nclasses, kernel_initializer = 'uniform', activation = 'softmax'))
    # Compiling Neural Network
#     adam = Adam(lr=par_lr)
    adam = Adam(learning_rate=par_lr)
    classifier.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy','top_k_categorical_accuracy'])
#     classifier.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy','top_k_categorical_accuracy',f1])
    # Fitting our model 
    history = classifier.fit(X_train, y_train1, validation_data = (X_test, y_test1), batch_size = par_batch_size, epochs = par_epochs, verbose=0)

    print("[Movelets:] OK")

#     print(labels)
#     return classifier.predict(X_test)
    return classifier, X_test