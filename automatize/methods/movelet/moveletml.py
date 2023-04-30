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
importer(['S', 'MC.report'], globals()) #, 'K'

def Approach1(X_train, y_train, X_test, y_test, par_batch_size, par_epochs, par_lr, par_dropout, save_results, dir_path, modelfolder='model') :
    
#     from ..main import importer
    importer(['S', 'NN'], globals())
    
#     from tensorflow.keras.models import Sequential
#     from tensorflow.keras.layers import Dense, Dropout
#     from tensorflow.keras.optimizers import Adam
#     import pandas as pd
#     import os
        
    nattr = len(X_train[1,:])    
    
    # Scaling y and transforming to keras format
#     from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train) 
    y_test = le.transform(y_test)
#     from tensorflow.keras.utils import to_categorical
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    nclasses = len(le.classes_)
    
    #Initializing Neural Network
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu', input_dim = (nattr)))
    classifier.add(Dropout( par_dropout ))
    # Adding the output layer   
    classifier.add(Dense(units = nclasses, kernel_initializer = 'uniform', activation = 'softmax'))
    # Compiling Neural Network
#     adam = Adam(lr=par_lr) # TODO: check for old versions...
    adam = Adam(learning_rate=par_lr)
    classifier.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy','top_k_categorical_accuracy',f1])
    # Fitting our model 
    history = classifier.fit(X_train, y_train1, validation_data = (X_test, y_test1), batch_size = par_batch_size, epochs = par_epochs)
    # ---------------------------------------------------------------------------------
    
    # ---------------------------------------------------------------------------------
    if (save_results) :
        if not os.path.exists(os.path.join(dir_path, modelfolder)):
            os.makedirs(os.path.join(dir_path, modelfolder))
        classifier.save(os.path.join(dir_path, modelfolder, 'model_approach1.h5'))
#         from numpy import argmax
        y_test_true_dec = le.inverse_transform(argmax(y_test1, axis = 1))
        y_test_pred_dec =  le.inverse_transform(argmax( classifier.predict(X_test) , axis = 1)) 
        report = classification_report(y_test_true_dec, y_test_pred_dec, output_dict=True)
        classification_report_dict2csv(report, os.path.join(dir_path, modelfolder, 'model_approach1_report.csv'),"Approach1")            
        pd.DataFrame(history.history).to_csv(os.path.join(dir_path, modelfolder, "model_approach1_history.csv"))
        pd.DataFrame(y_test_true_dec,y_test_pred_dec).to_csv(os.path.join(dir_path, modelfolder, 'model_approach1_prediction.csv'), header = True)    
    
# --------------------------------------------------------------------------------------
def Approach2(X_train, y_train, X_test, y_test, par_batch_size, lst_par_epochs, lst_par_lr, par_dropout, save_results, dir_path, modelfolder='model') :
    
#     from tensorflow.keras.models import Sequential
#     from tensorflow.keras.layers import Dense, Dropout
#     from tensorflow.keras.optimizers import Adam
#     import pandas as pd
#     import os
    
#     from ..main import importer
    importer(['S', 'MLP'], globals())
        
    nattr = len(X_train[1,:])    

    # Scaling y and transforming to keras format
#     from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train) 
    y_test = le.transform(y_test)
#     from tensorflow.keras.utils import to_categorical
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
#             from numpy import argmax
            y_test_true_dec = le.inverse_transform(argmax(y_test1, axis = 1))
            y_test_pred_dec =  le.inverse_transform(argmax( model.predict(X_test) , axis = 1)) 
            report = classification_report(y_test_true_dec, y_test_pred_dec, output_dict=True)
            classification_report_dict2csv(report, os.path.join(dir_path, modelfolder, 'model_approach2_report_Step'+str(k+1)+'.csv'), "Approach2_Step"+str(k+1)) 
            pd.DataFrame(history.history).to_csv(os.path.join(dir_path, modelfolder, 'model_approach2_history_Step'+str(k+1)+'.csv'))
            pd.DataFrame(y_test_true_dec,y_test_pred_dec).to_csv(os.path.join(dir_path, modelfolder, 'model_approach2_prediction_Step'+str(k+1)+'.csv'), header = True)  
    

def ApproachRF(X_train, y_train, X_test, y_test, n_trees_set, save_results, dir_path, modelfolder='model') :
        
#     from ..main import importer
    importer(['S', 'RF'], globals())
    
#     import os
#     import pandas as pd
    
#     from sklearn.ensemble import RandomForestClassifier    
    # ---------------------------------------------------------------------------------
    
    lines = list()
    
    for n_tree in n_trees_set:
        classifier = RandomForestClassifier(verbose=0, n_estimators = n_tree, n_jobs = -1, random_state = 1, criterion = 'gini', bootstrap=True)
        classifier.fit(X_train, y_train)
        acc = classifier.score(X_test,y_test)
        y_predicted = classifier.predict(X_test)
        accTop5 = calculateAccTop5(classifier, X_test, y_test, 5)
        line=[n_tree,acc,accTop5]
        lines.append(line)
        print(line)
        
        # ---------------------------------------------------------------------------------
        if (save_results) :
            if not os.path.exists(os.path.join(dir_path, modelfolder)):
                os.makedirs(os.path.join(dir_path, modelfolder))
                            
            report = classification_report(y_test, classifier.predict(X_test), output_dict=True)
            classification_report_dict2csv(report, os.path.join(dir_path, modelfolder, "model_approachRF"+ str(n_tree) +"_report.csv"),"RF")        
            pd.DataFrame(lines).to_csv(os.path.join(dir_path, modelfolder, "model_approachRF"+ str(n_tree) +"_history.csv"))
            pd.DataFrame(y_test, y_predicted).to_csv(os.path.join(dir_path, modelfolder, "model_approachRF"+ str(n_tree) +"_prediction.csv"), header = False) 
    
    print(lines)
    
    
def ApproachRFHP(X_train, y_train, X_test, y_test, save_results, dir_path, modelfolder='model') :
        
#     import os
#     import pandas as pd
#     import numpy as np
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.model_selection import RandomizedSearchCV
#     from ..main import importer
    importer(['S', 'RFHP'], globals())
        
    # ---------------------------------------------------------------------------------
    # Number of trees in random forest
    n_estimators = [300,350,400,450,500,550,600]    
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [30]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2,4,6]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [2,3,4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    criterion = ['entropy','gini']
    
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'criterion': criterion}
        
    rf = RandomForestClassifier(verbose=0, random_state = 1)
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=1, n_jobs = -1)
    rf_random.fit(X_train, y_train)
    
    print (rf_random.best_params_)
    
    classifier = rf_random.best_estimator_
    acc = classifier.score(X_test,y_test)    
    y_predicted = classifier.predict(X_test)
    accTop5 = calculateAccTop5(classifier, X_test, y_test, 5)
    line=[acc,accTop5]
    print(line)
        
    # ---------------------------------------------------------------------------------
    if (save_results) :
        if not os.path.exists(os.path.join(dir_path, modelfolder)):
            os.makedirs(os.path.join(dir_path, modelfolder))
        report = classification_report(y_test, classifier.predict(X_test), output_dict=True)
        classification_report_dict2csv(report, os.path.join(dir_path, modelfolder, "model_approachRFHP_report.csv"),"RFHP") 
        pd.DataFrame(line).to_csv(os.path.join(dir_path, modelfolder, "model_approachRFHP_history.csv")) 
        pd.DataFrame(y_test, y_predicted).to_csv(os.path.join(dir_path, modelfolder, "model_approachRFHP_prediction.csv"), header = False) 
    
    
# ----------------------------------------------------------------------------------

def ApproachDT(X_train, y_train, X_test, y_test, save_results, dir_path, modelfolder='model') :
        
#     import os
#     import pandas as pd
    
#     from sklearn.tree import DecisionTreeClassifier    
#     from ..main import importer
    importer(['S', 'DT'], globals())
    # ---------------------------------------------------------------------------------
    
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    acc = classifier.score(X_test,y_test)
    accTop5 = calculateAccTop5(classifier, X_test, y_test, 5)
    line=[acc, accTop5]
    print(line)
    
    # ---------------------------------------------------------------------------------
    if (save_results) :
        if not os.path.exists(os.path.join(dir_path, modelfolder)):
            os.makedirs(os.path.join(dir_path, modelfolder))
        report = classification_report(y_test, classifier.predict(X_test), output_dict=True)
        classification_report_dict2csv(report, os.path.join(dir_path, modelfolder, "model_approachDT_report.csv"),"DT") 
        pd.DataFrame(line).to_csv(os.path.join(dir_path, modelfolder, "model_approachDT_history.csv")) 
# ----------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------

def ApproachSVC(X_train, y_train, X_test, y_test, save_results, dir_path, modelfolder='model') :
        
#     import os
#     import pandas as pd
    
#     from sklearn import svm 
#     from ..main import importer
    importer(['S', 'SVC'], globals())  
    # ---------------------------------------------------------------------------------
    
    classifier = SVC(kernel="linear", probability=True)
    classifier.fit(X_train, y_train)
    acc = classifier.score(X_test,y_test)
    accTop5 = calculateAccTop5(classifier, X_test, y_test, 5)
    line=[acc, accTop5]
    print(line)
    
    # ---------------------------------------------------------------------------------
    if (save_results) :
        if not os.path.exists(os.path.join(dir_path, modelfolder)):
            os.makedirs(os.path.join(dir_path, modelfolder))
        report = classification_report(y_test, classifier.predict(X_test), output_dict=True)
        classification_report_dict2csv(report,os.path.join(dir_path, modelfolder, "model_approachSVC_report.csv"),"SVC") 
        pd.DataFrame(line).to_csv(os.path.join(dir_path, modelfolder, "model_approachSVC_history.csv")) 
# ----------------------------------------------------------------------------------

def ApproachMLP(X_train, y_train, X_test, y_test, par_batch_size, par_epochs, par_lr, par_dropout, save_results, dir_path, modelfolder='model') :
    
#     from tensorflow.keras.models import Sequential
#     from tensorflow.keras.layers import Dense, Dropout
#     from tensorflow.keras.optimizers import Adam
#     import pandas as pd
#     import os
#     from ..main import importer
    importer(['S', 'MLP'], globals())
        
    nattr = len(X_train[1,:])    

    # Scaling y and transforming to keras format
#     from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train) 
    y_test = le.transform(y_test)
#     from tensorflow.keras.utils import to_categorical
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    nclasses = len(le.classes_)
#     from tensorflow.keras import regularizers
    
    #Initializing Neural Network
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 100, kernel_initializer = 'uniform', kernel_regularizer= regularizers.l2(0.02), activation = 'relu', input_dim = (nattr)))
    #classifier.add(BatchNormalization())
    classifier.add(Dropout( par_dropout )) 
    # Adding the output layer       
    classifier.add(Dense(units = nclasses, kernel_initializer = 'uniform', activation = 'softmax'))
    # Compiling Neural Network
#     adam = Adam(lr=par_lr) # TODO: check for old versions...
    adam = Adam(learning_rate=par_lr)
    classifier.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy','top_k_categorical_accuracy',f1])
    # Fitting our model 
    history = classifier.fit(X_train, y_train1, validation_data = (X_test, y_test1), batch_size = par_batch_size, epochs = par_epochs)
    # ---------------------------------------------------------------------------------
    
    # ---------------------------------------------------------------------------------
    if (save_results) :
        if not os.path.exists(os.path.join(dir_path, modelfolder)):
            os.makedirs(os.path.join(dir_path, modelfolder))
        classifier.save(os.path.join(dir_path, modelfolder, 'model_MLP.h5'))
#         from numpy import argmax
        y_test_true_dec = le.inverse_transform(argmax(y_test1, axis = 1))
        y_test_pred_dec =  le.inverse_transform(argmax( classifier.predict(X_test) , axis = 1)) 
        report = classification_report(y_test_true_dec, y_test_pred_dec, output_dict=True)
        classification_report_dict2csv(report,os.path.join(dir_path, modelfolder, "model_approachMLP_report.csv"),"MLP") 
        pd.DataFrame(history.history).to_csv(os.path.join(dir_path, modelfolder, "model_MLP_history.csv"))
        pd.DataFrame(y_test_true_dec, y_test_pred_dec).to_csv(os.path.join(dir_path, modelfolder, "model_MLP_prediction.csv"), header = False)    
        

# Importing the Keras libraries and packages (Verificar de onde foi pego este codigo
# from tensorflow.keras import metrics
# from tensorflow.keras import backend as K
