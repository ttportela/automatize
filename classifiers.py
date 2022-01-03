'''
Created on May 16, 2018

@author: andres
'''
# from sklearn.metrics import classification_report
# from _dummy_thread import exit
from .main import importer #, display
importer(['S'], globals())

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
        report = classification_report(y_test_true_dec, y_test_pred_dec )
        classification_report_csv(report, os.path.join(dir_path, modelfolder, 'model_approach1_report.csv'),"Approach1")            
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
            report = classification_report(y_test_true_dec, y_test_pred_dec )
            classification_report_csv(report, os.path.join(dir_path, modelfolder, 'model_approach2_report_Step'+str(k+1)+'.csv'), "Approach2_Step"+str(k+1)) 
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
                            
            report = classification_report(y_test, classifier.predict(X_test) )
            classification_report_csv(report, os.path.join(dir_path, modelfolder, "model_approachRF"+ str(n_tree) +"_report.csv"),"RF")        
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
        report = classification_report(y_test, classifier.predict(X_test) )
        classification_report_csv(report, os.path.join(dir_path, modelfolder, "model_approachRFHP_report.csv"),"RFHP") 
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
        report = classification_report(y_test, classifier.predict(X_test) )
        classification_report_csv(report, os.path.join(dir_path, modelfolder, "model_approachDT_report.csv"),"DT") 
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
        report = classification_report(y_test, classifier.predict(X_test) )
        classification_report_csv(report,os.path.join(dir_path, modelfolder, "model_approachSVC_report.csv"),"SVC") 
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
        report = classification_report(y_test_true_dec, y_test_pred_dec )
        classification_report_csv(report,os.path.join(dir_path, modelfolder, "model_approachMLP_report.csv"),"MLP") 
        pd.DataFrame(history.history).to_csv(os.path.join(dir_path, modelfolder, "model_MLP_history.csv"))
        pd.DataFrame(y_test_true_dec, y_test_pred_dec).to_csv(os.path.join(dir_path, modelfolder, "model_MLP_prediction.csv"), header = False)    
        

# Importing the Keras libraries and packages (Verificar de onde foi pego este codigo
# from tensorflow.keras import metrics
# from tensorflow.keras import backend as K

def f1(y_true, y_pred):
#     from ..main import importer
    importer(['K'], globals())
    
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
# ----------------------------------------------------------------------------------


# def classification_report_csv(report, reportfile, classifier):
#     report_data = []
#     lines = report.split('\n')   
#     for line in lines[2:(len(lines)-3)]:
#         row = {}        
#         row_data = line.split()
#         row['class'] = row_data[0]
#         row['classifier'] = classifier
#         row['precision'] = float(row_data[1])
#         row['recall'] = float(row_data[2])
#         row['f1_score'] = float(row_data[3])
#         row['support'] = float(row_data[4])
#         print(row)
#         report_data.append(row)
#     import pandas as pd
#     dataframe = pd.DataFrame.from_dict(report_data)
#     dataframe.to_csv(reportfile, index = False)

# ---------------------------------------------------------------------------------
# def calculateAccTop5(classifier, X_test, y_test, K ):
#     import numpy as np
#     y_test_pred = classifier.predict_proba(X_test)
#     order=np.argsort(y_test_pred, axis=1)
#     n=classifier.classes_[order[:, -K:]]
#     soma = 0;
#     for i in range(0,len(y_test)) :
#         if ( y_test[i] in n[i,:] ) :
#             soma = soma + 1
#     accTopK = soma / len(y_test)
    
#     return accTopK


# by Tarlis ---------------------------------------------------------------------------------
def calculateAccTop5(classifier, X_test, y_test, K ):
#     import numpy as np
#     from ..main import importer
#     importer(['np'], locals())
    
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

# by Tarlis -----------------------------------------------------------------------------------------------
def classification_report_csv(report, reportfile, classifier):
#     from ..main import importer
#     importer(['pd'], locals())
    
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
#     import pandas as pd
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(reportfile, index = False)
