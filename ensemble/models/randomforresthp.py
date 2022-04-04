# -*- coding: utf-8 -*-
'''
Automatize: Multi-Aspect Trajectory Data Mining Tool Library
The present application offers a tool, called AutoMATize, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. The AutoMATize integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
License GPL v.3 or superior

@author: Tarlis Portela
'''
from ...main import importer #, display
importer(['S'], globals())

## TODO: Under Construction:
def model_rfhp(dir_path, dataset=''):
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
    
    print("[RFHP:] Building classifier model.")
#     classifier = RandomForestClassifier()

    keys, vocab_size, num_classes, max_length, x_train, y_train, x_test, y_test = loadData(dir_path, dataset)
    
    nx, nsamples, ny = np.shape(x_train)
    x_train = x_train.reshape((nsamples,nx*ny)) # wrong
    nx, nsamples, ny = np.shape(x_test)
    x_test = x_test.reshape((nsamples,nx*ny))
    
    rf = RandomForestClassifier(verbose=0, random_state = 1)
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=1, n_jobs = -1)
    rf_random.fit(x_train, y_train)
    
#     print (rf_random.best_params_)
    
    classifier = rf_random.best_estimator_
#     acc = classifier.score(X_test,y_test)    
#     y_predicted = classifier.predict(X_test)
    
#     print (classifier.best_params_)
    print("[RFHP:] Done.")

#     return classifier.predict_proba(x_test)
    return classifier, x_test

# --------------------------------------------------------------------------------
# Load data:
def loadData(dir_path, dataset='', tid_col='tid',
                     label_col='label', geo_precision=8, drop=[]):

#     from ..main import importer
    importer(['S', 'encoding'], globals())
#     import os
#     import pandas as pd
#     import numpy as np
#     from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#     from marc.core.utils.geohash import bin_geohash
    
    print("Loading data from file(s) " + dir_path + "*")
    if dataset == '':
        df_train = pd.read_csv(os.path.join(dir_path, 'train.csv'))
        df_test = pd.read_csv(os.path.join(dir_path, 'test.csv'))
    else:
        df_train = pd.read_csv(os.path.join(dir_path, dataset+'_train.csv'))
        df_test = pd.read_csv(os.path.join(dir_path, dataset+'_test.csv'))
#     print("Loading data from file(s) " + dir_path + "... ")
#     df_train = pd.read_csv(os.path.join(dir_path, 'train.csv'))
#     df_test = pd.read_csv(os.path.join(dir_path, 'test.csv'))
    df = df_train.copy().append(df_test)
    tids_train = df_train[tid_col].unique()

    keys = list(df.keys())
    vocab_size = {}
    keys.remove(tid_col)

    for col in drop:
        if col in keys:
            keys.remove(col)
            print("Column '" + col + "' dropped " +
                       "from input file!")
        else:
            print("Column '" + col + "' cannot be " +
                       "dropped because it was not found!")

    num_classes = len(set(df[label_col]))
    count_attr = 0
    lat_lon = False

    if 'lat' in keys and 'lon' in keys:
        keys.remove('lat')
        keys.remove('lon')
        lat_lon = True
        count_attr += geo_precision * 5
        print("Attribute Lat/Lon: " +
                   str(geo_precision * 5) + "-bits value")

    for attr in keys:
        if attr != label_col:
            df[attr] = LabelEncoder().fit_transform(df[attr])
            vocab_size[attr] = max(df[attr]) + 1

            values = len(set(df[attr]))
            count_attr += values
            print("Attribute '" + attr + "': " +
                       str(values) + " unique values")

    print("Total of attribute/value pairs: " +
               str(count_attr))
    keys.remove(label_col)

    x = [[] for key in keys]
    y = []
    idx_train = []
    idx_test = []
    max_length = 0
    trajs = len(set(df[tid_col]))

    if lat_lon:
        x.append([])

    for idx, tid in enumerate(set(df[tid_col])):
        traj = df.loc[df[tid_col].isin([tid])]
        features = np.transpose(traj.loc[:, keys].values)

        for i in range(0, len(features)):
            x[i].append(features[i])

        if lat_lon:
            loc_list = []
            for i in range(0, len(traj)):
                lat = traj['lat'].values[i]
                lon = traj['lon'].values[i]
                loc_list.append(bin_geohash(lat, lon, geo_precision))
            x[-1].append(loc_list)

        label = traj[label_col].iloc[0]
        y.append(label)

        if tid in tids_train:
            idx_train.append(idx)
        else:
            idx_test.append(idx)

        if traj.shape[0] > max_length:
            max_length = traj.shape[0]

    if lat_lon:
        keys.append('lat_lon')
        vocab_size['lat_lon'] = geo_precision * 5

    one_hot_y = OneHotEncoder().fit(df.loc[:, [label_col]])

    x = [np.asarray(f) for f in x]
    y = one_hot_y.transform(pd.DataFrame(y)).toarray()
    print("Loading data from files ... DONE!")
    
    x_train = np.asarray([f[idx_train] for f in x])
    y_train = y[idx_train]
    x_test = np.asarray([f[idx_test] for f in x])
    y_test = y[idx_test]

    print('Trajectories:  ' + str(trajs))
    print('Labels:        ' + str(len(keys)))
    print('Train size:    ' + str(len(x_train[0]) / trajs))
    print('Test size:     ' + str(len(x_test[0]) / trajs))
    print('x_train shape: ' + str(x_train.shape))
    print('y_train shape: ' + str(y_train.shape))
    print('x_test shape:  ' + str(x_test.shape))
    print('y_test shape:  ' + str(y_test.shape))
    
#     from keras.preprocessing.sequence import pad_sequences
    x_train = np.asarray([pad_sequences(f, max_length, padding='pre') for f in x_train])
    x_test  = np.asarray([pad_sequences(f, max_length, padding='pre') for f in x_test])

    return (keys, vocab_size, num_classes, max_length,
            x_train, y_train,
            x_test, y_test)