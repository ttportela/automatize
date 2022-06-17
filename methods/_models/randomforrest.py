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
from main import importer
importer(['S'], globals())

## TODO: Under Construction:
def model_rf(dir_path, dataset=''):
    importer(['S', 'RF'], globals())
    
    print("[Random Forrest:] Building classifier model.")
    classifier = RandomForestClassifier()

    keys, vocab_size, num_classes, max_length, x_train, y_train, x_test, y_test = loadData(dir_path, dataset)
    
    nx, nsamples, ny = np.shape(x_train)
    x_train = x_train.reshape((nsamples,nx*ny)) # wrong
    nx, nsamples, ny = np.shape(x_test)
    x_test = x_test.reshape((nsamples,nx*ny))

    classifier = RandomForestClassifier(n_estimators=500, n_jobs = -1, random_state = 1, criterion = 'gini', bootstrap=True)
    classifier.fit(x_train, y_train)
    print("[Random Forrest:] Done.")

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