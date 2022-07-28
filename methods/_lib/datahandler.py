# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Jun, 2022
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
import sys, os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from main import importer


###############################################################################
#   LOAD DATA
###############################################################################
def loadTrajectories(dir_path, 
                     file_prefix='', 
                     tid_col='tid', 
                     class_col='label',
                     space_geohash=False,
                     geo_precision=8,  
                     features_encoding=True, 
                     y_one_hot_encodding=False,
                     split_test_validation=True):

    importer(['S', 'io', 'encoding'], globals(), {'preprocessing': ['trainAndTestSplit']}) #, modules={'preprocessing': ['readDataset', 'organizeFrame']})
    
    print('\n###########      DATA LOADING        ###########')
    if file_prefix == '':
        train_file = os.path.join(dir_path, 'train.csv')
        test_file  = os.path.join(dir_path, 'test.csv')
    else:
        train_file = os.path.join(dir_path, file_prefix+'train.csv')
        test_file  = os.path.join(dir_path, file_prefix+'test.csv')
        
    df_train = readDataset(os.path.dirname(train_file), file=os.path.basename(train_file), missing='-999')
    df_test = readDataset(os.path.dirname(test_file), file=os.path.basename(test_file), missing='-999')
    
    _, columns_order = organizeFrame(df_train, tid_col=tid_col, class_col=class_col)
    organizeFrame(df_test, tid_col=tid_col, class_col=class_col)
    
    df_train = df_train[columns_order]
    df_test  = df_test[columns_order]
    
#    df_ = pd.concat([df_train, df_test])
    
    features = list(df_train.keys())
    num_classes = len(set(df_train[class_col])) 
    count_attr = 0
    space = False

    # TODO:
    if space_geohash and ('lat' in features and 'lon' in features):
#        keys.remove('lat')
#        keys.remove('lon')
        space = True
        count_attr += geo_precision * 5
        print("Attribute Space: " +
                   str(geo_precision * 5) + "-bits value")

    for attr in features:
        if attr != class_col:
            values = len(set(df_train[attr]))
            count_attr += values
            print("Attribute '" + attr + "': " + str(values) + " unique values")

    print("Total of attribute/value pairs: " + str(count_attr))
    
    if split_test_validation:
        df_train, df_val = trainAndTestSplit('', df_train, train_size=0.75, tid_col=tid_col, class_col=class_col, outformats=[])
        data = [df_train, df_val, df_test]
    else:
        data = [df_train, df_test]

    
    X, y, dic_parameters = generate_X_y(data, 
                                        features_encoding=True,       
                                        y_one_hot_encodding=False)
    
    return X, y, features, num_classes, space, dic_parameters
    

def generate_X_y( data,
            features_encoding=True,
            y_one_hot_encodding=True,
            lat_col='lat', 
            lon_col='lon',
            tid_col='tid', 
            class_col='label' ):
    
    print('\n\n###########      DATA ENCODING        ###########')
    
    input_total = len(data)
    assert (input_total > 0), "ERR: data is not set or < 1"
    
    
    if input_total > 1:
#        print('... concat dataframe')
        df_ = pd.concat(data)
    else:
#        print('... df_ is data')
        df_ = data[0]
    
    assert isinstance(data, list) and isinstance(df_, pd.DataFrame), "ERR: inform data as array of pandas.Dataframe()"
    assert class_col in df_, "ERR: class_col in not on dataframe"
    assert tid_col in df_, "ERR: tid_col in not on dataframe"
    
    features = list(df_.columns)
    col_drop = [lat_col, lon_col, tid_col, class_col] 
    features = [x for x in features if x not in col_drop]
    
    max_lenght = df_.groupby(tid_col).agg({class_col:'count'}).max()[0]
    
    dic_tid = {}
    for i, d in enumerate(data):
        dic_tid[i] = d[tid_col].unique()
        print('tid_{}: {}'.format(i, len(dic_tid[i])))
    
    dic_parameters = {}
    if features_encoding == True:
        print('Encoding string data to integer')
        if len(features) > 0:
            dic_parameters['encode_features'] = label_encoding(df_, col=features)

    col_groupby = {}
    for c in features:
        col_groupby[c] = list
    col_groupby[class_col] = 'first'
    
    traj = df_.groupby(tid_col, sort=False).agg(col_groupby)

    if y_one_hot_encodding == True:
        print('One Hot encoding on label y')
        ohe_y = OneHotEncoder()
        y = ohe_y.fit_transform(pd.DataFrame(traj[class_col])).toarray()
        dic_parameters['encode_y'] = ohe_y 
    else:
        print('Label encoding on label y')
        le_y = LabelEncoder()
        #y = np.array(le_y.fit_transform(pd.DataFrame(traj[class_col])))
        y = np.array(le_y.fit_transform(pd.DataFrame(traj[class_col]).values.ravel()))
        dic_parameters['encode_y'] = le_y
        
    if input_total == 1:
        y = np.array(y, ndmin=2)
    elif input_total > 1:
        start = 0
        end   = 0

        y_aux = []
        for i in range(0, input_total):
            end = end + len(dic_tid[i])
            y_ = y[start:end]
            y_aux.append(y_)
            start = end
        y = y_aux

    X = []
    for i, ip in enumerate(dic_tid):
        X_aux = []
        for c in features:
            pad_col = pad_sequences(traj.loc[dic_tid[i], c], 
                                    maxlen=max_lenght, 
                                    padding='pre',
                                    value=0.0)
        
            X_aux.append(pad_col) 
        X.append(np.concatenate(X_aux, axis=1))
    
    dic_parameters['features'] = features
    
    # TODO
#    print('Trajectories:  ' + str(trajs))
#    print('Labels:        ' + str(len(y[0])))
#    print('Train size:    ' + str(len(x_train[0]) / trajs))
#    print('Test size:     ' + str(len(x_test[0]) / trajs))
#    print('x_train shape: ' + str(x_train.shape))
#    print('y_train shape: ' + str(y_train.shape))
#    print('x_test shape:  ' + str(x_test.shape))
#    print('y_test shape:  ' + str(y_test.shape))
    
    return X, y, dic_parameters
    
    
def label_encoding(df_, col=[]): 
    if len(col) == 0:
#        print('... if col is empty, than col equal to df_columns')
        col = df_.columns
    
    assert set(col).issubset(set(df_.columns)), "ERR: some columns does not exist in df."
    label_encode = {}
    
    for colname in col:
        if not isinstance(df_[colname].iloc[0], np.ndarray):
            print('   Encoding: {}'.format(colname))
            le = LabelEncoder()
            df_[colname] = le.fit_transform(df_[colname])
            label_encode[colname] = le
    return label_encode