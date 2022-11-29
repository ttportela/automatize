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
                     space_geohash=False, # True: Geohash, False: indexgrid
                     geo_precision=8,     # Geohash: precision OR IndexGrid: meters
                     features=None,
                     features_encoding=True, 
                     y_one_hot_encodding=False,
                     split_test_validation=True,
                     data_preparation=1):

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
    
    df_train, _, columns_order = organizeFrame(df_train, tid_col=tid_col, class_col=class_col)
    df_test, _, _ = organizeFrame(df_test, tid_col=tid_col, class_col=class_col)

    # TODO:
    if 'lat' in df_train.columns and 'lon' in df_train.columns:
        if space_geohash:
    #        keys.remove('lat')
    #        keys.remove('lon')
            space = True
            count_attr += geo_precision * 5
            print("Attribute Space: " +
                       str(geo_precision * 5) + "-bits value")
        else: # Index Grid
            from automatize.methods._lib.pymove.processing.gridutils import create_virtual_grid, create_update_index_grid_feature
            df_ = pd.concat([df_train, df_test])
            dic_grid = create_virtual_grid(geo_precision, [df_['lat'].min(), df_['lon'].min(), df_['lat'].max(), df_['lon'].max()])
            create_update_index_grid_feature(df_train, dic_grid, sort=False)
            create_update_index_grid_feature(df_test, dic_grid, sort=False)
            columns_order = list(filter(lambda x: x not in ['space','lat','lon'], columns_order)) + ['index_grid']
            #df_train.drop(['space','lat','lon'], axis=1, errors='ignore', inplace=True)
            #df_test.drop(['space','lat','lon'], axis=1, errors='ignore', inplace=True)
    
    if features:
        columns_order = list(filter(lambda x: x in features + [tid_col,class_col], columns_order))
        
    if split_test_validation:
        df_train, df_val = trainAndTestSplit(df_train, train_size=0.75, tid_col=tid_col, class_col=class_col, outformats=[])
        
        df_train = df_train[columns_order]
        df_val  = df_val[columns_order]
        df_test  = df_test[columns_order]
        #data = [df_train, df_val, df_test]
        data = [df_train, df_val, df_test]
    else:
        df_train = df_train[columns_order]
        df_test  = df_test[columns_order]
        data = [df_train, df_test]

    #df_ = pd.concat(data)
    
    features = list(df_train.keys())
    num_classes = len(set(df_train[class_col])) 
    count_attr = 0
    space = False

    for attr in features:
        if attr != class_col:
            values = len(set(df_train[attr]))
            count_attr += values
            print("Attribute '" + attr + "': " + str(values) + " unique values")

    print("Total of attribute/value pairs: " + str(count_attr))

    
    if data_preparation == 1:
        X, y, dic_parameters = generate_X_y_ml(data, 
                                            features_encoding=features_encoding,       
                                            y_one_hot_encodding=y_one_hot_encodding)
    else:
        X, y, dic_parameters = generate_X_y_rnn(data, 
                                            features_encoding=features_encoding,       
                                            y_one_hot_encodding=y_one_hot_encodding)
    
    dic_parameters['num_classes'] = num_classes
    
    return X, y, features, num_classes, space, dic_parameters
    

def generate_X_y_ml( data,
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
        print('TIDs_{}: {}'.format(i, len(dic_tid[i])))#, dic_tid[i])
    
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
#    elif input_total > 1:
#        start = 0
#        end   = 0
#
#        y_aux = []
#        for i in range(0, input_total):
#            end = end + len(dic_tid[i])
#            print('TID', i, start, end)
#            display(y)
#            y_ = y[start:end]
#            y_aux.append(y_)
#            start = end
#        y = y_aux
    elif input_total == 2:
        y_train = y[:len(dic_tid[0])]
        y_test = y[len(dic_tid[0]):]
        y = []
        y.append(y_train)
        y.append(y_test)
        
    elif input_total == 3:
        y_train = y[:len(dic_tid[0])]
        y_val = y[len(dic_tid[0]):len(dic_tid[0])+len(dic_tid[1])]
        y_test = y[len(dic_tid[0])+len(dic_tid[1]):]
        y = []
        y.append(y_train)
        y.append(y_val)
        y.append(y_test)

        
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
    dic_parameters['max_lenght'] = max_lenght
    
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
    
def generate_X_y_rnn(data=[],
                        features_encoding=True,
                        y_one_hot_encodding=True,
                        lat_col='lat', 
                        lon_col='lon',
                        tid_col='tid', 
                        class_col='label'):
                        #label_lat='lat', 
                        #label_lon='lon', 
                        #label_y='label',
                        #label_segment='tid'):

    print('\n\n###########      DATA ENCODING        ###########\n')
    input_total = len(data)
    assert (input_total > 0) & (input_total <=3), "ERRO: data is not set or dimenssion > 3"

    print('Input total: {}'.format(input_total))

    if input_total > 1:
        #print('... concat dataframe')
        df_ = pd.concat(data)
    else:
        #print('... df_ is data')
        df_ = data[0]

    assert isinstance(df_, pd.DataFrame), "ERR: inform data as a list of pandas.Dataframe()"
    assert class_col in df_, "ERR: Label y in not on dataframe"
    assert tid_col in df_, "ERR: TID in not on dataframe"


    features = list(df_.columns)
    num_classes = len(set(df_[class_col]))#)df_[class_col].nunique()
    max_lenght = df_.groupby(tid_col).agg({class_col:'count'}).max()[0]

    dic_parameters = {}
    dic_parameters['max_lenght'] = max_lenght
    dic_parameters['num_classes'] = num_classes


    dic_tid = {}
    for i, d in enumerate(data):
        dic_tid[i] = d[tid_col].unique()
        print('... tid_{}: {}'.format(i, len(dic_tid[i])))

    dic_parameters['dic_tid'] = dic_tid


    print('col_name: {}...\n... num_classes: {}\n... max_lenght: {}'.format(features, num_classes, max_lenght))

    col_drop = [tid_col, lat_col, lon_col, class_col] 

    for c in col_drop:
        if c in features:
            print('Removing column {} of attr'.format(c))
            features.remove(c)

    if features_encoding == True:
        print('\n\n#####   Encoding string data to integer   ######')
        if len(features) > 0:
            dic_parameters['encode_features'] = label_encoding_df_to_rnn(df_, col=features)
        else:
            print('Encoding was not necessary')

    col_groupby = {}
    for c in features:
        col_groupby[c] = list
    col_groupby[class_col] = 'first'

    dic_parameters['col_groupby'] = col_groupby

    traj = df_.groupby(tid_col, sort=False).agg(col_groupby)

    print('\n\n###########      Generating y_train and y_test     ###########')       
    if y_one_hot_encodding == True:
        print('OneHot encoding on label y')
        ohe_y = OneHotEncoder()
        y = ohe_y.fit_transform(pd.DataFrame(traj[class_col])).toarray()
        dic_parameters['encode_y'] = ohe_y 
    else:
        print('Label encoding on label y')
        le_y = LabelEncoder()
        y = np.array(le_y.fit_transform(pd.DataFrame(traj[class_col])))
        dic_parameters['encode_y'] = le_y

    print('Input total: {}'.format(input_total))
    if input_total == 1:
        y = np.array(y, ndmin=2)
    elif input_total == 2:
        y_train = y[:len(dic_tid[0])]
        y_test = y[len(dic_tid[0]):]
        y = []
        y.append(y_train)
        y.append(y_test)

    elif input_total == 3:
        y_train = y[:len(dic_tid[0])]
        y_val = y[len(dic_tid[0]):len(dic_tid[0])+len(dic_tid[1])]
        y_test = y[len(dic_tid[0])+len(dic_tid[1]):]
        y = []
        y.append(y_train)
        y.append(y_val)
        y.append(y_test)

    X = []

    dic_parameters['features'] = features    
    vocab_size = {}
    for i, ip in enumerate(dic_tid):
        X_aux = []
        for c in features:
            if c == 'geohash':
                vocab_size[c] = traj[c].iloc[0][0].shape[0] #len(traj.iloc[0][c][0]) #geohash_precision * 5
                pad_col = pad_sequences(traj.loc[dic_tid[i], c], 
                    maxlen=max_lenght, 
                    padding='pre',
                    value=0.0)

            else:
                vocab_size[c] = df_[c].max() + 1 # label_encode + 1 due to padding sequence
                pad_col = pad_sequences(traj.loc[dic_tid[i], c], 
                                    maxlen=max_lenght, 
                                    padding='pre',
                                    value=0.0)

            X_aux.append(pad_col)  
        X.append(X_aux)

    dic_parameters['vocab_size'] = vocab_size
    
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

def label_encoding_df_to_rnn(df_, col=[]): 
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

def dencoding_df_to_rnn(df_, label_encode):
    for le in list(label_encode.keys()):
        print('Decoding le: {}'.format(le))
        df_[le] = label_encode[le].inverse_transform(df_[le])