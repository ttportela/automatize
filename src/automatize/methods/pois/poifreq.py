# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
@author: Francisco Vicenzi (adapted)
'''
import sys, os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from main import importer

## POI-F: POI Frequency
def poi(df_train, df_test, possible_sequences, seq2idx, sequence, dataset, feature, result_dir=None, tid_col='tid', class_col='label'):
#     from ..main import importer
#     importer(['S'], locals())
    
    print('Starting POI...')
    method = 'poi'
    
    # Train
    train_tids = df_train[tid_col].unique()
    x_train = np.zeros((len(train_tids), len(possible_sequences)))
    y_train = df_train.drop_duplicates(subset=[tid_col, class_col],
                                       inplace=False) \
                      .sort_values(tid_col, ascending=True,
                                   inplace=False)[class_col].values

    for i, tid in enumerate(train_tids):
        traj_pois = df_train[df_train[tid_col] == tid][feature].values
        for idx in range(0, (len(traj_pois)-(sequence - 1))):
            aux = []
            for b in range (0, sequence):
                aux.append(traj_pois[idx + b])
            aux = tuple(aux)
            x_train[i][seq2idx[aux]] += 1

    # Test
    test_tids = df_test[tid_col].unique()
    test_unique_features = df_test[feature].unique().tolist()
    x_test = np.zeros((len(test_tids), len(possible_sequences)))
    y_test = df_test.drop_duplicates(subset=[tid_col, class_col],
                                       inplace=False) \
                      .sort_values(tid_col, ascending=True,
                                   inplace=False)[class_col].values

    for i, tid in enumerate(test_tids):
        traj_pois = df_test[df_test[tid_col] == tid][feature].values
        for idx in range(0, (len(traj_pois)-(sequence - 1))):
            aux = []
            for b in range (0, sequence):
                aux.append(traj_pois[idx + b])
            aux = tuple(aux)
            if aux in possible_sequences:
                x_test[i][seq2idx[aux]] += 1
    
    if result_dir is not False:
        core_name = os.path.join(result_dir, method+'_'+feature+'_'+str(sequence)+'_'+dataset)
        to_file(core_name, x_train, x_test, y_train, y_test)
        
    return x_train, x_test, y_train, y_test
    
### NPOI-F: Normalized POI Frequency
def npoi(df_train, df_test, possible_sequences, seq2idx, sequence, dataset, feature, result_dir=None, tid_col='tid', class_col='label'):
#     from ..main import importer
#     importer(['S'], locals())
    
    print('Starting NPOI...')
    method = 'npoi'
    
    # Train
    train_tids = df_train[tid_col].unique()
    x_train = np.zeros((len(train_tids), len(possible_sequences)))
    y_train = df_train.drop_duplicates(subset=[tid_col, class_col],
                                       inplace=False) \
                      .sort_values(tid_col, ascending=True,
                                   inplace=False)[class_col].values

    for i, tid in enumerate(train_tids):
        traj_pois = df_train[df_train[tid_col] == tid][feature].values
        for idx in range(0, (len(traj_pois)-(sequence - 1))):
            aux = []
            for b in range (0, sequence):
                aux.append(traj_pois[idx + b])
            aux = tuple(aux)
            x_train[i][seq2idx[aux]] += 1
        x_train[i] = x_train[i]/len(traj_pois)

    # Test
    test_tids = df_test[tid_col].unique()
    test_unique_features = df_test[feature].unique().tolist()
    x_test = np.zeros((len(test_tids), len(possible_sequences)))
    y_test = df_test.drop_duplicates(subset=[tid_col, class_col],
                                       inplace=False) \
                      .sort_values(tid_col, ascending=True,
                                   inplace=False)[class_col].values

    for i, tid in enumerate(test_tids):
        traj_pois = df_test[df_test[tid_col] == tid][feature].values
        for idx in range(0, (len(traj_pois)-(sequence - 1))):
            aux = []
            for b in range (0, sequence):
                aux.append(traj_pois[idx + b])
            aux = tuple(aux)
            if aux in possible_sequences:
                x_test[i][seq2idx[aux]] += 1
        x_test[i] = x_test[i]/len(traj_pois)
        
    if result_dir is not False:
        core_name = os.path.join(result_dir, method+'_'+feature+'_'+str(sequence)+'_'+dataset)
        to_file(core_name, x_train, x_test, y_train, y_test)
        
    return x_train, x_test, y_train, y_test
    
### WNPOI-F: Weighted Normalized POI Frequency.
def wnpoi(df_train, df_test, possible_sequences, seq2idx, sequence, dataset, feature, result_dir=None, tid_col='tid', class_col='label'):
#     from ..main import importer
#     importer(['S'], locals())
    
    print('Starting WNPOI...')    
    method = 'wnpoi'
    
    train_labels = df_train[class_col].unique()
    weights = np.zeros(len(possible_sequences))
    for label in train_labels:
        aux_w = np.zeros(len(possible_sequences))
        class_pois = df_train[df_train[class_col] == label][feature].values
        for idx in range(0, (len(class_pois)-(sequence - 1))):
            aux = []
            for b in range (0, sequence):
                aux.append(class_pois[idx + b])
            aux = tuple(aux)
            seqidx = seq2idx[aux]
            if aux_w[seqidx] == 0:
                weights[seqidx] += 1
                aux_w[seqidx] = 1
    weights = np.log2(len(train_labels)/weights)
    # Train
    train_tids = df_train[tid_col].unique()
    x_train = np.zeros((len(train_tids), len(possible_sequences)))
    y_train = df_train.drop_duplicates(subset=[tid_col, class_col],
                                       inplace=False) \
                      .sort_values(tid_col, ascending=True,
                                   inplace=False)[class_col].values

    for i, tid in enumerate(train_tids):
        traj_pois = df_train[df_train[tid_col] == tid][feature].values
        for idx in range(0, (len(traj_pois)-(sequence - 1))):
            aux = []
            for b in range (0, sequence):
                aux.append(traj_pois[idx + b])
            aux = tuple(aux)
            x_train[i][seq2idx[aux]] += 1
        x_train[i] = x_train[i]/len(traj_pois)
        for w in range(0, len(possible_sequences)):
            x_train[i][w] *= weights[w]

    # Test
    test_tids = df_test[tid_col].unique()
    test_unique_features = df_test[feature].unique().tolist()
    x_test = np.zeros((len(test_tids), len(possible_sequences)))
    y_test = df_test.drop_duplicates(subset=[tid_col, class_col],
                                       inplace=False) \
                      .sort_values(tid_col, ascending=True,
                                   inplace=False)[class_col].values

    for i, tid in enumerate(test_tids):
        traj_pois = df_test[df_test[tid_col] == tid][feature].values
        for idx in range(0, (len(traj_pois)-(sequence - 1))):
            aux = []
            for b in range (0, sequence):
                aux.append(traj_pois[idx + b])
            aux = tuple(aux)
            if aux in possible_sequences:
                x_test[i][seq2idx[aux]] += 1
        x_test[i] = x_test[i]/len(traj_pois)
        for w in range(0, len(possible_sequences)):
            x_test[i][w] *= weights[w]
            
    if result_dir is not False:
        core_name = os.path.join(result_dir, method+'_'+feature+'_'+str(sequence)+'_'+dataset)
        to_file(core_name, x_train, x_test, y_train, y_test)
        
    return x_train, x_test, y_train, y_test
    
## --------------------------------------------------------------------------------------------
def poifreq_all(sequence, dataset, feature, folder, result_dir, tid_col='tid', class_col='label'):
#     from ..main import importer
#     importer(['S'], locals())
    print('Dataset: {}, Feature: {}, Sequence: {}'.format(dataset, feature, sequence))
#     df_train = pd.read_csv(folder+dataset+'_train.csv')
#     df_test = pd.read_csv(folder+dataset+'_test.csv')
    
    df_train, df_test = loadTrainTest([feature], folder, dataset)
    
    unique_features = df_train[feature].unique().tolist()
    
    points = df_train[feature].values
    possible_sequences = []
    for idx in range(0, (len(points)-(sequence - 1))):
        aux = []
        for i in range (0, sequence):
            aux.append(points[idx + i])
        aux = tuple(aux)
        if aux not in possible_sequences:
            possible_sequences.append(aux)

    seq2idx = dict(zip(possible_sequences, np.r_[0:len(possible_sequences)]))
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    pd.DataFrame(possible_sequences).to_csv(os.path.join(result_dir, feature+'_'+str(sequence)+'_'+dataset+'-sequences.csv'), index=False, header=None)
    
    poi(df_train, df_test, possible_sequences, seq2idx, sequence, dataset, feature, result_dir, tid_col, class_col)
    npoi(df_train, df_test, possible_sequences, seq2idx, sequence, dataset, feature, result_dir, tid_col, class_col)
    wnpoi(df_train, df_test, possible_sequences, seq2idx, sequence, dataset, feature, result_dir, tid_col, class_col)
    
## By Tarlis: Run this first...
## --------------------------------------------------------------------------------------------
def poifreq(sequences, dataset, features, folder, result_dir, method='npoi', save_all=False, doclass=True, tid_col='tid', class_col='label'):
#    from ..main import importer
    importer(['S', 'datetime'], globals(), {'preprocessing': ['dfVariance']})
#     print('Dataset: {}, Feature: {}, Sequence: {}'.format(dataset, feature, sequence))
#     from datetime import datetime

    time = datetime.now()
#     if dataset is '':
#         df_train = pd.read_csv(os.path.join(folder, 'train.csv'))
#         df_test = pd.read_csv(os.path.join(folder, 'test.csv'))
#     else:
#         df_train = pd.read_csv(os.path.join(folder, dataset+'_train.csv'))
#         df_test = pd.read_csv(os.path.join(folder, dataset+'_test.csv'))
    
    df_train, df_test = loadTrainTest(features, folder, dataset)
    
    if features is None:
        df_train
        stats = dfVariance(df[[x for x in df.columns if x not in [tid_col, class_col]]])
        features = [stats.iloc[0].index[0]]
    
    if save_all:
        save_all = result_dir
        
    agg_x_train = None
    agg_x_test  = None
    
    for sequence in sequences:
        aux_x_train = None
        aux_x_test  = None
        for feature in features:
            print('Dataset: {}, Feature: {}, Sequence: {}'.format(dataset, feature, sequence))
            unique_features = df_train[feature].unique().tolist()

            points = df_train[feature].values
            possible_sequences = []
            for idx in range(0, (len(points)-(sequence - 1))):
                aux = []
                for i in range (0, sequence):
                    aux.append(points[idx + i])
                aux = tuple(aux)
                if aux not in possible_sequences:
                    possible_sequences.append(aux)

            seq2idx = dict(zip(possible_sequences, np.r_[0:len(possible_sequences)]))

            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            pd.DataFrame(possible_sequences).to_csv(os.path.join(result_dir, \
               feature+'_'+str(sequence)+'_'+dataset+'-sequences.csv'), index=False, header=None)

            if method == 'poi':
                x_train, x_test, y_train, y_test = poi(df_train, df_test, possible_sequences, \
                                                       seq2idx, sequence, dataset, feature, result_dir=save_all, 
                                                       tid_col=tid_col, class_col=class_col)
            elif method == 'npoi':
                x_train, x_test, y_train, y_test = npoi(df_train, df_test, possible_sequences, \
                                                       seq2idx, sequence, dataset, feature, result_dir=save_all, 
                                                       tid_col=tid_col, class_col=class_col)
            else:
                x_train, x_test, y_train, y_test = wnpoi(df_train, df_test, possible_sequences, \
                                                       seq2idx, sequence, dataset, feature, result_dir=save_all, 
                                                       tid_col=tid_col, class_col=class_col)

            # Concat columns:
            if aux_x_train is None:
                aux_x_train = pd.DataFrame(x_train)
            else:
                aux_x_train = pd.concat([aux_x_train, pd.DataFrame(x_train)], axis=1)   

            if aux_x_test is None:
                aux_x_test = pd.DataFrame(x_test)
            else:
                aux_x_test = pd.concat([aux_x_test, pd.DataFrame(x_test)], axis=1)    
                
        # Write features concat:
        core_name = os.path.join(result_dir, method+'_'+('_'.join(features))+'_'+('_'.join([str(sequence)])) ) #+'_'+dataset)
        to_file(core_name, aux_x_train, aux_x_test, y_train, y_test)
        
        if agg_x_train is None:
            agg_x_train = aux_x_train
        else:
            agg_x_train = pd.concat([agg_x_train, aux_x_train], axis=1)   

        if agg_x_test is None:
            agg_x_test = aux_x_test
        else:
            agg_x_test = pd.concat([agg_x_test, aux_x_test], axis=1)    
                
    
    del df_train
    del df_test 
    del x_train
    del x_test   
   
    core_name = os.path.join(result_dir, method+'_'+('_'.join(features))+'_'+('_'.join([str(n) for n in sequences])) ) #+'_'+dataset)
    to_file(core_name, agg_x_train, agg_x_test, y_train, y_test)
    time_ext = (datetime.now()-time).total_seconds() * 1000
    
    del agg_x_train
    del agg_x_test 
    del y_train
    del y_test 
    
    if doclass:
        time = datetime.now()
        
#         from ensemble_models.poifreq import model_poifreq
        importer(['TEC.POIS'], locals())
        model, x_test = model_poifreq(core_name)
        model = model.predict(x_test)
        time_cls = (datetime.now()-time).total_seconds() * 1000
        
#         f=open(os.path.join(result_dir, method+'_results.txt'), "a+")
        f=open(os.path.join(result_dir, 'results_summary.txt'), "a+")
        f.write("Processing time: %d milliseconds\r\n" % (time_ext))
        f.write("Classification time: %d milliseconds\r\n" % (time_cls))
        f.write("Total time: %d milliseconds\r\n" % (time_ext+time_cls))
        f.close()
        
#     else:
        
# #         f=open(os.path.join(result_dir, method+'_results.txt'), "a+")
#         f=open(os.path.join(result_dir, 'results_summary.txt'), "a+")
#         f.write("Processing time: %d milliseconds\r\n" % (time_ext))
#         f.close()
        
    return core_name
   
    
## --------------------------------------------------------------------------------------------
def to_file(core_name, x_train, x_test, y_train, y_test):
#     from ..main import importer
#     importer(['pd'], locals())
    df_x_train = pd.DataFrame(x_train).to_csv(core_name+'-x_train.csv', index=False, header=None)
    df_x_test = pd.DataFrame(x_test).to_csv(core_name+'-x_test.csv', index=False, header=None)
    df_y_train = pd.DataFrame(y_train, columns=['label']).to_csv(core_name+'-y_train.csv', index=False)
    df_y_test = pd.DataFrame(y_test, columns=['label']).to_csv(core_name+'-y_test.csv', index=False)
    
def geoHasTransform(df, geo_precision=8):
#     from ..main import importer
    importer(['geohash'], globals()) #globals
#     from ensemble_models.utils import geohash
    return [geohash(df['lat'].values[i], df['lon'].values[i], geo_precision) for i in range(0, len(df))]

def loadTrainTest(features, folder, dataset=''):
#     from ..main import importer
    importer(['readDataset'], globals())
#     if dataset == '':
#         df_train = pd.read_csv(os.path.join(folder, 'train.csv'))
#         df_test = pd.read_csv(os.path.join(folder, 'test.csv'))
#     else:
#         df_train = pd.read_csv(os.path.join(folder, dataset+'_train.csv'))
#         df_test = pd.read_csv(os.path.join(folder, dataset+'_test.csv'))
    na_values = -999
    if dataset == '':
        df_train = readDataset(folder, file='train.csv', missing=na_values)
        df_test = readDataset(folder, file='test.csv', missing=na_values)
    else:
        df_train = readDataset(folder, file=dataset+'_train.csv', missing=na_values)
        df_test = readDataset(folder, file=dataset+'_test.csv', missing=na_values)
    
    if 'lat_lon' in features and ('lat' in df_train.columns and 'lon' in df_test.columns):
        df_train['lat_lon'] = geoHasTransform(df_train)
        df_test['lat_lon']  = geoHasTransform(df_test)
        
    return df_train, df_test