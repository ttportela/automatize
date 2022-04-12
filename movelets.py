# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
from .main import importer #, display
from .visualization import *
importer(['S', 'glob', 'np'], globals())

from pandas import json_normalize
# -----------------------------------------------------------------------
def movelets_class_dataframe(file_name, name='movelets', count=0):
    importer(['json'], globals())
    
    df = pd.DataFrame()
#     print(file_name)
    with open(file_name) as f:
        data = json.load(f)
        if name not in data.keys():
            name='shapelets'
        l = len(data[name])
        for x in range(0, l):
            aux_df = []

            points = data[name][x]['points_with_only_the_used_features']
            aux_df = json_normalize(points)

            aux_df['tid'] = data[name][x]['trajectory']
            aux_df['label'] = data[name][x]['label']
            aux_df['size'] = int(data[name][x]['quality']['size'])
            aux_df['quality'] = int(data[name][x]['quality']['quality'] * 100)
            aux_df['movelet_id'] = count
            df = df.append(aux_df)
            count += 1
        
    return redefine_dataframe(df)

def movelets_dataframe(path_name, name='movelets'):
    count = 0
    path_to_file = glob.glob(os.path.join(path_name, '**', 'moveletsOnTrain.json'), recursive=True)
    df = pd.DataFrame()
    for file_name in path_to_file:
        aux_df = movelets_class_dataframe(file_name, name, count)
        count += len(aux_df['movelet_id'].unique())
        df = pd.concat([df, aux_df])
#     print(df)
    cols = ['movelet_id', 'tid', 'label', 'size', 'quality']
    cols = cols + [x for x in df.columns if x not in cols]
    return redefine_dataframe(df[cols])

def movelets2csv(path_name, res_path, name='movelets'):
    count = 0
    method=os.path.split(os.path.abspath(path_name))[1]
    path_to_file = glob.glob(os.path.join(path_name, '**', 'moveletsOnTrain.json'), recursive=True)
    for file_name in path_to_file:
        aux_df = movelets_class_dataframe(file_name, name, count)
#         return aux_df
        count += len(aux_df['movelet_id'].unique())
        label = os.path.basename(os.path.dirname(os.path.abspath(file_name)))
        aux_df = redefine_dataframe(aux_df)
        save_to = os.path.join(res_path,label,method+'-movelets_stats.csv')
        if not os.path.exists(os.path.dirname(save_to)):
            os.makedirs(os.path.dirname(save_to))
        aux_df.to_csv(save_to, index=False)

# -----------------------------------------------------------------------
def redefine_dataframe(df):
#     names = df.columns.tolist()
#     new = []
#     names.remove('tid')
#     names.remove('label')
#     names.remove('size')
#     names.remove('quality')
#     names.remove('movelet_id')
#     print(names)
#     for x in names:
#         new.append(x.split('.')[0])
#     new.append('tid')
#     new.append('label')
#     new.append('size')
#     new.append('quality')
#     new.append('movelet_id')
#     df.columns = new
    
    df = df.fillna('-')
    return df

# -----------------------------------------------------------------------
def read_movelets_statistics(file_name, name='movelets', count=0):
    importer(['json'], globals())
    
    df_stats = pd.DataFrame()
    used_features = []
    with open(file_name) as f:
        data = json.load(f)    
        if name not in data.keys():
            name='shapelets'
        l = len(data[name])
        for x in range(0, l):
            points = data[name][x]['points_with_only_the_used_features']

            df_stats = df_stats.append({
                'movelet_id': count,
                'tid': data[name][x]['trajectory'],
                'label': data[name][x]['label'],
                'size': int(data[name][x]['quality']['size']),
                'quality': int(data[name][x]['quality']['quality'] * 100),
                'n_features': len(points[0].keys()),
                'features': str(list(points[0].keys())),
            }, ignore_index=True)

            used_features = used_features + list(points[0].keys())

            count += 1

        used_features = {x:used_features.count(x) for x in set(used_features)}
    return used_features, df_stats[['movelet_id', 'tid', 'label', 'size', 'quality', 'n_features', 'features']]

def read_movelets_statistics_bylabel(path_name, name='movelets'):
    count = 0
    path_to_file = glob.glob(os.path.join(path_name, '**', 'moveletsOnTrain.json'), recursive=True)
    df = pd.DataFrame()
    for file_name in path_to_file:
        used_features, aux_df = read_movelets_statistics(file_name, name, count)
        
        stats = aux_df.describe()
        count += len(aux_df['movelet_id'].unique())
        
        label = aux_df['label'].unique()[0]
        stats = {
            'label': label,
            'movelets': len(aux_df['movelet_id'].unique()),
            'mean_size': stats['size']['mean'],
            'min_size': stats['size']['min'],
            'max_size': stats['size']['max'],
            'mean_quality': stats['quality']['mean'],
            'min_quality': stats['quality']['min'],
            'max_quality': stats['quality']['max'],
            'mean_n_features': stats['n_features']['mean'],
            'min_n_features': stats['n_features']['min'],
            'max_n_features': stats['n_features']['max'],
#             'used_features': used_features,
#             'features': str(list(points[0].keys())),
        }
        
        stats.update(used_features)
        
        df = df.append(stats , ignore_index=True)
        
#     print(df)
    cols = ['label', 'movelets', 'mean_quality', 'min_quality', 'max_quality', 
            'mean_size', 'min_size', 'max_size',
            'mean_n_features', 'min_n_features', 'max_n_features']
    cols = cols + [x for x in df.columns if x not in cols]
    return redefine_dataframe(df[cols])

def movelets_statistics(movelets):
    importer(['json'], globals())
    
    df_stats = pd.DataFrame()
#     used_features = {}
    l = len(movelets)
    def processMov(m):
#     for m in movelets:
        points = m.data

        stats = {
            'movelet_id': m.mid,
            'tid': m.tid,
            'label': m.label,
            'size': m.size,
            'quality': m.quality,
            'n_features': len(points[0].keys()),
#             'features': ', '.join(list(points[0].keys())),
        }
        
        stats.update({k: 1 for k in list(points[0].keys())})
    
        return stats#df_stats.append(stats, ignore_index=True)

    df_stats = pd.DataFrame.from_records( list(map(lambda m: processMov(m), movelets)) )
#         if m.label not in used_features.keys():
#             used_features[m.label] = []
#         used_features[m.label] = used_features[m.label] + list(points[0].keys())

#     used_features = {l: {x: used_features[l].count(x) for x in used_features[l]} for l in used_features.keys()}
#     return used_features, df_stats[['movelet_id', 'tid', 'label', 'size', 'quality', 'n_features', 'features']]
    cols = ['movelet_id', 'tid', 'label', 'size', 'quality', 'n_features']#, 'features']
    cols = cols + [x for x in df_stats.columns if x not in cols]
    return df_stats[cols]

def movelets_statistics_bylabel(df, label='label'):
    df_stats = pd.DataFrame()
    
    def countFeatures(used_features, f):
        for s in f.split(', '):
            used_features[s] = used_features[s]+1 if s in used_features.keys() else 1

    cols = ['movelet_id', 'tid', 'label', 'size', 'quality', 'n_features']
    feat_cols = [x for x in df.columns if x not in cols]
            
    def processLabel(lbl):
#     for lbl in df[label].unique():
        aux_df = df[df['label'] == lbl]
        stats = aux_df.describe()
#         count += len(aux_df['movelet_id'].unique())

        stats = {
            'label': lbl,
            'movelets': len(aux_df['movelet_id'].unique()),
            'mean_size': stats['size']['mean'],
            'min_size': stats['size']['min'],
            'max_size': stats['size']['max'],
            'mean_quality': stats['quality']['mean'],
            'min_quality': stats['quality']['min'],
            'max_quality': stats['quality']['max'],
            'mean_n_features': stats['n_features']['mean'],
            'min_n_features': stats['n_features']['min'],
            'max_n_features': stats['n_features']['max'],
#             'used_features': used_features,
#             'features': str(list(points[0].keys())),
        }
        
        stats.update({k: aux_df[k].sum() for k in feat_cols})

#         used_features = dict()
#         list(map(lambda f: countFeatures(used_features, f), aux_df['features']))
#         stats.update(used_features[label])
#         print(used_features)
        return stats
#         df_stats = df_stats.append(stats, ignore_index=True)
        
    df_stats = pd.DataFrame.from_records( list(map(lambda lbl: processLabel(lbl), df[label].unique())) )
#     print(df_stats)
    cols = ['label', 'movelets', 'mean_quality', 'min_quality', 'max_quality', 
            'mean_size', 'min_size', 'max_size',
            'mean_n_features', 'min_n_features', 'max_n_features']
    cols = cols + [x for x in df_stats.columns if x not in cols]
    return redefine_dataframe(df_stats[cols])

def trajectory_statistics(ls_trajs):
    samples = len(ls_trajs)
    labels = set()
    top = 0
    bot = float('inf')
    npoints = 0
    classes = {}
    df = pd.DataFrame()
    for T in ls_trajs:
        labels.add(T.label)
        classes[T.label] = 1 if T.label not in classes.keys() else classes[T.label]+1
        npoints += T.size
        top = max(top, T.size)
        bot = min(bot, T.size)
        for p in T.points:
            df = df.append(p, ignore_index=True)
    
    labels = [str(l) for l in labels]
    labels.sort()
    avg_size = npoints / samples
    diff_size = max( avg_size - bot , top - avg_size)
    attr = list(ls_trajs[0].points[0].keys())
    num_attr = len(attr)
#     stats=pd.DataFrame()
#     df = df.select_dtypes(include=np.number)
#     stats["Mean"]=df.mean()
#     stats["Std.Dev"]=df.std()
#     stats["Var"]=df.var()
    stats=pd.DataFrame()
    dfx = df.apply(pd.to_numeric, args=['coerce'])
    stats["Mean"]=dfx.mean(axis=0, skipna=True)
    stats["Std.Dev"]=dfx.std(axis=0, skipna=True)
    stats["Variance"]=dfx.var(axis=0, skipna=True)
    
    stats = stats.sort_values('Variance', ascending=False)
    
    return labels, samples, top, bot, npoints, avg_size, diff_size, attr, num_attr, classes, stats

# -----------------------------------------------------------------------
# def initMovelet(points):
#     m = Movelet(points)
#     return m
