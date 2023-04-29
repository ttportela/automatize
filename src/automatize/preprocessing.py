# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
from .main import importer #, display
importer(['S', 'glob'], globals())

from .inc.io.converter import *
from .inc.script_def import getDescName
#-------------------------------------------------------------------------->>

def readDataset(data_path, folder=None, file='train.csv', class_col = 'label', missing='?'):
#     from ..main import importer
    
    if folder:
        data_path = os.path.join(data_path, folder)
    
    if '.csv' in data_path or '.zip' in data_path or '.ts' in data_path:
        url = data_path
    elif '.csv' in file or '.zip' in file or '.ts' in file:
        url = os.path.join(data_path, file)
    else:
        url = os.path.join(data_path, file+'.csv')
    
    url = os.path.abspath(url)
    if '.csv' in url and os.path.exists(url):
        df = pd.read_csv(url, na_values=missing)
    elif ('.zip' in url and os.path.exists(url)) or ('.csv' in url and os.path.exists(url.replace('.csv', '.zip'))):
        file = file.replace('.csv', '.zip')
        df = zip2df(data_path, file, class_col=class_col)
    elif '.ts' in url or os.path.exists(url.replace('train', 'TRAIN').replace('test', 'TEST').replace('.csv', '.ts')):
        importer(['ts_io'], globals())
        url = url.replace('train', 'TRAIN').replace('test', 'TEST').replace('.csv', '.ts')
        df = load_from_tsfile_to_dataframe(url, replace_missing_vals_with=missing)
#    elif '.mat' in url or os.path.exists(url.replace('.csv', '.mat')): #TODO
#        url = url.replace('.csv', '.mat')
#        df = mat2df(url, replace_missing_vals_with=missing)
    else:
        df = pd.read_csv(url, na_values=missing) # should not be used
    return df

def organizeFrame(df, columns_order=None, tid_col='tid', class_col='label'):
    if (set(df.columns) & set(['lat', 'lon'])) and not 'space' in df.columns:
        df[["lat", "lon"]] = df[["lat", "lon"]].astype(str) 
        df['space'] = df["lat"] + ' ' + df["lon"]
        df[["lat", "lon"]] = df[["lat", "lon"]].astype(float)

        if columns_order is not None:
            columns_order.insert(columns_order.index('lon')-1, 'space')
            
    elif ('space' in df.columns or 'lat_lon' in df.columns) and not (set(df.columns) & set(['lat', 'lon'])):
        if 'lat_lon' in df.columns:
            df.rename(columns={'lat_lon': 'space'}, inplace=True)
            if columns_order is not None:
                columns_order[columns_order.index('lat_lon')] = 'space'
        
        ll = df['space'].str.split(" ", n = 1, expand = True) 
        df["lat"]= ll[0].astype(float)
        df["lon"]= ll[1].astype(float)
        
        if columns_order is not None:
            columns_order.insert(columns_order.index('space'), 'lat')
            columns_order.insert(columns_order.index('space')+1, 'lon')
    
    # For Columns ordering:
    if columns_order is None:
        columns_order = df.columns
            
    columns_order = [x for x in columns_order if x not in [tid_col, class_col]]
    columns_order = columns_order + [tid_col, class_col]
            
    columns_order_zip = [x for x in columns_order if x not in ['lat', 'lon']]
    columns_order_csv = [x for x in columns_order if x not in ['space']]
    
    return df, columns_order_zip, columns_order_csv

#-------------------------------------------------------------------------->>
def readDsDesc(data_path, folder=None, file='train.csv', tid_col='tid', class_col='label', missing='?'):
    df = readDataset(data_path, folder, file, class_col, missing)
    
    columns_order = [x for x in df.columns if x not in [tid_col, class_col]]
    df = df[columns_order + [tid_col, class_col]]
    
    if folder == None:
        folder = os.path.basename(data_path)
        data_path = os.path.dirname(data_path)
    
    desc = glob.glob(os.path.join(data_path, 'descriptors', \
                                  getDescName(os.path.basename(data_path), folder)+'_specific_hp.json'))
    if len(desc) > 0:
        attrs = readAttributes(desc[0])
        df.set_axis(attrs + [tid_col, class_col], axis=1, inplace=True)
    
    return df

def readAttributes(desc_file):
    importer(['json'], globals())
    f = open(desc_file)
    desc = json.load(f)
    f.close()
    
    attrs = []
    for at in desc['attributes']:
        if at['type'].startswith('composite'):
            for i in ['x', 'y', 'z'] if at['type'].startswith('composite3_') else ['x', 'y']:
                attrs.append(at['text'].split('_')[0] + '_' + i)
        else:
            attrs.append(at['text'])
#         attrs.append(at['text'])
        
    return attrs

# --------------------------------------------------------------------------------
def featuresJSON(df, version=1, deftype='nominal', defcomparator='equals', tid_col='tid', label_col='label', file=False):
    
    if isinstance(df, list):
        cols = {x: deftype for x in df}
    elif isinstance(df, dict):
        cols = df
    else:
        cols = descTypes(df)
    
    if tid_col not in cols.keys() or label_col not in cols.keys():
        aux = {tid_col: 'numeric', label_col: 'nominal'}
        cols = {**aux, **cols}
            
    if version == 1:
        s = '{\n   "readsDesc": [\n'

        order = 1
        for f, deftype in cols.items():
            s += ('    {\n          "order": '+str(order)+',\n          "type": "'+deftype+'",\n          "text": "'+f+'"\n    }')
            if len(cols) == order:
                s += ('\n')
            else:
                s += (',\n')
            order += 1

        s += ('    ],\n    "pointFeaturesDesc": [],\n    "subtrajectoryFeaturesDesc": [],\n')
        s += ('    "trajectoryFeaturesDesc": [],\n    "pointComparisonDesc": {\n      "pointDistance": "euclidean",\n')
        s += ('      "featureComparisonDesc": [\n')

        order = 1
        for f, deftype in cols.items():
            if f != tid_col and f != label_col:
                s += ('            {\n              "distance": "'+defcomparator+'",\n              "maxValue": -1,\n              "text": "'+f+'"\n            }')
                if len(cols)-1 == order:
                    s += ('\n')
                else:
                    s += (',\n')
            order += 1

        s += ('        ]\n    },\n    "subtrajectoryComparisonDesc": {\n        "subtrajectoryDistance": "euclidean",\n')
        s += ('        "featureComparisonDesc": [\n            {\n              "distance": "euclidean",\n              "text": "points"\n')
        s += ('            }\n        ]\n    }\n}')
    else: # VERSION 2 (*_hp.json)      
        s  = '{\n   "input": {\n          "train": ["train"],\n          "test": ["test"],\n          "format": "CSV",\n'
        s += '          "loader": "interning"\n   },\n'
        s += '   "idFeature": {\n          "order": '+str(list(cols.keys()).index(tid_col)+1)+',\n          "type": "numeric",\n          "text": "'+tid_col+'"\n    },\n'
        s += '   "labelFeature": {\n          "order": '+str(list(cols.keys()).index(label_col)+1)+',\n          "type": "nominal",\n          "text": "label"\n    },\n'
        s += '   "attributes": [\n'
        
        order = 1
        for f, deftype in cols.items():
            if f != tid_col and f != label_col:
                s += '        {\n              "order": '+str(order)+',\n              "type": "'+deftype+'",\n              "text": "'+str(f)+'",\n              "comparator": {\n                "distance": "'+defcomparator+'"\n              }\n        }'
                if len(cols)-1 == order:
                    s += ('\n')
                else:
                    s += (',\n')
            order += 1
        s += '    ]\n}'
        
    if file:
        file = open(file, 'w')
        print(s, file=file)
        file.close()
    else:
        print(s)
    
#-------------------------------------------------------------------------->>
def countClasses(data_path, folder, file='train.csv', tid_col = 'tid', class_col = 'label', markd=False):
    df = readDataset(data_path, folder, file, class_col, tid_col, markd)
    return countClasses_df(df, tid_col, class_col, markd)

def countClasses_df(df, tid_col = 'tid', class_col = 'label', markd=False):
    group = df.groupby([class_col, tid_col])
    df2 = group.apply(lambda x: ', '.join([str(s) for s in list(x[class_col].unique())]))
    md = "Number of Samples: " + str(len(df[tid_col].unique()))
    md += '\n\r'
    md += "Samples by Class:"
    md += '\n\r'
        
    if markd:
        md += '\n\r'
        md += df2.value_counts().to_markdown(tablefmt="github", headers=["Label", "#"])
        return md
    else:
        print(md)
        print(df2.value_counts())
        return df2.value_counts()

def dfVariance(df):
    stats=pd.DataFrame()
    dfx = df.apply(pd.to_numeric, args=['coerce'])
    #stats["Mean"]=dfx.mean(axis=0, skipna=True)
    #stats["Std.Dev"]=dfx.std(axis=0, skipna=True)
    stats["Variance"]=dfx.var(axis=0, skipna=True)

    dfx = df.fillna('?')
    for col in df.columns:
        if not np.issubdtype(dfx[col].dtype, np.number):
            categories = list(dfx[col].unique())
            dfx[col] = pd.Categorical(dfx[col], categories, ordered=True)
            #stats["Mean"][col] = categories[int( np.median(dfx[col].cat.codes) )]
            #stats["Std.Dev"][col] = np.std(dfx[col].cat.codes)
            stats["Variance"][col] = np.var(dfx[col].cat.codes)
    
    return stats.sort_values('Variance', ascending=False)

def dfStats(df):
    stats=pd.DataFrame()
    dfx = df.apply(pd.to_numeric, args=['coerce'])
    stats["Mean"]=dfx.mean(axis=0, skipna=True)
    stats["Std.Dev"]=dfx.std(axis=0, skipna=True)
    stats["Variance"]=dfx.var(axis=0, skipna=True)

    dfx = df.fillna('?')
    for col in df.columns:
        if not np.issubdtype(dfx[col].dtype, np.number):
            categories = list(dfx[col].unique())
            dfx[col] = pd.Categorical(dfx[col], categories, ordered=True)
            stats["Mean"][col] = categories[int( np.median(dfx[col].cat.codes) )]
            stats["Std.Dev"][col] = np.std(dfx[col].cat.codes)
            stats["Variance"][col] = np.var(dfx[col].cat.codes)
    
    return stats.sort_values('Variance', ascending=False)
    
def datasetStatistics(data_path, folder, file_prefix='', tid_col = 'tid', class_col = 'label', to_file=False):
#     from ..main import importer
#     importer(['S'], locals())

    def addLine(i):
        return '\n\r' + ('&nbsp;'.join(['\n\r' for x in range(i)])) + '\n\r'
    
    train = readDsDesc(data_path, folder, file_prefix+'train.csv', tid_col, class_col, missing='NaN')
    test = readDsDesc(data_path, folder, file_prefix+'test.csv', tid_col, class_col, missing='NaN')
    
    md = '##### Descriptive Statistics for ' + folder
    
    sam_train = len(train.tid.unique())
    sam_test  = len(test.tid.unique())
    points    = len(train) + len(test)
    samples = sam_train + sam_test
    top_train = train.groupby(['tid']).count().sort_values('label').tail(1)['label'].iloc[0]
    bot_train = train.groupby(['tid']).count().sort_values('label').head(1)['label'].iloc[0]
    top_test  = test.groupby(['tid']).count().sort_values('label').tail(1)['label'].iloc[0]
    bot_test  = test.groupby(['tid']).count().sort_values('label').head(1)['label'].iloc[0]
    classes = train[class_col].unique()
    avg_size = points / samples
    diff_size = max( avg_size - min(bot_train, bot_test) , max(top_train, top_test) - avg_size )
    
    stats_df = pd.DataFrame({
        'Number of Classes': [len(classes), '-', '-'],
        'Number of Attributes': [len(train.columns), '-', '-'],
        'Avg Size of Trajs': ['{:.2f}'.format(avg_size) + ' / ±' + str(diff_size), '-', '-'],
        'Number of Trajs': [str(samples), str(sam_train), str(sam_test)],
        'Hold-out': ['100%', '{:.2f}%'.format(sam_train*100/samples),
                     '{:.2f}%'.format(sam_test*100/samples)],
        'Number of Points': [str(points), str(len(train)), str(len(test))],
        'Longest Size':  [str(max(top_train, top_test)), str(top_train), str(top_test)],
        'Shortest Size': [str(max(bot_train, bot_test)), str(bot_train), str(bot_test)],
    }, index=['Total', 'Train', 'Test'])
    
    md += addLine(1)
    md += stats_df.to_markdown(tablefmt="github", colalign="right")
    md += addLine(2)
    
#     print('\n--------------------------------------------------------------------')
    md += '###### Attributes: '
    md += ', '.join([str(x) for x in train.columns])
    md += addLine(1)
    md += '###### Labels: '
#     md += addLine(1)
    md += ', '.join([str(x) for x in classes])
    md += addLine(2)
    
#     md += addLine(2)
    df = pd.concat([train, test])
    df.drop(['tid'], axis=1, inplace=True)
    stats = df.describe(include='all').fillna('')
    md += stats.to_markdown(tablefmt="github")
    md += addLine(2)
    
    md += 'Descriptive Statistics (by Variance): '
    md += addLine(1)
    #stats=pd.DataFrame()
    #dfx = df.apply(pd.to_numeric, args=['coerce'])
    #stats["Mean"]=dfx.mean(axis=0, skipna=True)
    #stats["Std.Dev"]=dfx.std(axis=0, skipna=True)
    #stats["Variance"]=dfx.var(axis=0, skipna=True)
#
    #df.fillna('?', inplace=True)
    #for col in df.columns:
    #    if not np.issubdtype(df[col].dtype, np.number):
    #        categories = list(df[col].unique())
    #        df[col] = pd.Categorical(df[col], categories, ordered=True)
    #        stats["Mean"][col] = categories[int( np.median(df[col].cat.codes) )]
    #        stats["Std.Dev"][col] = np.std(df[col].cat.codes)
    #        stats["Variance"][col] = np.var(df[col].cat.codes)
    
    md += dfStats(df).to_markdown(tablefmt="github")
    #md += stats.sort_values('Variance', ascending=False).to_markdown(tablefmt="github")
    md += addLine(2)
    
    
    if len(classes) < 15:
    #     print('\n--------------------------------------------------------------------')
        md += '###### Labels for TRAIN:'
        md += addLine(1)
        md += countClasses_df(train, markd=True)
        md += addLine(2)
    #     md += train.describe().to_markdown(tablefmt="github")
    #     md += addLine(2)

    #     print('\n--------------------------------------------------------------------')
        md += '###### Labels for TEST.:'
        md += addLine(1)
        md += countClasses_df(test, markd=True)
#         md += addLine(2)
    #     md += test.describe().to_markdown(tablefmt="github")
    #     md += addLine(2)
    
    if to_file:
        f = open(to_file, "w")
        f.write(f''+md)
        f.close()
    else:
        print('\n--------------------------------------------------------------------')
        print(md)
    return md

#-------------------------------------------------------------------------->>
def splitTIDs(df, train_size=0.7, random_num=1, tid_col='tid', class_col='label', min_elements=1):
    importer(['S', 'random'], globals())
    train = list()
    test = list()
    
    df_ = df.groupby(tid_col).first().reset_index()[[tid_col, class_col]]
    
    def splitByLabel(label):
        nonlocal df_, train, test
        tids = df_.loc[df_[class_col] == label][tid_col].unique()
        random.seed(random_num)
        
        n = int(float(len(tids))*train_size)
        n = min_elements if n < min_elements else n
            
        train_index = random.sample(list(tids), n)
        test_index  = tids[np.isin(tids, train_index, invert=True)]
        
        train = train + list(train_index)
        test  = test  + list(test_index)
        
    list(map(lambda label: splitByLabel(label), tqdm(df_[class_col].unique())))
    
    return train, test, df_

def trainAndTestSplit(df, train_size=0.7, random_num=1, tid_col='tid', class_col='label', fileprefix='', \
                      data_path='.', outformats=['zip', 'csv', 'mat'], verbose=False, organize_columns=True):
#     from ..main import importer
    importer(['S', 'random'], globals())
    if verbose:
        print(str(train_size)+"% train and test split in... " + data_path)
    
    if organize_columns:
        df, columns_order_zip, columns_order_csv = organizeFrame(df, None, tid_col, class_col)
    else:
        columns_order_zip = list(df.columns)
        columns_order_csv = list(df.columns)
    
    train_index, test_index, _ = splitTIDs(df, train_size, random_num, tid_col, class_col, min_elements=1)
    
    train = df.loc[df[tid_col].isin(train_index)]
    test  = df.loc[df[tid_col].isin(test_index)]
    
#    train = pd.DataFrame()
#    test = pd.DataFrame()
#    
#    tqdm(df[class_col].unique())
#    #for label in df[class_col].unique():
#    def splitByLabel(label):
#        nonlocal train, test
#                
#        tids = df.loc[df[class_col] == label][tid_col].unique()
#        if verbose:
#            pbar.set_description("Processing label {}, {} trajs.".format(str(label), str(len(tids))))
#        
#        random.seed(random_num)
#        train_index = random.sample(list(tids), int(len(tids)*train_size))
#        test_index  = tids[np.isin(tids, train_index, invert=True)] #np.delete(test_index, train_index)
#
#        train = pd.concat([train,df.loc[df[tid_col].isin(train_index)]])
#        test  = pd.concat([test, df.loc[df[tid_col].isin(test_index)]])
#
#        if verbose:
#            #print('Label', label, ', TIDs = ', tids)
#            pbar.set_description("Train samples: {}".format(str(train.loc[train[class_col] == label][tid_col].unique())))
#            pbar.set_description("Test samples: {}".format(str(test.loc[test[class_col] == label][tid_col].unique())))
#    
#    list(map(lambda label: splitByLabel(label), pbar))
    
    # WRITE Train / Test Files
    for outType in outformats:
        writeFiles(data_path, fileprefix, train, test, tid_col, class_col, \
                 columns_order_zip if outType in ['zip', 'mat'] else columns_order_csv, outformat=outType)
#     write_trainAndTest(data_path, train, test, tid_col, class_col, fileprefix)
    
    if verbose:
        print("Done.")
        print(" --------------------------------------------------------------------------------")
    return train, test

def splitData(df, k, random_num, tid_col='tid', class_col='label', opLabel='Spliting Data', ignore_ltk=True):
    
    if ignore_ltk: # removes labels with less than k trajectories...
        df = dropLabelsltk(df, k, tid_col, class_col)
    
    ktrain = []
    ktest = []
    for x in range(k):
        ktrain.append( pd.DataFrame() )
        ktest.append( pd.DataFrame() )

#     print("Spliting data...")
    kfold = KFold(n_splits=k, shuffle=True, random_state=random_num)
#         for label in df[class_col].unique(): 
    def addData(label):
        tids = df.loc[df[class_col] == label][tid_col].unique()
        x = 0
        for train_idx, test_idx in kfold.split(tids):
            ktrain[x] = pd.concat([ktrain[x], df.loc[df[tid_col].isin(tids[train_idx])]])
            ktest[x]  = pd.concat([ktest[x],  df.loc[df[tid_col].isin(tids[test_idx])]])
            x += 1
    list(map(lambda label: addData(label), tqdm(df[class_col].unique(), desc=opLabel)))
    
    return ktrain, ktest

def dropLabelsltk(df, k, tid_col='tid', class_col='label'):
    df_ = df.groupby(by=class_col, as_index=False).agg({tid_col: pd.Series.nunique})
    index_names = df[df[class_col].isin(df_[df_[tid_col] < k][class_col])].index
    return df.drop(index_names)

def kfold_trainAndTestSplit(data_path, k, df, random_num=1, tid_col='tid', class_col='label', fileprefix='', 
                            columns_order=None, ktrain=None, ktest=None, mat_columns=None, outformats=['zip', 'csv', 'mat'], verbose=False):
#     from ..main import importer
    importer(['S', 'KFold'], globals())
    
    print(str(k)+"-fold train and test split in... " + data_path)
    
    df, columns_order_zip, columns_order_csv = organizeFrame(df, columns_order, tid_col, class_col)
    
    if not ktrain:
        ktrain, ktest = splitData(df, k, random_num, tid_col, class_col)
    else:
        print("Train and test data provided.")
    
#     print("Writing files...")
    for x in range(k):
        path = 'run'+str(x+1)
        
        if not os.path.exists(os.path.join(data_path, path)):
            os.makedirs(os.path.join(data_path, path))
            
        train_aux = ktrain[x]
        test_aux  = ktest[x]
            
        
        print("Writing files ... " + str(x+1) +'/'+str(k))
        for outType in outformats:
            writeFiles(data_path, os.path.join(path, fileprefix), train_aux, test_aux, tid_col, class_col, \
                     columns_order_zip if outType in ['zip', 'mat'] else columns_order_csv, mat_columns, None, \
                     outType, opSuff=str(x+1))
    print("Done.")
    print(" --------------------------------------------------------------------------------")
    
    return ktrain, ktest

def stratify(df, sample_size=0.5, train_size=0.7, random_num=1, tid_col='tid', class_col='label', 
             organize_columns=True, mat_columns=None, fileprefix='', outformats=['zip', 'csv', 'mat'], data_path='.'):
    importer(['S', 'train_test_split'], globals())
    
    train_index, _, _ = splitTIDs(df, sample_size, random_num, tid_col, class_col, min_elements=2)
    
    df = df.loc[df[tid_col].isin(train_index)]
    
    train_index, test_index, _ = splitTIDs(df, train_size, random_num, tid_col, class_col, min_elements=1)
    
    if organize_columns:
        df, columns_order_zip, columns_order_csv = organizeFrame(df, None, tid_col, class_col)
    else:
        columns_order_zip = list(df.columns)
        columns_order_csv = list(df.columns)
    
    train = df.loc[df[tid_col].isin(train_index)]
    test  = df.loc[df[tid_col].isin(test_index)]
    
    path = 'S'+str(int(sample_size*100))
    if not os.path.exists(os.path.join(data_path, path)):
            os.makedirs(os.path.join(data_path, path))
    
    for outType in outformats:
        writeFiles(data_path, os.path.join(path, fileprefix), train, test, tid_col, class_col, \
                 columns_order_zip if outType in ['zip', 'mat'] else columns_order_csv, mat_columns, None, outType, opSuff=path)
    
    return train, test
    
# TODO fix stratify:
def kfold_stratify(data_path, df, k=10, inc=1, limit=10, random_num=1, tid_col='tid', class_col='label', fileprefix='', 
             ktrain=None, ktest=None, organize_columns=True, mat_columns=None, outformats=['zip', 'csv', 'mat'], ignore_ltk=True):
    importer(['S', 'KFold'], globals())
    
    print(str(k)+"-fold stratification of train and test in... " + data_path)
    
    if organize_columns:
        df, columns_order_zip, columns_order_csv = organizeFrame(df, None, tid_col, class_col)
    else:
        columns_order_zip = list(df.columns)
        columns_order_csv = list(df.columns)
        
    if not ktrain:
        ktrain, ktest = splitData(df, k, random_num, tid_col, class_col, ignore_ltk=ignore_ltk)
    else:
        print("Train and test data provided.")
    
    for x in range(0, limit, inc):
        path = 'S'+str((x+1)*int(100/k))
        
        if not os.path.exists(os.path.join(data_path, path)):
            os.makedirs(os.path.join(data_path, path))
            
        train_aux = ktrain[0]
        test_aux  = ktest[0]
        for y in range(1, x+1):
            train_aux = pd.concat([train_aux,  ktrain[y]])
            test_aux  = pd.concat([test_aux,   ktest[y]])
            
        for outType in outformats:
            writeFiles(data_path, os.path.join(path, fileprefix), train_aux, test_aux, tid_col, class_col, \
                     columns_order_zip if outType in ['zip', 'mat'] else columns_order_csv, mat_columns, None, outType, opSuff=str(x+1))
    print(" Done.")
    print(" --------------------------------------------------------------------------------")
    
    return ktrain, ktest

#def stratify(data_path, df, k=10, inc=1, limit=10, random_num=1, tid_col='tid', class_col='label', fileprefix='', 
#             ktrain=None, ktest=None, organize_columns=True, mat_columns=None, outformats=['zip', 'csv', 'mat'], ignore_ltk=True):
#    importer(['S', 'KFold'], globals())
#    
#    print(str(k)+"-fold stratification of train and test in... " + data_path)
#    
#    if organize_columns:
#        df, columns_order_zip, columns_order_csv = organizeFrame(df, None, tid_col, class_col)
#    else:
#        columns_order_zip = list(df.columns)
#        columns_order_csv = list(df.columns)
#        
#    if not ktrain:
#        ktrain, ktest = splitData(df, k, random_num, tid_col, class_col, ignore_ltk=ignore_ltk)
#    else:
#        print("Train and test data provided.")
#    
#    for x in range(0, limit, inc):
#        path = 'S'+str((x+1)*int(100/k))
#        
#        if not os.path.exists(os.path.join(data_path, path)):
#            os.makedirs(os.path.join(data_path, path))
#            
#        train_aux = ktrain[0]
#        test_aux  = ktest[0]
#        for y in range(1, x+1):
#            train_aux = pd.concat([train_aux,  ktrain[y]])
#            test_aux  = pd.concat([test_aux,   ktest[y]])
#            
#        for outType in outformats:
#            writeFiles(data_path, os.path.join(path, fileprefix), train_aux, test_aux, tid_col, class_col, \
#                     columns_order_zip if outType in ['zip', 'mat'] else columns_order_csv, mat_columns, None, outType, opSuff=str(x+1))
#    print(" Done.")
#    print(" --------------------------------------------------------------------------------")
#    
#    return ktrain, ktest
        
def writeFile(data_path, df, file, tid_col, class_col, columns_order, mat_columns=None, desc_cols=None, outformat='zip', opSuff=''):
    if outformat == 'zip':
        # WRITE ZIP >> FOR MASTERMovelets:
        df2zip(data_path, df, file, tid_col, class_col, select_cols=columns_order,\
               opLabel='Writing - ZIP |' + opSuff)
        
    elif outformat == 'csv':
        print('Writing - CSV |' + opSuff)
        df[columns_order].to_csv(os.path.join(data_path, file+".csv"), index = False)
        
    elif outformat == 'mat':
        # WRITE MAT Files >> FOR HiPerMovelets:
        df2mat(df, data_path, file, cols=columns_order, mat_cols=mat_columns, tid_col=tid_col, class_col=class_col, \
               desc_cols=desc_cols, opLabel='Writing - MAT|' + opSuff)
        
def writeFiles(data_path, file, train, test, tid_col, class_col, columns_order, mat_columns=None, desc_cols=None, outformat='zip', opSuff=''):
    # WRITE Train
    writeFile(data_path, train, file+'train', tid_col, class_col, columns_order, mat_columns, desc_cols, 
              outformat, opSuff='TRAIN - '+opSuff)
    # WRITE Test
    writeFile(data_path, test,  file+'test',  tid_col, class_col, columns_order, mat_columns, desc_cols, 
              outformat, opSuff='TEST - '+ opSuff)
    
#def writeFiles(data_path, file, train, test, tid_col, class_col, columns_order, mat_columns=None, outformat='zip', opSuff=''):
#    if outformat == 'zip':
#        # WRITE ZIP Train / Test Files >> FOR MASTERMovelets:
#        df2zip(data_path, train, file+'train', tid_col, class_col, select_cols=columns_order,\
#               opLabel='Writing TRAIN - ZIP|' + opSuff)
#        df2zip(data_path, test,  file+'test', tid_col, class_col, select_cols=columns_order, \
#               opLabel='Writing TEST  - ZIP|' + opSuff)
#        
#    elif outformat == 'csv':
#        print('Writing TRAIN / TEST - CSV|' + opSuff)
#        train[columns_order].to_csv(os.path.join(data_path, file+"train.csv"), index = False)
#        test[ columns_order].to_csv(os.path.join(data_path, file+"test.csv"),  index = False)
#        
#    elif outformat == 'mat':
#        # WRITE ZIP Train / Test Files >> FOR HiPerMovelets:
#        df2mat(train, data_path, file+'train', cols=columns_order, mat_cols=mat_columns, tid_col=tid_col, class_col=class_col, \
#               opLabel='Writing TRAIN - MAT|' + opSuff)
#        df2mat(test,  data_path, file+'test', cols=columns_order, mat_cols=mat_columns, tid_col=tid_col, class_col=class_col, \
#               opLabel='Writing TEST  - MAT|' + opSuff)
#        
    
#-------------------------------------------------------------------------->>
def splitframe(data, name='tid'):
#     from ..main import importer
#     importer(['S'], locals())
    
    n = data[name][0]

    df = pd.DataFrame(columns=data.columns)

    datalist = []

    for i in range(len(data)):
        if data[name][i] == n:
            df = df.append(data.iloc[i])
        else:
            datalist.append(df)
            df = pd.DataFrame(columns=data.columns)
            n = data[name][i]
            df = df.append(data.iloc[i])

    return datalist
    
#-------------------------------------------------------------------------->>    

#-------------------------------------------------------------------------->>
def joinTrainAndTest(dir_path, train_file="train.csv", test_file="test.csv", tid_col='tid', class_col = 'label', to_file=False):
#     from ..main import importer
#     importer(['S'], locals())
    
    print("Joining train and test data from... " + dir_path)
    
    # Read datasets
    dataset_train = readDataset(dir_path, None, train_file)
    dataset_test  = readDataset(dir_path, None, test_file)
    
    dataset = pd.concat([dataset_train, dataset_test])
#     if '.csv' in train_file:
#         print("Reading train file...")
#         dataset_train = pd.read_csv(os.path.join(dir_path, train_file))
#     else:
#         print("Converting train file...")
#         dataset_train = zip2csv(dir_path, train_file, cols, class_col)
#     print("Done.")
        
#     if '.csv' in test_file:
#         print("Reading test file...")
#         dataset_test  = pd.read_csv(os.path.join(dir_path, test_file))
#     else:
#         print("Converting test file...")
#         dataset_test = zip2csv(dir_path, test_file, cols, class_col)
    dataset.sort_values([class_col, tid_col])
    print("Done.")
    
    if to_file:
        print("Saving joined dataset as: " + os.path.join(dir_path, 'joined.csv'))

        dataset.to_csv(os.path.join(dir_path, 'joined.csv'), index=False)
        print("Done.")
    print(" --------------------------------------------------------------------------------")
    
    return dataset

#--------------------------------------------------------------------------------
def convertDataset(dir_path, k=None, cols = None, fileprefix='', tid_col='tid', class_col='label'):
    def convert_file(file, cols):
        df = readDataset(dir_path, fileprefix+file+'.csv')
#        if os.path.exists(os.path.join(dir_path, 'specific_'+file+'.csv')):
#            # Option 1:
#            df = pd.read_csv(os.path.join(dir_path, 'specific_'+file+'.csv'))
#        elif os.path.exists(os.path.join(dir_path, file+'.zip')):
#            df = convert_zip2csv(dir_path, file, cols, class_col)
#        else:
##             print("File "+file+" not found, nothing to do.")
#            raise Exception("File "+file+" not found, nothing to do.")
            
        if not cols:
            cols = list(df.columns)
        df, columns_order_zip, columns_order_csv = organizeFrame(df, cols, tid_col, class_col)
        
        outformats = []
        if not os.path.exists(os.path.join(dir_path, file+'.zip')):
            outformats.append('zip')
#            print("Saving dataset as: " + os.path.join(dir_path, file+'.zip'))
#            df2zip(dir_path, df, file, tid_col, class_col, select_cols=columns_order_zip)
        if not os.path.exists(os.path.join(dir_path, fileprefix+file+'.csv')):
            outformats.append('csv')
#            print("Saving dataset as: " + os.path.join(dir_path, file+'.csv'))
#            df[columns_order_csv].to_csv(os.path.join(dir_path, fileprefix+file+'.csv'), index = False)
        if not os.path.exists(os.path.join(dir_path, fileprefix+file+'.mat')):
            outformats.append('mat')
#            print("Saving dataset as: " + os.path.join(dir_path, file+'.mat'))
#            df[columns_order_csv].to_csv(os.path.join(dir_path, fileprefix+file+'.mat'), index = False)
            
        return df, columns_order_zip, columns_order_csv, outformats
        
    df_test, columns_order_zip, columns_order_csv, outformats = convert_file('test', cols)
    df_train, columns_order_zip, columns_order_csv, outformats = convert_file('train', cols)
    
    for outType in outformats:
        writeFiles(dir_path, fileprefix, df_train, df_test, tid_col, class_col, \
                 columns_order_zip if outType in ['zip', 'mat'] else columns_order_csv, None, outType, opSuff='')
    
    data = pd.concat([df_train,df_test])

    if k and not os.path.exists(os.path.join(dir_path, 'run1')):
        train, test = kfold_trainAndTestSplit(dir_path, k, data, fileprefix=fileprefix, random_num=1, tid_col=tid_col, class_col=class_col, columns_order=columns_order_csv)
        for i in range(1, k+1):
            for file in ['train', 'test']:
                os.rename(os.path.join(dir_path, 'run'+str(i), fileprefix+file+'.zip'), 
                          os.path.join(dir_path, 'run'+str(i), file+'.zip'))

        if 'space' in columns_order_zip:
            kfold_trainAndTestSplit(dir_path, k, None, random_num=1, fileprefix='raw_', tid_col=tid_col, class_col=class_col, columns_order=columns_order_csv, ktrain=train, ktest=test)
            for i in range(1, k+1):
                for file in ['train', 'test']:
                    os.remove(os.path.join(dir_path, 'run'+str(i), 'raw_'+file+'.zip'))
    
    print("All Done.")
#--------------------------------------------------------------------------------