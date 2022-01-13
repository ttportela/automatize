'''
Created on Jun, 2020

@author: Tarlis Portela
'''
# --------------------------------------------------------------------------------
# # PREPROCESSOR - DATASETS
# import os
# from zipfile import ZipFile
# import pandas as pd
# import numpy as np 
# import glob2
# import random
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
# import matplotlib.pyplot as plt
from .main import importer #, display
importer(['S'], globals())
# --------------------------------------------------------------------------------

def readData_Folders_File(path, col_names, file_ext='.txt', delimiter=',', file_prefix=''):
#     from ..main import importer
    importer(['S', 'glob'], globals())
    
    filelist = []
    filesList = []

    # 1: Build up list of files:
    for files in glob.glob(os.path.join(path, '**/*'+file_ext )):
        fileName, fileExtension = os.path.splitext(files)
        filelist.append(fileName) #filename without extension
        filesList.append(files) #filename with extension
    
    # 2: Create and concatenate in a DF:
    ct = 1
    df = pd.DataFrame()
    for ijk in filesList:
        frame = pd.read_csv(ijk, names=col_names, sep=delimiter)
        s = ijk[len(folder):-4].split("/")
        
        fname = s[0][:-3]
        
        frame['id']    = s[1]
        frame['tid']   = ct
        frame['label'] = str(fname).replace(file_prefix, '')
        frame['time']  = frame.index +1
        df = pd.concat([df,frame])
        ct += 1
#     df['t'] = df.index +1
#     df = df[['time', 'signal', 'tid', 'id', 'label']]
#     df.reset_index(level=0, inplace=True)
    return df

def readData_Files(path, col_names, file_ext='.txt', delimiter=',', file_prefix=''):   
#     from ..main import importer
    importer(['S', 'glob'], globals())
     
    filelist = []
    filesList = []

    # 1: Build up list of files:
    for files in glob.glob(os.path.join(path, '*'+file_ext)):
        fileName, fileExtension = os.path.splitext(files)
        filelist.append(fileName) #filename without extension
        filesList.append(files) #filename with extension
    
    # 2: Create and concatenate in a DF:
    ct = 1
    df = pd.DataFrame()
    for ijk in filesList:
        frame = pd.read_csv(ijk, names=col_names, sep=delimiter)
        
        fname = os.path.basename(ijk)[:-4]
        frame['label'] = str(fname).replace(file_prefix, '')
        
        #2.1: split by activity:
        ls = splitframe(frame, 'activity_id')
        for frameAux in ls:
            frameAux['tid']   = ct
            ct += 1
            frameAux.insert(0, 'time', range(1, 1 + len(frameAux)))
            df = pd.concat([df,frameAux])
    
    df.reset_index()
    
    return df
# ----------------------------------------------------------------------------->>


# --------------------------------------------------------------------------------
def printFeaturesJSON(df, version=1, deftype='nominal', defcomparator='equals', label_col='label', file=False):
    
    if isinstance(df, list):
        cols = df
    else:
        cols = list(df.columns)
            
    if version == 1:
        s = '{\n   "readsDesc": [\n'

        order = 1
        for f in cols:
            s += ('	{\n          "order": '+str(order)+',\n          "type": "'+deftype+'",\n          "text": "'+f+'"\n        }')
            if len(cols) == order:
                s += ('\n')
            else:
                s += (',\n')
            order += 1

        s += ('      ],\n    "pointFeaturesDesc": [\n      ],\n    "subtrajectoryFeaturesDesc": [\n	  ],\n')
        s += ('    "trajectoryFeaturesDesc": [\n      ],\n    "pointComparisonDesc": {\n      "pointDistance": "euclidean",\n')
        s += ('      "featureComparisonDesc": [\n')

        order = 1
        for f in cols:
            if f != 'tid' and f != label_col:
                s += ('			{\n			  "distance": "'+defcomparator+'",\n			  "maxValue": -1,\n			  "text": "'+f+'"\n			}')
                if len(cols) == order:
                    s += ('\n')
                else:
                    s += (',\n')
            order += 1

        s += ('		]\n    },\n    "subtrajectoryComparisonDesc": {\n      "subtrajectoryDistance": "euclidean",\n')
        s += ('      "featureComparisonDesc": [\n			{\n			  "distance": "euclidean",\n			  "text": "points"\n')
        s += ('			}\n		]\n    }\n}')
    else:        
        s  = '{\n   "input": {\n          "train": ["train"],\n          "test": ["test"],\n          "format": "CZIP",\n'
        s += '          "loader": "interning"\n   },\n'
        s += '   "idFeature": {\n          "order": '+str(cols.index('tid')+1)+',\n          "type": "numeric",\n          "text": "tid"\n    },\n'
        s += '   "labelFeature": {\n          "order": '+str(cols.index(label_col)+1)+',\n          "type": "nominal",\n          "text": "label"\n    },\n'
        s += '   "attributes": [\n'
        
        order = 1
        for f in cols:
            if f != 'tid' and f != label_col:
                s += '	    {\n	          "order": '+str(order)+',\n	          "type": "'+deftype+'",\n	          "text": "'+str(f)+'",\n	          "comparator": {\n	            "distance": "'+defcomparator+'"\n	          }\n	    }'
                if len(cols) == order:
                    s += ('\n')
                else:
                    s += (',\n')
            order += 1
        s += '	]\n}'
        
    if file:
        file = open(file, 'w')
        print(s, file=file)
        file.close()
    else:
        print(s)
    
#-------------------------------------------------------------------------->>
def readDataset(data_path, folder, file='train.csv', class_col = 'label'):
#     from ..main import importer
#     importer(['S'], locals())
    
    if '.csv' in file or '.zip' in file:
        url = os.path.join(data_path, folder, file)
    else:
        url = os.path.join(data_path, folder, file+'.csv')
    
    if (not os.path.exists(url)) and '.csv' in file:
        file = file.replace('.csv', '.zip')
        df = convert_zip2csv(os.path.join(data_path, folder), file, class_col=class_col)
    else:
        df = pd.read_csv(url)
    return df

def countClasses(data_path, folder, file='train.csv', class_col = 'label'):
    df = readDataset(data_path, folder, file, class_col)
    return countClasses_df(df, class_col)

def countClasses_df(df, class_col = 'label'):
    group = df.groupby([class_col, 'tid'])
    df2 = group.apply(lambda x: x[class_col].unique())
    print("Number of Samples: " + str(len(df['tid'].unique())))
    print("Samples by Class:")
    print(df2.value_counts())
    
    return df2.value_counts()

def datasetStatistics(data_path, folder, file_prefix='', class_col = 'label'):
#     from ..main import importer
#     importer(['S'], locals())
    
    train = readDataset(data_path, folder, file_prefix+'train.csv', class_col)
    test = readDataset(data_path, folder, file_prefix+'test.csv', class_col)
    print('\n--------------------------------------------------------------------')
    print('Descriptive Statistics for', folder)
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
    
    print('Number of Classes:     =>', len(classes))
    print('Number of Attributes:  =>', len(train.columns))
    print('Number of Trajs:       =>', samples, 'total /', sam_train, 'train +', sam_test, 'test')
    print('Number of Points:      =>', points, 'total /', len(train), 'train +', len(test), 'test')
    print('Avg Size of Trajs:     =>', '{:.2f}'.format(avg_size), ' / ±', diff_size)
    print('Longest Size:          =>', top_train, 'train /', top_test, 'test')
    print('Shortest Size:         =>', bot_train, 'train /', bot_test, 'test')
    print('Train / Test hold-out: =>', sam_train, '-', '{:.2f}% ..'.format(sam_train*100/samples),
                                       sam_test,  '-', '{:.2f}%'.format(sam_test*100/samples))
    
    print('\n--------------------------------------------------------------------')
    print('Attributes: ')
    print(list(train.columns))
    print('\nFeatures Selection (by Variance): ')
    stats=pd.DataFrame()
    stats["mean"]=train.mean()
    stats["Std.Dev"]=train.std()
    stats["Var"]=train.var()
    print(stats.sort_values('Var', ascending=False))
    print('\nClasses: ')
    print(classes, '\n')
    print('\n--------------------------------------------------------------------')
    print('Statistics from TRAIN:')
    countClasses_df(train)
    print()
    train.describe()
    print('\n--------------------------------------------------------------------')
    print('Statistics from TEST.:')
    countClasses_df(test)
    print()
    test.describe()

#-------------------------------------------------------------------------->>
def trainAndTestSplit(data_path, df, train_size=0.7, random_num=1, tid_col='tid', class_col='label', fileprefix=''):
#     from ..main import importer
    importer(['S', 'random'], globals())
    
    # TODO separete/join lat, lon <=> space
#     df[["lat", "lon"]] = df[["lat", "lon"]].astype(str) # joins:
#     df['space'] = df['lat'] + ' ' + df['lon']
#     ll = df['space'].str.split(" ", n = 1, expand = True) # separate:
#     df["lat"]= ll[0]
#     df["lon"]= ll[1]
    
    train = pd.DataFrame()
    test = pd.DataFrame()
    for label in df[class_col].unique():
        
        tids = df.loc[df[class_col] == label][tid_col].unique()
        print(label)
        print(tids)
        
        random.seed(random_num)
        train_index = random.sample(list(tids), int(len(tids)*train_size))
        test_index  = tids[np.isin(tids, train_index, invert=True)] #np.delete(test_index, train_index)

        train = pd.concat([train,df.loc[df[tid_col].isin(train_index)]])
        test  = pd.concat([test, df.loc[df[tid_col].isin(test_index)]])

        print("Train samples: " + str(train.loc[train[class_col] == label][tid_col].unique()))
        print("Test samples: " + str(test.loc[test[class_col] == label][tid_col].unique()))
    
#     # WRITE Train / Test Files >> FOR MASTERMovelets:
#     # TODO here goes space column
#     createZIP(data_path, train, fileprefix+'train', tid_col, class_col)
#     createZIP(data_path, test, fileprefix+'test', tid_col, class_col)

#     # WRITE Train / Test Files >> FOR V3:
#     # TODO here goes lat,lon columns
#     train.to_csv(os.path.join(data_path, fileprefix+"train.csv"), index = False)
#     test.to_csv(os.path.join(data_path, fileprefix+"test.csv"), index = False)
    
    write_trainAndTest(data_path, train, test, tid_col, class_col, fileprefix)
    
    return train, test

def write_trainAndTest(data_path, train, test, tid_col='tid', class_col='label', fileprefix=''):
#     from ..main import importer
#     importer(['S'], locals())
    
    # WRITE Train / Test Files >> FOR MASTERMovelets:
    # TODO here goes space column
    createZIP(data_path, train, fileprefix+'train', tid_col, class_col)
    createZIP(data_path, test, fileprefix+'test', tid_col, class_col)

    # WRITE Train / Test Files >> FOR V3:
    # TODO here goes lat,lon columns
    train.to_csv(os.path.join(data_path, fileprefix+"train.csv"), index = False)
    test.to_csv(os.path.join(data_path, fileprefix+"test.csv"), index = False)
    
    return train, test

#-------------------------------------------------------------------------->>
def organizeFrame(df, columns_order, tid_col='tid', class_col='label'):
    if (set(df.columns) & set(['lat', 'lon'])) and not 'space' in df.columns:
        df[["lat", "lon"]] = df[["lat", "lon"]].astype(str) 
        df['space'] = df["lat"] + ' ' + df["lon"]

        if columns_order is not None:
            columns_order.insert(columns_order.index('lat'), 'space')
            
    elif 'space' in df.columns and not (set(df.columns) & set(['lat', 'lon'])):
        ll = df['space'].str.split(" ", n = 1, expand = True) 
        df["lat"]= ll[0]
        df["lon"]= ll[1]
        
        if columns_order is not None:
            columns_order.insert(columns_order.index('space')+1, 'lat')
            columns_order.insert(columns_order.index('space')+2, 'lon')
        
    # For Columns ordering:
    if columns_order is None:
        columns_order = df.columns
            
    columns_order = [x for x in columns_order if x not in [tid_col, class_col]]
    columns_order = columns_order + [tid_col, class_col]
            
    columns_order_zip = [x for x in columns_order if x not in ['lat', 'lon']]
    columns_order_csv = [x for x in columns_order if x not in ['space']]
    
    return columns_order_zip, columns_order_csv

def kfold_trainAndTestSplit(data_path, k, df, random_num=1, tid_col='tid', class_col='label', fileprefix='', 
                            columns_order=None, ktrain=None, ktest=None):
#     from ..main import importer
    importer(['S', 'KFold'], globals())
    
    print(str(k)+"-fold train and test split in... " + data_path)
    
    columns_order_zip, columns_order_csv = organizeFrame(df, columns_order, tid_col, class_col)
    
    if not ktrain:
        ktrain = []
        ktest = []
        for x in range(k):
            ktrain.append( pd.DataFrame() )
            ktest.append( pd.DataFrame() )


        print("Spliting data...")
        kfold = KFold(n_splits=k, shuffle=True, random_state=random_num)
        for label in df[class_col].unique(): 
            tids = df.loc[df[class_col] == label][tid_col].unique()
    #         print("Trajectory IDs for label: " + label)
    #         print(tids)

            x = 0
            for train_idx, test_idx in kfold.split(tids):
                ktrain[x] = pd.concat([ktrain[x], df.loc[df[tid_col].isin(tids[train_idx])]])
                ktest[x]  = pd.concat([ktest[x],  df.loc[df[tid_col].isin(tids[test_idx])]])
                x += 1
        print("Done.")
    else:
        print("Train and test data provided.")
    
    print("Writing files...")
    for x in range(k):
        path = 'run'+str(x+1)
        
        if not os.path.exists(os.path.join(data_path, path)):
            os.makedirs(os.path.join(data_path, path))
            
        train_aux = ktrain[x]
        test_aux  = ktest[x]
            
        
        # WRITE ZIP Train / Test Files >> FOR MASTERMovelets:
        createZIP(data_path, train_aux, os.path.join(path, fileprefix+'train'), tid_col, class_col, select_cols=columns_order_zip)
        createZIP(data_path, test_aux, os.path.join(path,  fileprefix+'test'), tid_col, class_col, select_cols=columns_order_zip)

        # WRITE CSV Train / Test Files >> FOR HIPERMovelets:
        train_aux[columns_order_csv].to_csv(os.path.join(data_path, path, fileprefix+"train.csv"), index = False)
        test_aux[columns_order_csv].to_csv(os.path.join(data_path, path,  fileprefix+"test.csv"), index = False)
    print("Done.")
    print(" --------------------------------------------------------------------------------")
    
    return ktrain, ktest

def stratify(data_path, df, k=10, inc=1, limit=10, random_num=1, tid_col='tid', class_col='label', fileprefix='', 
                            columns_order=None, ktrain=None, ktest=None):
#     from ..main import importer
    importer(['S', 'KFold'], globals())
    
    print(str(k)+"-fold stratification of train and test in... " + data_path)
    
    columns_order_zip, columns_order_csv = organizeFrame(df, columns_order, tid_col, class_col)
        
    if not ktrain:
        ktrain = []
        ktest = []
        for x in range(k):
            ktrain.append( pd.DataFrame() )
            ktest.append( pd.DataFrame() )

        print("Spliting data...")
        kfold = KFold(k, True, random_num)
        for label in df[class_col].unique(): 
            tids = df.loc[df[class_col] == label].tid.unique()
    #         print("Trajectory IDs for label: " + label)
    #         print(tids)

            x = 0
            for train_idx, test_idx in kfold.split(tids):
                ktrain[x] = pd.concat([ktrain[x], df.loc[df[tid_col].isin(tids[train_idx])]])
                ktest[x]  = pd.concat([ktest[x],  df.loc[df[tid_col].isin(tids[test_idx])]])
                x += 1
        print("Done.")
    else:
        print("Train and test data provided.")
    
    print("Writing files...")
    for x in range(0, limit, inc):
        path = 'S'+str((x+1)*int(100/k))
        
        if not os.path.exists(os.path.join(data_path, path)):
            os.makedirs(os.path.join(data_path, path))
            
        train_aux = ktrain[0]
        test_aux  = ktest[0]
        for y in range(1, x+1):
            train_aux = pd.concat([train_aux,  ktrain[y]])
            test_aux  = pd.concat([test_aux,   ktest[y]])
            
        
        # WRITE ZIP Train / Test Files >> FOR MASTERMovelets:
        createZIP(data_path, train_aux, os.path.join(path, fileprefix+'train'), tid_col, class_col, select_cols=columns_order_zip)
        createZIP(data_path, test_aux, os.path.join(path,  fileprefix+'test'), tid_col, class_col, select_cols=columns_order_zip)

        # WRITE CSV Train / Test Files >> FOR HIPERMovelets:
        train_aux[columns_order_csv].to_csv(os.path.join(data_path, path, fileprefix+"train.csv"), index = False)
        test_aux[columns_order_csv].to_csv(os.path.join(data_path, path,  fileprefix+"test.csv"), index = False)
        print(path + '; ', end='')
    print(" Done.")
    print(" --------------------------------------------------------------------------------")
    
    return ktrain, ktest
        
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
def createZIP(data_path, df, file, tid_col='tid', class_col='label', select_cols=None):
#     from ..main import importer
    importer(['S', 'zip'], globals())
    
    EXT = '.r2'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    zipf = ZipFile(os.path.join(data_path, file+'.zip'), 'w')
    
    n = len(str(len(df.index)))
    tids = df[tid_col].unique()
    for x in tids:
        filename = str(x).rjust(n, '0') + ' s' + str(x) + ' c' + str(df.loc[df[tid_col] == x][class_col].iloc[0]) + EXT
        data = df[df.tid == x]
        if select_cols is not None:
            data = data[select_cols]
        
        # Remove tid and label:
        data = data.drop([tid_col, class_col], axis=1)
        
        data.to_csv(filename, index=False, header=False)
        zipf.write(filename)
        os.remove(filename)
    
    # close the Zip File
    zipf.close()
    
#-------------------------------------------------------------------------->>
def convert_zip2csv(folder, file, cols=None, class_col = 'label'):
#     from ..main import importer
    importer(['S', 'zip'], globals())
    
    data = pd.DataFrame()
    print("Converting "+file+" data from... " + folder)
    if '.zip' in file:
        url = os.path.join(folder, file)
    else:
        url = os.path.join(folder, file+'.zip')
    with ZipFile(url) as z:
        files = z.namelist()
        files.sort()
        for filename in files:
#             data = filename.readlines()
#             print(filename)
            if cols is not None:
                df = pd.read_csv(z.open(filename), names=cols)
            else:
                df = pd.read_csv(z.open(filename), header=None)
            df['tid']   = filename.split(" ")[1][1:]
            df[class_col] = filename.split(" ")[2][1:-3]
            data = pd.concat([data,df])
    print("Done.")
    return data
    
def zip2csv(folder, file, cols, class_col = 'label'):
#     from ..main import importer
#     importer(['S'], locals())
    
#     data = pd.DataFrame()
#     print("Converting "+file+" data from... " + folder)
#     if '.zip' in file:
#         url = os.path.join(folder, file)
#     else:
#         url = os.path.join(folder, file+'.zip')
#     with ZipFile(url) as z:
#         for filename in z.namelist():
# #             data = filename.readlines()
#             df = pd.read_csv(z.open(filename), names=cols)
# #             print(filename)
#             df['tid']   = filename.split(" ")[1][1:]
#             df[class_col] = filename.split(" ")[2][1:-3]
#             data = pd.concat([data,df])
#     print("Done.")
    data = convert_zip2csv(folder, file, cols, class_col)
    print("Saving dataset as: " + os.path.join(folder, file+'.csv'))
    data.to_csv(os.path.join(folder, file+'.csv'), index = False)
    print("Done.")
    print(" --------------------------------------------------------------------------------")
    return data

def convertToCSV(path): 
#     from ..main import importer
#     importer(['S'], locals())
    
    dir_path = os.path.dirname(os.path.realpath(path))
    files = [x for x in os.listdir(dir_path) if x.endswith('.csv')]

    for file in files:
        try:
            df = pd.read_csv(file, sep=';', header=None)
            print(df)
            df.drop(0, inplace=True)
            print(df)
            df.to_csv(os.path.join(folder, file), index=False, header=None)
        except:
            pass

def zip2arf(folder, file, cols, tid_col='tid', class_col = 'label'):
    data = pd.DataFrame()
    print("Converting "+file+" data from... " + folder)
    if '.zip' in file:
        url = os.path.join(folder, file)
    else:
        url = os.path.join(folder, file+'.zip')
    with ZipFile(url) as z:
        for filename in z.namelist():
#             data = filename.readlines()
            df = pd.read_csv(z.open(filename), names=cols)
#             print(filename)
            df[tid_col]   = filename.split(" ")[1][1:]
            df[class_col] = filename.split(" ")[2][1:-3]
            data = pd.concat([data,df])
    print("Done.")
    
    print("Saving dataset as: " + os.path.join(folder, file+'.csv'))
    data.to_csv(os.path.join(folder, file+'.csv'), index = False)
    print("Done.")
    print(" --------------------------------------------------------------------------------")
    return data

def convert2ts(data_path, folder, file, cols=None, tid_col='tid', class_col = 'label'):
    print("Converting "+file+" data from... " + data_path + " - " + folder)
    data = readDataset(data_path, folder, file, class_col)
    
    file = file.replace('specific_',  '')
    
    tsName = os.path.join(data_path, folder, folder+'_'+file.upper()+'.ts')
    tsDesc = os.path.join(data_path, folder, folder+'.md')
    print("Saving dataset as: " + tsName)
    if cols == None:
        cols = [x for x in data.columns if x not in [tid_col, class_col]]
    
    f = open(tsName, "w")
    
    if os.path.exists(tsDesc):
        fd = open(tsDesc, "r")
        for line in fd:
            f.write("# " + line)
#         fd.close()

    f.write("#\n")
    f.write("@problemName " + folder + '\n')
    f.write("@univariate false\n")
    f.write("@dimensions " + str(len(cols)) + '\n')
#     f.write("@equalLength true" + '\n')
#     f.write("@seriesLength " + ? + '\n')
    f.write("@classLabel true " + ' '.join([str(x).replace(' ', '_') for x in list(data[class_col].unique())]) + '\n')
    f.write("@data\n")
    
    for tid in data[tid_col].unique():
        df = data[data[tid_col] == tid]
        line = ''
        for col in cols:
            line += ','.join(map(str, list(df[col]))) + ':'
        f.write(line + str(df[class_col].unique()[0]) + '\n')
        
    f.write('\n')
    f.close()
    print("Done.")
    print(" --------------------------------------------------------------------------------")
    return data

def joinTrainAndTest(dir_path, cols, train_file="train.csv", test_file="test.csv", tid_col='tid', class_col = 'label'):
#     from ..main import importer
#     importer(['S'], locals())
    
    print("Joining train and test data from... " + dir_path)
    
    # Read datasets
    if '.csv' in train_file:
        print("Reading train file...")
        dataset_train = pd.read_csv(os.path.join(dir_path, train_file))
    else:
        print("Converting train file...")
        dataset_train = zip2csv(dir_path, train_file, cols, class_col)
    print("Done.")
        
    if '.csv' in test_file:
        print("Reading test file...")
        dataset_test  = pd.read_csv(os.path.join(dir_path, test_file))
    else:
        print("Converting test file...")
        dataset_test = zip2csv(dir_path, test_file, cols, class_col)
    print("Done.")
        
    print("Saving joined dataset as: " + os.path.join(dir_path, 'joined.csv'))
    dataset = pd.concat([dataset_train, dataset_test])

    dataset.sort_values([class_col, tid_col])
    
    dataset.to_csv(os.path.join(dir_path, 'joined.csv'), index=False)
    print("Done.")
    print(" --------------------------------------------------------------------------------")
    
    return dataset

#--------------------------------------------------------------------------------
def convertDataset(dir_path, k=None, cols = None, tid_col='tid', class_col='label'):
    def convert_file(file, cols):
        if os.path.exists(os.path.join(dir_path, 'specific_'+file+'.csv')):
            # Option 1:
            df = pd.read_csv(os.path.join(dir_path, 'specific_'+file+'.csv'))
        elif os.path.exists(os.path.join(dir_path, file+'.zip')):
            df = convert_zip2csv(dir_path, file, cols, class_col)
        else:
            print("File "+file+" not found, nothing to do.")
            
        if not cols:
            cols = df.columns
        columns_order_zip, columns_order_csv = organizeFrame(df, cols, tid_col, class_col)
        
        if os.path.exists(os.path.join(dir_path, 'specific_'+file+'.csv')) and \
            not os.path.exists(os.path.join(dir_path, file+'.zip')):
            print("Saving dataset as: " + os.path.join(dir_path, file+'.zip'))
#             os.rename(os.path.join(dir_path, file+'.zip'),os.path.join(dir_path, file+'-old.zip'))
            createZIP(dir_path, df, file, tid_col, class_col, select_cols=columns_order_zip)
        elif os.path.exists(os.path.join(dir_path, file+'.zip')) and \
            not os.path.exists(os.path.join(dir_path, 'specific_'+file+'.csv')):
            print("Saving dataset as: " + os.path.join(dir_path, file+'.csv'))
            df[columns_order_csv].to_csv(os.path.join(dir_path, 'specific_'+file+'.csv'), index = False)
            
        return df, columns_order_zip, columns_order_csv
        
    df_test, columns_order_zip, columns_order_csv = convert_file('test', cols)
    df_train, columns_order_zip, columns_order_csv = convert_file('train', cols)
    data = pd.concat([df_train,df_test])

    if k and not os.path.exists(os.path.join(dir_path, 'run1')):
        train, test = kfold_trainAndTestSplit(dir_path, k, data, fileprefix='specific_', random_num=1, tid_col=tid_col, class_col=class_col, columns_order=columns_order_csv)
        for i in range(1, k+1):
            for file in ['train', 'test']:
                os.rename(os.path.join(dir_path, 'run'+str(i), 'specific_'+file+'.zip'), 
                          os.path.join(dir_path, 'run'+str(i), file+'.zip'))

        if 'space' in columns_order_zip:
            kfold_trainAndTestSplit(dir_path, k, None, random_num=1, fileprefix='raw_', tid_col=tid_col, class_col=class_col, columns_order=columns_order_csv, ktrain=train, ktest=test)
            for i in range(1, k+1):
                for file in ['train', 'test']:
                    os.remove(os.path.join(dir_path, 'run'+str(i), 'raw_'+file+'.zip'))
    
    print("All Done.")

# --------------------------------------------------------------------------------
