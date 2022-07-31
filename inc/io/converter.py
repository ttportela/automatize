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
from main import importer #, display
importer(['S', 'tqdm'], globals())


def zip2df(folder, file, cols=None, class_col = 'label', tid_col='tid', missing='?'):
#     from ..main import importer
    importer(['S', 'zip'], globals())
    
#     data = pd.DataFrame()
    print("Converting "+file+" data from... " + folder)
    if '.zip' in file:
        url = os.path.join(folder, file)
    else:
        url = os.path.join(folder, file+'.zip')
        
    print("Done.")
    data = read_zip(ZipFile(url), cols, class_col, tid_col, missing)
    return data
    
def read_zip(zipFile, cols=None, class_col='label', tid_col='tid', missing='?', opLabel='Reading ZIP'):
    data = pd.DataFrame()
    with zipFile as z:
        files = z.namelist()
        files.sort()
#         for filename in files:
#             if cols is not None:
#                 df = pd.read_csv(z.open(filename), names=cols, na_values=missing)
#             else:
#                 df = pd.read_csv(z.open(filename), header=None, na_values=missing)
#             df[tid_col]   = filename.split(" ")[1][1:]
#             df[class_col] = filename.split(" ")[2][1:-3]
#             data = pd.concat([data,df])
        def readCSV(filename):
            if cols is not None:
                df = pd.read_csv(z.open(filename), names=cols, na_values=missing)
            else:
                df = pd.read_csv(z.open(filename), header=None, na_values=missing)
            df[tid_col]   = filename.split(" ")[1][1:]
            df[class_col] = filename.split(" ")[2][1:-3]
            return df
        data = list(map(lambda filename: readCSV(filename), tqdm(z.namelist(), desc=opLabel)))
        data = pd.concat(data)
    return data

#-------------------------------------------------------------------------->>
def zip2csv(folder, file, cols, class_col = 'label', tid_col='tid', missing='?'):
#     from ..main import importer
#     importer(['S'], locals())

    data = zip2df(folder, file, cols, class_col, tid_col, missing)
    print("Saving dataset as: " + os.path.join(folder, file+'.csv'))
    data.to_csv(os.path.join(folder, file+'.csv'), index = False)
    print("Done.")
    print(" --------------------------------------------------------------------------------")
    return data

def df2zip(data_path, df, file, tid_col='tid', class_col='label', select_cols=None, opLabel='Writing MAT'):
#     from ..main import importer
    importer(['S', 'zip'], globals())
    
    EXT = '.r2'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    zipf = ZipFile(os.path.join(data_path, file+'.zip'), 'w')
    
    n = len(str(len(df.index)))
    tids = df[tid_col].unique()
    
    if not select_cols:
        select_cols = list(df.columns)
    select_cols = [x for x in select_cols if x not in [tid_col, class_col]]
    
    def writeMAT(x):
        filename = str(x).rjust(n, '0') + ' s' + str(x) + ' c' + str(df.loc[df[tid_col] == x][class_col].iloc[0]) + EXT
        data = df[df.tid == x]
        # Selected
        if select_cols is not None:
            data = data[select_cols]
        
        # Remove tid and label:
        data = data.drop([tid_col, class_col], axis=1, errors='ignore')
        
        data.to_csv(filename, index=False, header=False)
        zipf.write(filename)
        os.remove(filename)
    list(map(lambda x: writeMAT(x), tqdm(tids, desc=opLabel)))
#     for x in tids:
#         filename = str(x).rjust(n, '0') + ' s' + str(x) + ' c' + str(df.loc[df[tid_col] == x][class_col].iloc[0]) + EXT
#         data = df[df.tid == x]
#         if select_cols is not None:
#             data = data[select_cols]
        
#         # Remove tid and label:
#         data = data.drop([tid_col, class_col], axis=1)
        
#         data.to_csv(filename, index=False, header=False)
#         zipf.write(filename)
#         os.remove(filename)
    
    # close the Zip File
    zipf.close()
#--------------------------------------------------------------------------------

# def convertToCSV(path): 
# #     from ..main import importer
# #     importer(['S'], locals())
    
#     dir_path = os.path.dirname(os.path.realpath(path))
#     files = [x for x in os.listdir(dir_path) if x.endswith('.csv')]

#     for file in files:
#         try:
#             df = pd.read_csv(file, sep=';', header=None)
#             print(df)
#             df.drop(0, inplace=True)
#             print(df)
#             df.to_csv(os.path.join(folder, file), index=False, header=None)
#         except:
#             pass

def zip2arf(folder, file, cols, tid_col='tid', class_col = 'label', missing='?', opLabel='Reading CSV'):
    data = pd.DataFrame()
    print("Converting "+file+" data from... " + folder)
    if '.zip' in file:
        url = os.path.join(folder, file)
    else:
        url = os.path.join(folder, file+'.zip')
    with ZipFile(url) as z:
#         for filename in z.namelist():
# #             data = filename.readlines()
#             df = pd.read_csv(z.open(filename), names=cols, na_values=missing)
# #             print(filename)
#             df[tid_col]   = filename.split(" ")[1][1:]
#             df[class_col] = filename.split(" ")[2][1:-3]
#             data = pd.concat([data,df])
        def readCSV(filename):
#             data = filename.readlines()
            df = pd.read_csv(z.open(filename), names=cols, na_values=missing)
#             print(filename)
            df[tid_col]   = filename.split(" ")[1][1:]
            df[class_col] = filename.split(" ")[2][1:-3]
            return df
        data = list(map(lambda filename: readCSV(filename), tqdm(z.namelist(), desc=opLabel)))
        data = pd.concat(data)
    print("Done.")
    
    print("Saving dataset as: " + os.path.join(folder, file+'.csv'))
    data.to_csv(os.path.join(folder, file+'.csv'), index = False)
    print("Done.")
    print(" --------------------------------------------------------------------------------")
    return data

def any2ts(data_path, folder, file, cols=None, tid_col='tid', class_col = 'label', opLabel='Converting TS'):
    print("Converting "+file+" data from... " + data_path + " - " + folder)
    data = readDataset(data_path, folder, file, class_col)
    
    file = file.replace('specific_',  '')
    
    tsName = os.path.join(data_path, folder, folder+'_'+file.upper()+'.ts')
    tsDesc = os.path.join(data_path, folder, folder+'.md')
    print("Saving dataset as: " + tsName)
    if not cols:
        cols = list(data.columns)
    cols = [x for x in cols if x not in [tid_col, class_col]]
    
    f = open(tsName, "w")
    
    if os.path.exists(tsDesc):
        fd = open(tsDesc, "r")
        for line in fd:
            f.write("# " + line)
#         fd.close()

    f.write("#\n")
    f.write("@problemName " + folder + '\n')
    f.write("@timeStamps false")
    f.write("@missing "+ str('?' in data)+'\n')
    f.write("@univariate "+ ('false' if len(cols) > 1 else 'true') +'\n')
    f.write("@dimensions " + str(len(cols)) + '\n')
    f.write("@equalLength false" + '\n')
    f.write("@seriesLength " + str(len(data[data[tid_col] == data[tid_col][0]])) + '\n')
    f.write("@classLabel true " + ' '.join([str(x).replace(' ', '_') for x in list(data[class_col].unique())]) + '\n')
    f.write("@data\n")
    
#     for tid in data[tid_col].unique():
    def writeLine(tid):
        df = data[data[tid_col] == tid]
        line = ''
        for col in cols:
            line += ','.join(map(str, list(df[col]))) + ':'
        f.write(line + str(df[class_col].unique()[0]) + '\n')
    list(map(lambda tid: writeLine(tid), tqdm(data[tid_col].unique(), desc=opLabel)))
    
    f.write('\n')
    f.close()
    print("Done.")
    print(" --------------------------------------------------------------------------------")
    return data

def xes2csv(folder, file, cols=None, tid_col='tid', class_col = 'label', opLabel='Converting XES', save=False, start_tid=1):
    start_tid = start_tid-1
    def getTrace(log, tid):
        t = dict(log[tid].attributes)
    #     t.update(log[tid].attributes)
        return t
    
    def getEvent(log, tid , j, attrs):
        ev = dict(log[tid][j])
        
        eqattr = set(attrs.keys()).intersection(set(ev.keys()))
        for k in eqattr:
            attrs[k+'_t'] = attrs.pop(k)
        
        ev.update(attrs)
        ev['tid'] = start_tid+tid+1
        return ev
    
    
    import pm4py
    if '.xes' in file:
        url = os.path.join(folder, file)
    else:
        url = os.path.join(folder, file+'.xes')
    
    print("Reading "+file+" data from: " + folder)
    log = pm4py.read_xes(url)
    
#     if show_progress:
#         import tqdm
    data = list(map(lambda tid: 
                pd.DataFrame(list(map(lambda j: getEvent(log, tid , j, getTrace(log, tid)), range(len(log[tid]))))),
                tqdm(range(len(log)), desc=opLabel)))
#     else:
#         data = list(map(lambda tid: 
#                     pd.DataFrame(list(map(lambda j: getEvent(log, tid , j, getTrace(log, tid)), range(len(log[tid]))))),
#                     range(len(log))))

    df = pd.concat(data, ignore_index=True)
    
    if save:
        print("Saving dataset as: " + os.path.join(folder, file+'.csv'))
        df.to_csv(os.path.join(folder, file+'.csv'), index = False)

    print("Done.")
    print(" --------------------------------------------------------------------------------")
    return df

def df2mat(df, folder, file, cols=None, mat_cols=None, desc_cols=None, label_columns=None, other_dsattrs=None,
           tid_col='tid', class_col='label', opLabel='Converting MAT'):
    
    if '.mat' in file:
        url = os.path.join(folder, file)
        file = file.replace('.mat', '')
    else:
        url = os.path.join(folder, file+'.mat')
    
#     print("Converting data to MAT ...")
    
    if not cols:
        cols = list(df.columns)
    cols = [x for x in cols if x not in [tid_col, class_col]]
    
    if mat_cols:
        mat_cols = [x for x in mat_cols if x not in [tid_col, class_col]]
    
    f = open(url, "w")
    f.write("# Dataset: " + os.path.basename(folder) + ' (comment description)\n')
    f.write("@problemName " + os.path.basename(folder) + '\n')
    
    if label_columns:
        f.write('@labelColumns ' + (','.join(label_columns)) + '\n')
        
    f.write("@missing "+ str(df.apply(lambda ts: '?' in ts.values, axis=1).any() or df.isnull().any().any())+'\n')
    f.write("@aspects " + str(len(cols)) + '\n')
    f.write('@aspectNames ' + (','.join(cols)) + '\n')
    if mat_cols:
        f.write('@trajectoryAspectNames ' + (','.join(mat_cols)) + '\n')
        
    if not desc_cols:
        # dictionary in the format: {'aspectName': 'type', 'aspectName': 'type'}
        desc_cols = descTypes(df)    
    f.write('@aspectDescriptor ' + (','.join(':'.join((key,val)) for (key,val) in desc_cols.items())) + '\n')
    
    if other_dsattrs:
        for k,v in other_dsattrs:
            f.write('@'+k+' ' + (','.join(v)) + '\n')
    
    f.write("@data\n")
    def getTrace(df, tid):
        s = ''
        s += '@trajectory \n' + str(tid) + ',' + str(df[class_col].values[0]) + '\n'
        if mat_cols:
            s += '@trajectoryAspects\n'
            s += df[mat_cols][0:1].to_csv(index=False,header=False, quotechar='"')
#             s += '\n'
                
        s += '@trajectoryPoints\n'
        s += df[cols].to_csv(index=False,header=False, quotechar='"')
#         s += '\n'
#         print(s)
        return s
           
#     import tqdm
#     if tqdm.notebook:
#         from tqdm.notebook import tqdm as tq
#         list(map(lambda tid: f.write(getTrace(df[df[tid_col] == tid], tid)),
#                 tqdm(df[tid_col].unique(), desc='Converting')))
#     else:
#         from tqdm import tqdm as tq
#         list(map(lambda tid: f.write(getTrace(df[df[tid_col] == tid], tid)),
#                 df[tid_col].unique()))
    list(map(lambda tid: f.write(getTrace(df[df[tid_col] == tid], tid)),
            tqdm(df[tid_col].unique(), desc=opLabel)))

    f.close()
#     df = pd.concat(data, ignore_index=True)

#     print("Done.")
#     print(" --------------------------------------------------------------------------------")

def mat2df(folder, file, cols=None, class_col = 'label', tid_col='tid', missing='?'):
    
    if '.mat' in file:
        url = os.path.join(folder, file)
    else:
        url = os.path.join(folder, file+'.mat')

# --------------------------------------------------------------------------------
def descTypes(df):
    def getType(k, t):
        if t.name == 'category':
            return 'nominal'
        elif t is int or t is float or np.issubdtype(t, np.number):
            return 'numeric'
        elif 'space' in k or 'lat_lon' in k:
            return 'space2d'
        else:
            return 'nominal'
            
    return {k: getType(k, df.dtypes[k]) for k in df.columns}
    