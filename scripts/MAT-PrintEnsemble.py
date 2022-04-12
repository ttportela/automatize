# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
@deprecated
'''
import sys, os 
sys.path.insert(0, os.path.abspath('.'))
import pandas as pd
import numpy as np

from automatise.main import display
# from main import importer, display
# importer(['S'], globals())

# print(len(sys.argv))
if len(sys.argv) < 2:
    print('Please run as:')
    print('\tPrintEnsemble.py', '"PATH TO FOLDER/Ensemble/Prefix"')
    print('Example:')
    print('\tPrintEnsemble.py', '"./results/method/dataset"')
    print('OR:')
    print('\tPrintEnsemble.py', '"PATH TO FOLDER"', '"subset1,subset2"', '"pois_attribute"')
    print('\tPrintEnsemble.py', '"./results/method/dataset"', '"specific"', '"poi"')
    exit()

results_path = sys.argv[1]
results_path = os.path.abspath(results_path).rsplit(os.sep, 2)

# Pre-confg:
method = results_path[1]
prefix = results_path[2]
results_path = results_path[0]
ds  = 'specific'
dim = '*'

if len(sys.argv) > 2:
    ds = sys.argv[2] 
if len(sys.argv) > 3:
    dim = sys.argv[3]     

    

# dirr = os.path.join(results_path)
#coringa = ""

# df = results2df(dirr, '', method)
list_stats = [
    ['ACC Movelets', 'ACC','movelets movelets_nn movelets_mlp'],
    ['ACC Npoi', 'ACC','poi npoi wnpoi'],
    ['ACC Marc', 'ACC','marc'],
    ['ACC TEC', 'ACC','EnsembleClassifier'],
    ['Time (Movelets)',  'time',    'movelets movelets_nn movelets_mlp'],
    ['Time (Npoi)',      'time',    'poi npoi wnpoi'],
    ['Time (Marc)',      'time',    'marc'],
    ['Time (TEC)',      'time',    'EnsembleClassifier'],
]

# def display(table):
#     print(table)

def createdf(prefix, list_stats=list_stats):
#     import pandas as pd
#     import numpy as np
    df = pd.DataFrame()
    df[' '] = np.array(list_stats)[:,0]
    df['Dataset'] = ""
    df['Dataset'][0] = prefix
    return df

def printECResults(res_path, methods_dic, prefixes, datasets, to_csv=False, print_latex=False, isformat=False):
#     importer(['S', 'results2df', 'printLatex', 'STATS', 'format_stats'], globals())
    from results import results2df, printLatex, format_stats, STATS
    
    run = 'run*'
    num_runs = 1
    
    for prefix in prefixes:
        table = pd.DataFrame()
        for dataset in datasets:
            data = pd.DataFrame()
            for folder, methods in methods_dic.items():
                for method in methods:
#                     the_csv = os.path.join(res_path, folder, prefix+'-'+method+'-'+dataset+'-rdf.csv')
#                     if os.path.exists(the_csv):
#                         print("Looking in:", the_csv)
#                         df = pd.read_csv(the_csv, index_col=0, keep_default_na=False)
                    if 'TEC' == method:
                        df = kModelResults(os.path.join(res_path, folder), prefix, 'Ensemble', dataset,
                                           model_folder='model_ensemble', 
                                           isformat=True)#isformat)
            
#                         display(df)
#                         df[method+'-'+dataset] = df.loc[:, df.columns[:-1]]
                        if to_csv:
                            df.to_csv('RDF/'+prefix+'-'+method+'-'+dataset+'-r2df'+('f' if isformat else '')+'.csv')
                    
                    elif method in ['BI-TEC', 'H-TEC', 'Hp-TEC', 'R-TEC', 'U-TEC']:
#                     elif 'BI-TEC' in method:
                        md = {'BI-TEC':'bi_ensemble', 'H-TEC':'ensemble_h', 'Hp-TEC':'ensemble_hp', 
                              'R-TEC':'ensemble_r', 'U-TEC':'ensemble_u'}
                        df = kModelResults(os.path.join(res_path, folder), prefix, 'Ensemble', dataset,
                                           model_folder=md[method], #'model_ensemble', 
                                           isformat=True)#isformat)         
#                         display(df)
                        df[method+'-'+dataset] = df['Ensemble-'+dataset]
#                         df[method+'-'+dataset] = df.loc[:, df.columns[:-1]]
#                         display(df)
                        
#                     elif 'MLP-TEC' in method:
#                         df = kModelResults(os.path.join(res_path, folder), prefix, method, dataset,
#                                            model_folder='mlp_ensemble', #'model_ensemble', 
#                                            isformat=True)#isformat)         
#                         display(df)

                    elif method in ['M2L', 'MML', 'HL', 'HpL', 'HLT50', 'HpLT50', 'U', 'RL']:
                        df = createdf(prefix)
                        for i in range(1, num_runs+1):
                            dfx = resultsk2df(os.path.join(res_path, folder, prefix, run, method+'-'+dataset,\
                                            'MLP-'+str(i), 'classification_times.csv'), isformat=isformat)
#                                 os.path.join(res_path, folder), prefix, method+'-'+dataset,
#                                          strsearch=os.path.join(run, method+'-'+dataset), 
#                                              modelfolder='MLP-'+str(i), isformat=False)
                            
#                             display(dfx)
                            if to_csv:
                                dfx.to_csv('RDF/'+prefix+'-'+method+'-'+dataset+'-r2df'+('f' if isformat else '')+'.csv')
                    
                            if len(dfx.columns) > 2:
                                df['MLP-'+str(i)] = (dfx.values[:, -1][4],0,0,0,dfx.values[:, -1][8],0,0,0)
#                             if method+'-'+dataset in dfx.columns:
#                                 df['MLP-'+str(i)] = (dfx[method+'-'+dataset][4],0,0,0,dfx[method+'-'+dataset][8],0,0,0)
# #                             else:
# #                                 df['MLP-'+str(i)] = (0,0,0,0,0,0,0,0)
#                         df[method+'-'+dataset] = df.loc[:, df.columns[2:]].mean(axis=1)
#                         display(df)
                        df[method+'-'+dataset] = df.loc[:, df.columns[2:]].mean(axis=1)
#                         from results import format_stats
                        if (not to_csv):
                            for column in df.columns[2:]:
                                df[column] = format_stats(df, column, list_stats)
                    elif 'POIS' in method:
                        df = createdf(prefix)
                        jstr = method.strip('POIS-') #.split('-')[0]
#                         print(jstr)
#                         import results as re
                        for i in range(1, num_runs+1):
#                             dfx = results2df(os.path.join(res_path, folder), prefix, method,
#                                          strsearch=os.path.join('run1', method, 
#                                          'POIS-'+jstr+'-'+str(i)), isformat=False)
                            dfx = resultsk2df(os.path.join(res_path, folder, prefix, run, 'NPOI*-specific', \
                                            'POIS-'+jstr+'-'+str(i), 'poifreq_results.txt'), isformat=isformat)
                            
#                             display(dfx)
                            if to_csv:
                                dfx.to_csv('RDF/'+prefix+'-'+method+'-'+dataset+'-r2df'+('f' if isformat else '')+'.csv')
                    
                            if len(dfx.columns) > 2:
                                df['NPOI-'+str(i)] = (0,dfx.values[:, -1][4],0,0,0,dfx.values[:, -1][8],0,0)
#                             else:
#                                 df['NPOI-'+str(i)] = (0,0,0,0,0,0,0,0)
#                         display(df)
                        df[method+'-'+dataset] = df.loc[:, df.columns[2:]].mean(axis=1)
#                         from results import format_stats
                        if (not to_csv):
                            for column in df.columns[2:]:
                                df[column] = format_stats(df, column, list_stats)
                    elif 'MARC' in method:
                        df = createdf(prefix)
                        for i in range(1, num_runs+1):
                            dfx = resultsk2df(os.path.join(res_path, folder, prefix, run, method+'-'+dataset,\
                                        method+'-'+dataset+'_'+str(i), '*_results.txt'), isformat=isformat)
                            
#                             display(dfx)
                            if to_csv:
                                dfx.to_csv('RDF/'+prefix+'-'+method+'-'+dataset+'-r2df'+('f' if isformat else '')+'.csv')
                            
                            if len(dfx.columns) > 2:
                                df['MARC-'+str(i)] = (0,0,dfx.values[:, -1][4],0,0,0,dfx.values[:, -1][7],0)
#                             else:
#                                 df['MARC-'+str(i)] = (0,0,0,0,0,0,0,0)
#                         display(df)
                        df[method+'-'+dataset] = df.loc[:, df.columns[2:]].mean(axis=1)
#                         from results import format_stats
                        if (not to_csv):
                            for column in df.columns[2:]:
                                df[column] = format_stats(df, column, list_stats)
                            
                            
#                     df['Dataset'] = ""
#                     df['Dataset'][0] = prefix
#                     df = kFoldResults(os.path.join(res_path, folder), prefix, method+'-'+dataset)
#                     display(df)
#                     return
                    if print_latex and to_csv:
                        printLatex(df, ajust=9)
                    print('% ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   --- ')
                    if not len(data.columns) > 0:
                        data = df[['Dataset', ' ']]
                    if method+'-'+dataset in df.columns:
                        data[method] = df[[method+'-'+dataset]]
            data.at[0,'Dataset'] = prefix + ' ('+dataset+')'
            table = pd.concat([table, data], axis=0)
        table = table.reset_index(drop=True)
#         display(table)
        
#         if print_latex:
#             printLatex(table, ajust=9)
        print('% ------------------------------------------------------------------------ ')
        return table

def kModelResults(search_str, prefix, method, dataset, model_folder='model_ensemble', isformat=True):
#     from ..main import importer
#     importer(['S', 'glob', 'format_stats'], globals())
#     import os
#     import pandas as pd
    import glob2 as glob
    from results import format_stats

    search_str = os.path.join(search_str, prefix, '**', method+'-'+dataset,\
                              model_folder, 'model_approachEnsemble_history.csv')
    
    filesList = []
    print("Looking for result files in " + search_str)
    for files in glob.glob(search_str):
        fileName, fileExtension = os.path.splitext(files)
        filesList.append(files) #filename with extension
    
    cols = []
    rows = []
    
#     import numpy as np
    df = createdf(prefix)

    for ijk in filesList:
        print("Looking in:", ijk)
        model = os.path.basename(os.path.dirname(ijk))[len(model_folder):]
        run = ijk[ijk.find('run'):ijk.find('run')+4]
        
        data = pd.read_csv(ijk, index_col=0)
        
        methods = {}
        
        data = data.set_index('classifier')
#         print(data)
        
        for stat in list_stats:
            aux = [i for i in stat[2].split() if i in data.index]
            if (stat[1] == 'ACC'):
                if aux:
                    methods[stat[0]] = data['accuracy'][aux[0]] * 100
                else:
                    methods[stat[0]] = 0
            else: # stat[1] == 'time'
                if aux:
                    methods[stat[0]] = float(data['time'][aux[0]]) if 'time' in data.columns else 0
                else:
                    methods[stat[0]] = 0

#         for cls in data.index:
#             methods["ACC "+cls.capitalize()] = data['accuracy'][cls] * 100

#         for cls in data.index:
#             methods["Time ("+cls.capitalize()+")"] = float(data['time'][cls]) if 'time' in data.columns else 0
            
        cols.append(run+model)
        df[run+model] = list(methods.values())
        rows = list(methods)
    
    if len(cols) > 0:
        df[method+'-'+dataset] = df.loc[:, cols].mean(axis=1)
    else:
        df[method+'-'+dataset] = 0
    
    cols = cols + [method+'-'+dataset]
    if isformat:
        for column in cols:
            df[column] = format_stats(df, column, list_stats)
        
    return df[['Dataset',' '] + cols]

# def resultsk2df(search_str, prefix, method, dataset, model_folder='MLP', isformat=True):
def resultsk2df(search_str, isformat=True):
#     from ..main import importer
#     importer(['S', 'glob', 'format_stats', 'get_stats'], globals())
#     import os
#     import pandas as pd
    import glob2 as glob
    from results import get_stats, format_stats, STATS

#     search_str = os.path.join(search_str, prefix, '**', method+'-'+dataset, \
#                               model_folder+'-1', 'classification_times.csv')
    
    filesList = []
    print("Looking for result files in " + search_str)
    for files in glob.glob(search_str):
        fileName, fileExtension = os.path.splitext(files)
        filesList.append(files) #filename with extension
    
#     print(filesList)
    
    cols = []
    rows = []
    
    list_stats=STATS(['*'])
    df = createdf('prefix', list_stats)

    method = os.path.basename(os.path.dirname(os.path.dirname(search_str))).split('-')[0]
    dataset = os.path.basename(os.path.dirname(os.path.dirname(search_str))).split('-')[1]
#     print(method, dataset)
    
    for ijk in filesList:
        print("Looking in:", ijk)
        model = os.path.basename(os.path.dirname(ijk))
        model_run = '-'+model.rsplit('-', 1)[1]

        run = ijk[ijk.find('run'):ijk.find('run')+4]
        path = os.path.abspath(os.path.join(os.path.dirname(ijk), '..'))
        
        method = os.path.basename(path).split('-')[0]
        dataset = os.path.basename(path).split('-')[1]
        txt = os.path.join(path, method+'-'+dataset+'.txt')
        
        if 'POIS' in ijk:
            method = os.path.basename(os.path.dirname(ijk)).split('-')[0]
            dataset = os.path.basename(os.path.dirname(ijk)).split('-')[1]
            
        if 'MARC' in ijk:
            txt = ijk
        
#         print(txt)
        
        cols.append(run+model_run)
        df[run+model_run] = get_stats(txt, path, method, list_stats, model)
    
    if len(cols) > 0:
        df[method+'-'+dataset] = df.loc[:, cols].mean(axis=1)
    else:
        df[method+'-'+dataset] = 0
    
    cols = cols + [method+'-'+dataset]
    if isformat:
        for column in cols:
            df[column] = format_stats(df, column, list_stats)
        
#     display(df)
    
    return df[['Dataset',' '] + cols]


methods_dic = {
        method: ['MML', 'M2L', 'HL', 'HpL', 'U', 'RL', 
                 'POIS-'+dim+'_1', 'POIS-'+dim+'_2', 'POIS-'+dim+'_3', 'POIS-'+dim+'_1_2_3', 
                 'MARC', 'TEC', 'BI-TEC', 'H-TEC', 'Hp-TEC', 'R-TEC', 'U-TEC'],
}
prefixes = [prefix] 
datasets = ds.split(',')

table = printECResults(results_path, methods_dic, prefixes, datasets, False, True, False)
display(table)

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(table)
