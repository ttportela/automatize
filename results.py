# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Feb, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
from .main import importer, display
importer(['S', 'glob', 'datetime', 're'], globals())

from .inc.script_def import getSubset
# --------------------------------------------------------------------------------

def STATS(name=['*']):
    list_stats = []
    
    # * - All
    # S - Simple combination / s - Simpler combination
    # M/m = Movelets and Candidates / AT/at - Acc and Time / C - covered trajectories-> sections
    # N - for NPOI-S methods
    
    if set(name) & set(['*', '#', 'M', 'S']): # Movelets & Candidates All
        list_stats = list_stats + [
            ['Candidates',           'sum',   'Number of Candidates'],
            ['Scored',               'sum',   'Scored Candidates'],
            ['Recovered',            'sum',   'Recovered Candidates'],
            ['Movelets',             'sum',   'Total of Movelets'],
        ]
    elif set(name) & set(['m', 's']): # Movelets & Candidates  Simple
        list_stats = list_stats + [
            ['Candidates',           'sum',   'Number of Candidates'],
            ['Movelets',             'sum',   'Total of Movelets'],
        ]
        
     # ACC & Time All
    if set(name) & set(['*', '#', 'AT', 'at', 'MLP', 'N', 'S', 's', 'AccTT']):
        list_stats = list_stats + [
            ['ACC (MLP)',            'ACC',     'MLP'],
        ]
    if set(name) & set(['*', 'AT', 'at', 'RF', 'S', 's']): 
        list_stats = list_stats + [
            ['ACC (RF)',             'ACC',     'RF'],
        ]
    if set(name) & set(['*', 'AT', 'SVM', 'S']): 
        list_stats = list_stats + [
            ['ACC (SVM)',            'ACC',     'SVM'],
        ]
    
    if set(name) & set(['*', '#', 'AT', 'at', 'N', 'S', 's', 'TIME']):
        list_stats = list_stats + [
            ['Time (Movelets)',      'time',    'Processing time'],
        ]
    if set(name) & set(['*', '#', 'AT', 'at', 'MLP', 'N', 'S', 's']):
        list_stats = list_stats + [
            ['Time (MLP)',           'accTime', 'MLP'],
        ]
    if set(name) & set(['*', '#', 'AT', 'at', 'RF', 'S', 's']):
        list_stats = list_stats + [
            ['Time (RF)',            'accTime', 'RF'],
        ]
    if set(name) & set(['*', '#', 'AT', 'SVM', 'S']):
        list_stats = list_stats + [
            ['Time (SVM)',           'accTime', 'SVM'],
        ]
    if set(name) & set(['*', '#', 'AccTT']):
        list_stats = list_stats + [
            ['Time',      'totalTime',    'Processing time|MLP'],
        ]
    
    if set(name) & set(['*', 'C', 'S']): # Hiper Covered Trajectories 
        list_stats = list_stats + [
            ['Trajs. Compared',      'sum',   'Trajs. Looked'],
            ['Trajs. Pruned',        'sum',   'Trajs. Ignored'],
        ]
        
    
    if set(name) & set(['*', 'F']): # Features Extra
        list_stats = list_stats + [
            ['Max # of Features',    'max',     'Used Features'],
            ['Min # of Features',    'min',     'Used Features'],
            ['Avg Features',         'mean',    'Used Features'],
            ['Max Size',             'max',     'Max Size'],
        ]
    elif set(name) & set(['S1F']): # Features from SUPERv1
        list_stats = list_stats + [
            ['Max # of Features',    'max',     'Max number of Features'],
            ['Min # of Features',    'min',     'Max number of Features'],
            ['Sum # of Features',    'sum',     'Max number of Features'],
            ['Avg Features',         'mean',    'Used Features'],
            ['Max # of Ranges',      'max',     'Number of Ranges'],
            ['Sum # of Ranges',      'sum',     'Number of Ranges'],
            ['Max Limit Size',       'max',     'Limit Size'],
            ['Max Size',             'max',     'Max Size'],
        ]
    
    if set(name) & set(['*', 'T']): # Trajectories Extra
        list_stats = list_stats + [
            ['Trajectories',         'count',   'Trajectory'],
            ['Max Traj Size',        'max',     'Trajectory Size'],
            ['Min Traj Size',        'min',     'Trajectory Size'],
            ['Avg Traj Size',        'mean',    'Trajectory Size'],
        ]
        
    if set(name) & set(['*', 'MSG', 'err', 'warn', 'TC']):
        if set(name) & set(['*', 'MSG', 'err']):
            list_stats = list_stats + [
                ['Messages', 'msg', 'msg'],
            ]
        if set(name) & set(['*', 'err']):
            list_stats = list_stats + [
                ['Error', 'msg', 'err'],
            ]
        if set(name) & set(['*', 'warn']):
            list_stats = list_stats + [
                ['Warning', 'msg', 'warn'],
            ]
        if set(name) & set(['*', 'TC']):
            list_stats = list_stats + [
                ['Finished', 'msg', 'TC'],
            ]
        
    if set(name) & set(['*', '#', 'D']):
        list_stats = list_stats + [
            ['Date', 'endDate', ''],
        ]

    return list_stats
# --------------------------------------------------------------------------------
def results2df(res_path, prefix, method, strsearch=None, list_stats=STATS(['S']), modelfolder='model', isformat=True):
#     from main import importer
    importer(['S', 'glob'], globals())
#     glob = importer(['glob'])['glob']
#     print(glob.glob('./*'))
    
#     filelist = []
    filesList = []

    # 1: Build up list of files:
    if strsearch:
        search = os.path.join(res_path, prefix, strsearch )
        search = os.path.join(search, '*.txt') if '.txt' not in search else search
    else:
        search = os.path.join(res_path, prefix, '**', method, method+'.txt' )
    print("Looking for result files in " + search)
    for files in glob.glob(search):
        fileName, fileExtension = os.path.splitext(files)
#         filelist.append(fileName) #filename without extension
        filesList.append(files) #filename with extension
    
    # 2: Create and concatenate in a DF:
#     ct = 1
    df = pd.DataFrame()
    
    cols = []
    rows = []
    for x in list_stats:
        rows.append(x[0])

    df[' '] = rows
    df['Dataset'] = ""
    df['Dataset'][0] = prefix
            
    for ijk in filesList:
        method = os.path.basename(ijk)[:-4]
        path = os.path.dirname(ijk)
#         run = os.path.basename(os.path.abspath(os.path.join(path, '..')))
        
        cols.append(method)
        df[method] = get_stats(ijk, path, method, list_stats, modelfolder)
        
        if containErrors(ijk) or containWarnings(ijk):
            print('*** Warning: '+method+' may contain errors ***')
        
    # ---
#     df[method] = df.loc[:, cols[:-1]].mean(axis=1)
#     df[method].loc[df.index[-1]] = -1
    
    if isformat:
#         for column in cols:
        for i in range(2, len(df.columns)):
            df[df.columns[i]] = format_stats(df, df.columns[i], list_stats)

#         df[method] = format_stats(df, method, list_stats)
        
    cols = ['Dataset',' '] + cols
    return df[cols]

def resultsk2df(res_path, prefix, method, list_stats=STATS(['S']), modelfolder='model', isformat=True):
#     from main import importer
    importer(['S', 'glob'], globals())
    
#     filelist = []
    filesList = []
    
    if 'MARC' in method:
        search = os.path.join(res_path, prefix, 'run*', method, '**', method+'*.txt')
    else:
        search = os.path.join(res_path, prefix, 'run*', method, method+'.txt' )

    # 1: Build up list of files:
    print("Looking for result files in " + search)
    for files in glob.glob(search):
        fileName, fileExtension = os.path.splitext(files)
#         filelist.append(fileName) #filename without extension
        filesList.append(files) #filename with extension
    
    # 2: Create and concatenate in a DF:
#     ct = 1
    df = pd.DataFrame()
    
    cols = []
    rows = []
    for x in list_stats:
        rows.append(x[0])

    df[' '] = rows
    df['Dataset'] = ""
    df['Dataset'][0] = prefix
            
    for ijk in filesList:
        path = os.path.dirname(ijk)
        run = path[path.find('run'):path.find('run')+4] #os.path.basename(os.path.abspath(os.path.join(path, '..')))
        
        cols.append(run)
        df[run] = get_stats(ijk, path, method, list_stats, modelfolder)
        
        if containErrors(ijk) or containWarnings(ijk):
            print('*** Warning: '+method+' contains errors ***')
        
    # ---
    df[method] = df.loc[:, cols].mean(axis=1)
    # TEMP: todo something better:
    if list_stats[-1][1] == 'enddate':
        df[method].loc[df.index[-1]] = -1
    
    display(df)
    
    if isformat:
        for column in cols:
            df[column] = format_stats(df, column, list_stats)

        df[method] = format_stats(df, method, list_stats)
        
    cols = ['Dataset',' '] + cols + [method]
    return df[cols]

# --------------------------------------------------------------------------------------
def results2tex(res_path, methods_dic, prefixes, datasets, list_stats=STATS(['S']), modelfolder='model', to_csv=False, isformat=True, print_latex=True, clines=[]):
#     from main import importer, display
    importer(['S', 'printLatex'], globals())
    
#     import os
#     import pandas as pd
#     from IPython.display import display
#     from PACKAGE_NAME.results import printLatex
    
    for prefix in prefixes:
        table = pd.DataFrame()
        for dataset in datasets:
            data = pd.DataFrame()
            for folder, methods in methods_dic.items():
                for method in methods:
                    df = resultsk2df(os.path.join(res_path, folder), prefix, method+'-'+dataset, 
                                     list_stats, modelfolder, isformat)
                    
                    if method+'-'+dataset in df.columns:
                        df = read_rdf(df, os.path.join(res_path, folder, prefix+'-'+method+'-'+dataset), list_stats)
                    
                    display(df)
#                     printLatex(df, ajust=9)
                    if to_csv:
                        df.to_csv(prefix+'-'+method+'-'+dataset+('-ft-' if isformat else '-')+'r2df.csv')
                    if print_latex and to_csv:
                        printLatex(df, ajust=9, clines=clines)
                    print('% ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   --- ')
                    if not len(data.columns) > 0:
                        data = df[['Dataset', ' ']]
                    if method+'-'+dataset in df.columns:
                        data[method] = df[[method+'-'+dataset]]
                    else:
                        data[method] = ''
                    
            data.at[0,'Dataset'] = prefix + ' ('+dataset+')'
            table = pd.concat([table, data], axis=0)
        table = table.reset_index(drop=True)
        display(table)
        if print_latex:
            printLatex(table, ajust=9, clines=clines)
        print('% ------------------------------------------------------------------------ ')

def read_rdf(df, file, list_stats):
#     from main import importer
#     importer(['S'], locals())
    
    if os.path.exists(file+'-rdf.csv'):
        print("Loading: ", file+'-rdf.csv')
        rdf = pd.read_csv(file+'-rdf.csv', index_col=0)
#         display(rdf)
        cols = rdf.columns[2:]
        rows = ['Candidates', 'Movelets', 'ACC (MLP)', 'ACC (RF)', 'ACC (SVM)', 'Time (Movelets)', 'Time (MLP)', 'Time (RF)', 'Time (SVM)', 'Trajs. Compared', 'Trajs. Pruned']
        for col in cols:
            vals = []
            for x in list_stats:
                if x[0] in rows:
                    e = rdf.loc[rows.index(x[0]), col] 
                else:
                    e = '-'
                vals.append(e)
            df[col] = vals

    elif os.path.exists(file+'-r2df.csv'): # TODO: format
        print("Loading: ", file+'-r2df.csv')
        rows = []
        for x in list_stats:
            rows.append(x[0])
        
        rdf = pd.read_csv(file+'-r2df.csv', index_col=0)
        rdf = rdf[rdf[' '] in rows]
        for column in rdf.columns[2:]:
            df[column] = format_stats(rdf, column, list_stats)

    return df
        
# --------------------------------------------------------------------------------------
def results2csv(res_path, methods_dic, prefixes, datasets):
#     from main import importer, display
#     importer(['S'], locals())
    for prefix in prefixes:
        table = pd.DataFrame()
        for dataset in datasets:
            data = pd.DataFrame()
            for folder, methods in methods_dic.items():
                for method in methods:
                    the_csv = os.path.join(res_path, folder, prefix+'-'+method+'-'+dataset+'-rdf.csv')
                    df = resultsk2df(os.path.join(res_path, folder), prefix, method+'-'+dataset, \
                                      isformat=False)
                    display(df)
                    
                    # TO CSV:
                    csvfile = prefix+'-'+method+'-'+dataset+'-r2df.csv'
                    print('Saving ... ', csvfile)
                    df.to_csv(csvfile)
        print('Done.')

# ------------------------------------------------------------
def containErrors(file):
    txt = open(file, 'r').read()
    return txt.find('java.') > -1 or txt.find('heap') > -1 or txt.find('error') > -1
def containWarnings(file):
    txt = open(file, 'r').read()
    return txt.find('Empty movelets set') > -1
def containTimeout(file):
    txt = open(file, 'r').read()
    return txt.find('[Warning] Time contract limit timeout.') > -1

def resultsDiff(df, ref_cols=[2], list_stats=STATS(['S']), isformat=True, istodisplay=True):
    n = len(df.columns)
    for ref in ref_cols:
        for col in range(2, n):
            if col not in ref_cols:
                a = df.iloc[:,ref]
                b = df.iloc[:,col]
                df[str(ref)+'-'+str(col)] = ((b-a) / b * 100.0)
    
#     from PACKAGE_NAME.results import format_stats
    if isformat:
        for column in df.columns[2:n]:
            df[column] = format_stats(df, column, list_stats)
        for column in df.columns[n:]:
            df[column] = df[column].map(lambda x: '{:.2f}%'.format(x))

    if istodisplay:
        display(df)
        
    return df

# ----------------------------------------------------------------------------------
def read_approach(path, approach_file, modelfolder='model'):
    res_file = os.path.join(path, modelfolder, approach_file)
    if os.path.isfile(res_file):
        data = pd.read_csv(res_file)
        return data
    else:
        return None

def getACC_RF(path, modelfolder='model'):
    acc = 0
    data = read_approach(path, 'model_approachRF300_history.csv', modelfolder)
    if data is not None:
        acc = data['1'].iloc[-1]
    return acc

def getACC_SVM(path, modelfolder='model'):
    acc = 0
    data = read_approach(path, 'model_approachSVC_history.csv', modelfolder)
    if data is not None:
        acc = data.loc[0].iloc[-1]
    return acc

def getACC_MLP(path, method, modelfolder='model'):
    acc = 0
    data = read_approach(path, 'model_approach2_history_Step5.csv', modelfolder)
    if data is not None:
        acc = data['val_accuracy'].iloc[-1]
    return acc

def getACC_NN(resfile, path, method, modelfolder='model'):
    acc = 0
    if isMethod(resfile, 'MARC'):
        res_file = getMARCFile(path, method, modelfolder)
        if res_file:
            data = pd.read_csv(res_file)
            acc = data['test_acc'].iloc[-1]
    elif isMethod(resfile, 'POIF'):
        data = read_csv(resfile)
        acc = get_first_number("Acc: ", data)
    elif isMethod(resfile, 'TEC'):
        data = pd.read_csv(resfile, index_col=0)
        data = data.set_index('classifier')
        acc = data['accuracy']['EnsembleClassifier']
    else:
        acc = getACC_MLP(path, method, modelfolder)

    return acc

def getACC_time(resfile, path, label, modelfolder='model'):
    acc = 0.0
    if isMethod(resfile, 'POIF') and label in ['MLP', 'NN']:
        if resfile:
            data = read_csv(resfile)
            acc = get_first_number("Classification Time: ", data)
    elif isMethod(resfile, 'TEC') and label in ['MLP', 'NN']:
        data = pd.read_csv(resfile, index_col=0)
        data = data.set_index('classifier')
        acc = float(data['time']['EnsembleClassifier']) if 'time' in data.columns else 0
    else:
        data = read_approach(path, 'classification_times.csv', modelfolder)
        if data is not None:
            acc = data[label][0]
    return acc

def getRuntime(resfile, path, label, data, modelfolder='model'):
    time = 0
    if isMethod(resfile, 'TEC'):
        time = getACC_time(resfile, path, 'NN', modelfolder) 
    elif isMethod(resfile, 'POIF'):
        datax = getLogFile(path)
        if datax:
            time = get_last_number_of_ms(label, read_csv(datax))
    else:
        time = get_last_number_of_ms(label, data) 

    return time 

def getMARCFile(path, method, modelfolder):
    res_file = os.path.join(path, modelfolder + '_results.csv')
    if not os.path.isfile(res_file):
        res_file = os.path.join(path, modelfolder,  modelfolder + '_results.csv')
    if not os.path.isfile(res_file):
        res_file = glob.glob(os.path.join(path, '**', method+'*' + '_results.csv'), recursive=True)
        if len(res_file) > 0 and os.path.isfile(res_file[0]):
            res_file = res_file[0]
        else:
            return False
    return res_file

def getLogFile(path):
    res_file = glob.glob(os.path.join(path, os.path.basename(path) + '.txt'))
    if len(res_file) > 0 and os.path.isfile(res_file[0]):
        return res_file[0]
    else:
        return False
# ----------------------------------------------------------------------------------
def get_stats(resfile, path, method, list_stats, modelfolder='model', show_warnings=True):
#     from main import importer
#     importer(['S'], locals())

    stats = []
    data = read_csv(resfile)
    
    for x in list_stats:
        ssearch = x[2]+": "
        if x[1] == 'max':
            stats.append( get_max_number_of_file_by_dataframe(ssearch, data) )
        
        elif x[1] == 'min':
            stats.append( get_min_number_of_file_by_dataframe(ssearch, data) )
        
        elif x[1] == 'sum':
            stats.append( get_sum_of_file_by_dataframe(ssearch, data) )
        
        elif x[1] == 'count':
            stats.append( get_count_of_file_by_dataframe(ssearch, data) )
        
        elif x[1] == 'mean':
            a = get_sum_of_file_by_dataframe(ssearch, data)
            b = get_count_of_file_by_dataframe(ssearch, data)
            if b > 0:
                stats.append( a / b )
            else:
                stats.append( 0 )
        
        elif x[1] == 'first':
            stats.append( get_first_number(ssearch, data) )
        
#         elif x[1] == 'last':
#             stats.append( get_last_number_of_ms(ssearch, data) )
        
        elif x[1] == 'time':
#             time = 0
#             if isMethod(resfile, 'TEC'):
#                 time = getACC_time(resfile, path, 'NN', modelfolder) 
#             elif isMethod(resfile, 'POIF'):
#                 datax = getLogFile(path, method)
#                 if datax:
#                     time = get_last_number_of_ms(ssearch, read_csv(datax))
#             else:
#                 time = get_last_number_of_ms(ssearch, data) 
            
            stats.append( getRuntime(resfile, path, ssearch, data, modelfolder) )
        
        elif x[1] == 'accTime':
            if isMethod(resfile, 'MARC') and x[2] in ['MLP', 'NN']:
                stats.append( getRuntime(resfile, path, 'Processing time: ', data, modelfolder) )
                #get_last_number_of_ms('Processing time: ', data) )
            else:
                stats.append( getACC_time(resfile, path, x[2], modelfolder) )
        
        elif x[1] == 'totalTime':
            timeRun = getRuntime(resfile, path, x[2].split('|')[0]+": ", data, modelfolder)
            #get_last_number_of_ms(x[2].split('|')[0]+": ", data) 
            timeAcc = getACC_time(resfile, path, x[2].split('|')[1], modelfolder) 
            if isMethod(resfile, 'TEC') or isMethod(resfile, 'MARC'):
                stats.append(timeRun)
            else:
                if show_warnings and (timeRun <= 0 or timeAcc <= 0):
#                     and not (isMethod(resfile, 'POIF') or isMethod(resfile, 'TEC') or isMethod(resfile, 'MARC')):
                    print('*** Warning ***', 'timeRun:', timeRun, 'timeAcc:', timeAcc, 'for '+resfile+'.')
                    timeRun = -timeRun
                    timeAcc = -timeAcc
                
                stats.append(timeRun + timeAcc)
        
        elif x[1] == 'ACC':
            if x[2] in ['MLP', 'NN']:
                acc = getACC_NN(resfile, path, method, modelfolder)
            elif x[2] == 'RF':
                acc = getACC_RF(path, modelfolder)
            elif x[2] == 'SVM':
                acc = getACC_SVM(path, modelfolder)
            
            stats.append( acc * 100 )
        
        elif x[1] == 'msg':
            e = False
            if x[2] == 'msg':
                e = runningProblems(resfile)
            if x[2] == 'isMsg':
                e = runningProblems(resfile) != False
            elif x[2] == 'err':
                e = containErrors(resfile)
            elif x[2] == 'warn':
                e = containWarnings(resfile)
            elif x[2] == 'TC':
                e = containTimeout(resfile)
            
            stats.append(e)
            
        elif x[1] == 'endDate':
            try:
                importer(['dateparser'], globals())
    
                dtstr = data.iloc[-1]['content']
                stats.append( dateparser.parse(dtstr).timestamp() )
            except ValueError:
                stats.append( -1 )
                
    return stats


# ------------------------------------------------------------
def format_stats(df, method, list_stats):
    line = []
    
    for i in range(0, len(list_stats)):
        x = list_stats[i]
        if x[1] in ['max', 'min', 'sum', 'count', 'first']:
            line.append(format_cel(df, method, i, '{val:,}'))

        elif x[1] in ['ACC', 'mean']:
            line.append(format_celf(df, method, i, '{val:.3f}'))

        elif x[1] in ['time', 'accTime', 'totalTime']:
            line.append(format_celh(df, method, i, '%dh%02dm%02ds'))
        
        elif x[1] == 'endDate':
            line.append(format_date(df.at[i,method]))
            
        else:
            line.append(str(df.at[i,method]))
            
    return list(map(str, line))
    
def format_cel(df, method, row, pattern):
    value = int(df.at[row,method])
    return format_float(value, pattern)
    
def format_celf(df, method, row, pattern):
    value = float(df.at[row,method])
    value = format_float(value, pattern)
    return value
    
def format_celh(df, method, row, pattern):
    return format_hour(df.at[row,method])

def format_float(value, pattern='{val:.3f}'):
    if value > 0:
        return pattern.format(val=value)
    else: 
        return "-"

def format_date(ts):
    importer(['datetime'], globals())
    
    try:
        return datetime.fromtimestamp(ts).strftime("%d/%m/%y-%H:%M:%S") if ts > -1 else '-'
    except TypeError:
        return ts

def format_hour(millis):
    appnd = '*' if millis < 0 else ''
    millis = abs(millis)
    if millis > 0:
        hours, rem = divmod(millis, (1000*60*60))
        minutes, rem = divmod(rem, (1000*60))
        seconds, rem = divmod(rem, 1000)
        value = ''
        if hours > 0:
            value = value + ('%dh' % hours)
        if minutes > 0:
            value = value + (('%02dm' % minutes) if value != '' else ('%dm' % minutes))
        if seconds > 0:
            value = value + (('%02ds' % seconds) if value != '' else ('%ds' % seconds))
        if value == '':
            value = value + (('%02.3fs' % (rem/1000)) if value != '' else ('%.3fs' % (rem/1000)))
        return value + appnd
    else: 
        return "-"

def split_runtime(millis):
    millis = int(millis)
    seconds=(millis/1000)%60
    seconds = int(seconds)
    minutes=(millis/(1000*60))%60
    minutes = int(minutes)
    hours=(millis//(1000*60*60))

    return (hours, minutes, seconds)

# ----------------------------------------------------------------------------------
def summaryRuns(df, method, run_cols, list_stats):
    stats = df.loc[:, run_cols].mean(axis=1)
    
    for i in range(len(list_stats)):
        if list_stats[i][1] == 'endDate':
            val = -1
            for rc in run_cols:
                val = max(val, df.at[i, rc])
            stats[i] = val
        elif list_stats[i][1] == 'msg':
            val = False
            for rc in run_cols:
                val = True if val or df.at[i, rc] else False
            stats[i] = val
    
    return stats

def runningProblems(ijk):
    e1 = containErrors(ijk)
    e2 = containWarnings(ijk)
    e3 = containTimeout(ijk)
    s = False
    if e1 or e2 or e3:
        s = ('[ER]' if e1 else '[--]')+('[WN]' if e2 else '[--]')+('[TC]' if e3 else '[--]')
    return s

# --------------------------------------------------------------------------------->
def getResultFiles(res_path):
    def findFiles(x):
        search = os.path.join(res_path, '**', x)
        fl = []
        for files in glob.glob(search, recursive=True):
            fileName, fileExtension = os.path.splitext(files)
            fl.append(files) #filename with extension
        return fl
       
    filesList = []
    filesList = filesList + findFiles('classification_times.csv') # NN / RF / SVM
    filesList = filesList + findFiles('poifreq_results.txt') # POI-F / POI-FS
    filesList = filesList + findFiles('MARC-*.txt') # MARC
    filesList = filesList + findFiles('model_approachEnsemble_history.csv') # TEC
    
    return filesList

def decodeURL(ijk):
    rpos = ijk.find('run')
    path = ijk[:ijk.find(os.path.sep, rpos+5)]
    
    method = path[path.rfind(os.path.sep)+1:]
    subset = method.split('-')[-1]
    method = method.split('-')[0]
        
    run = path[rpos:rpos+4]
    run = (run)[3:]

    prefix = os.path.basename(path[:rpos-1])

    model = os.path.dirname(ijk)
    model = model[model.rfind(os.path.sep)+1:]
    
    if isMethod(ijk, 'POIF'):
        subsubset = model.split('-')[1] 
        method = method+ '_' + subsubset[re.search(r"_\d", subsubset).start()+1:]
    else:
        subsubset = subset
    
    if isMethod(ijk, 'TEC'):
        method += '_' + model.split('_')[-1]

    if isMethod(ijk, 'POIF'):
        random = '1' if model.count('-') <= 1 else model.split('-')[-1]
    else:
        random = '1' if '-' not in model else model.split('-')[-1]
    
    if not random.isdigit():
        random = 1
    
    if not (isMethod(ijk, 'POIF') or isMethod(ijk, 'TEC') or isMethod(ijk, 'MARC')):
        files = glob.glob(os.path.join(path, method+'*.txt'), recursive=True)
        ijk = files[0] if len(files) > 0 else ijk
        
    return run, random, method, subset, subsubset, prefix, model, path, ijk

def organizeResults(filesList, subsets=None):
    results = {}
    for ijk in filesList:
        run, random, method, subset, subsubset, prefix, model, path, file = decodeURL(ijk)
 
        is_POIF = isMethod(ijk, 'POIF')
        is_TEC  = isMethod(ijk, 'TEC')
        
        if subsets and subset not in subsets:
            continue
            
        dataset = prefix +'-'+ subset 
        mname   = method +'-'+ subset 
        
        if dataset not in results.keys():
            results[dataset] = {}
        if mname not in results[dataset].keys():
            results[dataset][mname] = []
        results[dataset][mname].append([run, random, method, subset, subsubset, prefix, model, path, file])

    return results

def history(res_path): #, prefix, method, list_stats=STATS(['S']), modelfolder='model', isformat=True):
    
    importer(['S', 'glob', 'STATS', 'get_stats', 'containErrors', 'np'], globals())
    histres = pd.DataFrame(columns=['#','timestamp','dataset','subset','subsubset','run','random','method', 'classifier','accuracy','runtime','cls_runtime','error','file'])

    filesList = getResultFiles(res_path)
    
    list_stats = STATS(['#'])
    list_stats_ind = [p[x] for p in list_stats for x in range(len(p))]
            
    for ijk in filesList:
        run, random, method, subset, subsubset, prefix, model, path, file = decodeURL(ijk)
        
        stats = get_stats(file, path, method+'-'+subset, list_stats, modelfolder=model)
        def gstati(x):
            return stats[list_stats_ind.index(x) // 3]
        def gstat(x):
            return stats[list_stats.index(x)]
        def getrow(run, random, method, subset, subsubset, prefix, model, path, file, result):
            
            return {
                '#': 0,
                'timestamp': gstati('endDate'),
                'dataset': prefix,
                'subset': subset,
                'run': run,
                'subsubset': subsubset,
                'random': random,
                'method': method,
                'classifier': result[0],
                'accuracy': result[1],
                'runtime': result[2] if isMethod(ijk, 'TEC') else gstati('time'),
                'cls_runtime': result[2],
                'error': containErrors(file) or containWarnings(file),
                'file': ijk,
            }
        
        if isMethod(ijk, 'TEC'):
            data = pd.read_csv(file, index_col=0)
            data = data.set_index('classifier')
            for index, row in data[:-1].iterrows():
                classifier = ['#'+index, row['accuracy'] * 100, row['time']]
                aux_hist = getrow(run, random, method, subset, subsubset, prefix, model, path, file, classifier)
                histres = pd.concat([histres, pd.DataFrame([aux_hist])])
                
            classifier = [model.split('-')[0].split('_')[-1], data['accuracy'][-1] * 100, data['time'][-1]]
            aux_hist = getrow(run, random, method, subset, subsubset, prefix, model, path, file, classifier)
            histres = pd.concat([histres, pd.DataFrame([aux_hist])])
        else:
#             lscls = ['MLP'] if 'MARC' in ijk else ['MLP', 'RF'] # ['MLP', 'RF', 'SVM']
#             for x in ['MLP', 'RF', 'SVM']:
            x = 'MLP'
            acc = gstat(['ACC ('+x+')', 'ACC', x])
            if acc > 0:
                classifier = [x, acc] + [ gstat(['Time ('+x+')', 'accTime', x],) ]
                aux_hist = getrow(run, random, method, subset, subsubset, prefix, model, path, file, classifier)
                histres = pd.concat([histres, pd.DataFrame([aux_hist])])
        
    # ---
    # Post treatment:
    histres['total_time']   = histres['runtime'] + histres['cls_runtime']
    histres['name']   = histres['method'].map(str) + '-' + histres['subsubset'].map(str) + '-' + histres['classifier'].map(str)
    histres['key'] = histres['dataset'].map(str) + '-' + histres['subset'].map(str) + '-' + histres['run'].map(str)
    
    # Ordering / Renaming:
    histres.reset_index(drop=True, inplace=True)

    return histres

def check_run(res_path, show_warnings=False):
    
    importer(['S', 'glob', 'STATS', 'get_stats', 'check_run', 'format_hour', 'np'], globals())
    
    filesList = getResultFiles(res_path)
    
    def adj(s, size=15):
        return s.ljust(size, ' ')
    
    SEP = ' ' #'\t'
    filesList.sort()
    for ijk in filesList:
        if 'POI-' in ijk and not ijk.endswith('poifreq_results.txt'): # DEPRECATED
            continue
        run, random, method, subset, subsubset, prefix, model, path, file = decodeURL(ijk)
        e = runningProblems(file)
        
        res = get_stats(file, path, method, STATS(['AccTT']), model, show_warnings=show_warnings)
        line = '[' + adj(format_float(res[0]),6) +']['+ format_hour(res[1])+']'
        
        if e:
            print('[*] NOT OK:'+SEP, adj(method, 20), SEP, adj(prefix), SEP, adj(run,3), SEP, adj(subset), SEP, e, line)
        else:
            print('        OK:'+SEP, adj(method, 20), SEP, adj(prefix), SEP, adj(run,3), SEP, adj(subset), SEP, line)


def compileResults(res_path, subsets=['specific'], list_stats=STATS(['S']), isformat=True, k=False):
    importer(['S', 'glob'], globals())
    
    results = organizeResults(getResultFiles(res_path), subsets)

    cols = []
    
    table = pd.DataFrame()
    for dataset in results.keys():
        data = pd.DataFrame()
        for mname in results[dataset].keys():
            cols.append(mname)
            # 1: Create and concatenate in a DF:
            df = pd.DataFrame()
            run_cols = []
            rows = []
            for x in list_stats:
                rows.append(x[0])
            df[' '] = rows
            df['Dataset'] = ""
            df['Dataset'][0] = dataset.split('-')[0] + ' ('+dataset.split('-')[1]+')'

            # ---
            partial_result = False
            for run, random, method, subset, subsubset, prefix, model, path, file in results[dataset][mname]:
                run_cols.append(run)
                df[run] = get_stats(file, path, method, list_stats, model)

                e = runningProblems(file)
                if e:
                    partial_result = True
                    print('*** Warning: '+mname+'-'+run+' contains errors > ' + e)
            # ---
            if k and len(run_cols) != k:
                partial_result = True
            df[method] = summaryRuns(df, method, run_cols, list_stats)
            
            print('Adding:', dataset, '\t', len(run_cols), 'run(s)', '\t', mname)
            if isformat:
                for column in run_cols:
                    df[column] = format_stats(df, column, list_stats)

                df[method] = format_stats(df, method, list_stats)
                
                if partial_result and not ('MARC' in method or 'POI' in method or 'TEC' in method):
                    df[method] = df[method].add('*')
    
            # ---
            if not len(data.columns) > 0:
                data = df[['Dataset', ' ']].copy()
            if method in df.columns:
                data[method] = df[[method]]
            else:
                data[method] = ''
        # ---
        table = pd.concat([table, data], axis=0)
    # ---
    table.fillna(value='', inplace=True)
    table = table.reset_index(drop=True)
    return table

def isMethod(file, key):
    if key == 'POIF' and file.endswith('poifreq_results.txt'):
        return True
    elif key == 'MARC' and 'MARC' in file:
        return True
    elif key == 'TEC' and file.endswith('model_approachEnsemble_history.csv'):
        return True
    else:
        return False
    
# --------------------------------------------------------------------------------->  
def toLatex(df, cols=None, ajust=9, clines=[]):
    if not cols:
        cols = df.columns
        
    def countCols(cols):
        ct = 0
        for c in cols:
            if type(c) == list:
                ct += len(c)
            else:
                ct += 1
        return ct
    
    def getLine(df, cols, l, ajust=12):
        def searchColumn(df, col):
            if '*' in col:
                import fnmatch
                filtered = fnmatch.filter(df.columns, col)
                for c1 in filtered:
                    if df.at[l,c1] != '-' and df.at[l,c1]:
                        return c1
                return filtered[0] if len(filtered) > 0 else col
            return col
        
        line = '&'+ str(df.at[l,df.columns[1]]).rjust(15, ' ') + ' '
        for g in cols[2:]:
            if type(g) == list:
                g1 = g
            else:
                g1 = [g]
            for c in g1:
                c1 = searchColumn(df, c)   
                value = df.at[l,c1] if c1 in df.columns else '-'
                line = line + '& '+ str(value).rjust(ajust, ' ') + ' '
        line = line + '\\\\'
        return line

    n_cols = (countCols(cols)-2)
    n_ds = len(df['Dataset'].unique()) -1
    n_rows = int(int(len(df)) / n_ds)
    
    df.fillna('-', inplace=True)
    
    print('\\begin{table*}[!ht]')
    print('\\centering')
    print('\\resizebox{\columnwidth}{!}{')
    print('\\begin{tabular}{|c|r||'+('r|'*n_cols)+'}')
    print('\\hline')
    
    if n_cols > len(cols): # We have column groups
        colg = []
        for i in range(len(cols)):
            if type(cols[i]) == list:
                colg.append('\\multicolumn{'+str(len(cols[i]))+'}{r|}{GROUP' + str(i)+'}')
            else:
                colg.append(' ')
        print((' & '.join(colg)) + ' \\\\')
        
    colg = []
    for i in range(len(cols)):
        if type(cols[i]) == list:
            colg = colg + [x.replace('_', '-') for x in cols[i]]
        else:
            colg.append(cols[i].replace('_', '-'))
    print((' & '.join(colg)) + ' \\\\')
    
    for k in range(0, int(len(df)), n_rows):
        print('\n\\hline')
        print('\\hline')
        print('\\multirow{'+str(n_rows)+'}{2cm}{'+df.at[k,'Dataset'].replace('_', ' ')+'}')
        for j in range(0, n_rows):
            print(getLine(df, cols, k+j, ajust))
            if j in clines:
                print('\\cline{2-'+str(n_cols+2)+'}')
    
    print('\\hline')
    print('\\end{tabular}}')
    print('\\caption{Results for '+df['Dataset'][0]+' dataset.}')
    print('\\label{tab:results_'+df['Dataset'][0]+'}')
    print('\\end{table*}')
    
def printLatex(df, ajust=9, clines=[]):
    n_cols = (len(df.columns)-2)
    n_ds = len(df['Dataset'].unique()) -1
    n_rows = int(int(len(df)) / n_ds)
    
    print('\\begin{table*}[!ht]')
    print('\\centering')
    print('\\resizebox{\columnwidth}{!}{')
    print('\\begin{tabular}{|c|r||'+('r|'*n_cols)+'}')
    print('\\hline')
#     print('\\hline')
    print((' & '.join(df.columns)) + ' \\\\')
    
    for k in range(0, int(len(df)), n_rows):
        print('\n\\hline')
        print('\\hline')
        print('\\multirow{'+str(n_rows)+'}{2cm}{'+df.at[k,'Dataset']+'}')
        for j in range(0, n_rows):
            print(printLatex_line(df, k+j, ajust))
            if j in clines:
                print('\\cline{2-'+str(n_cols+2)+'}')
    
#     print('\\hline')
    print('\\hline')
    print('\\end{tabular}}')
    print('\\caption{Results for '+df['Dataset'][0]+' dataset.}')
    print('\\label{tab:results_'+df['Dataset'][0]+'}')
    print('\\end{table*}')
    
def printLatex_line(df, l, ajust=12):
    line = '&'+ str(df.at[l,df.columns[1]]).rjust(15, ' ') + ' '
    for i in range(2, len(df.columns)):
        line = line + '& '+ str(df.at[l,df.columns[i]]).rjust(ajust, ' ') + ' '
    line = line + '\\\\'
    return line

# --------------------------------------------------------------------------------->   
def read_csv(file_name):
#     from main import importer
#     importer(['S'], locals())

    # Check Py Version:
    from inspect import signature
    if ('on_bad_lines' in signature(pd.read_csv).parameters):
        data = pd.read_csv(file_name, header = None, delimiter='-=-', engine='python', on_bad_lines='skip')
    else:
        data = pd.read_csv(file_name, header = None, delimiter='-=-', engine='python', error_bad_lines=False, warn_bad_lines=False)
    data.columns = ['content']
    return data

def get_lines_with_separator(data, str_splitter):
    lines_with_separation = []
    for index,row in data.iterrows():#
        if str_splitter in row['content']:
#             print(row)
            lines_with_separation.insert(len(lines_with_separation), index)
    return lines_with_separation

def get_titles(data):
    titles = []
    for index,row in data.iterrows():#
        if "Loading train and test data from" in row['content']:
            titles.insert(len(titles), row['content'])
    return titles

def split_df_to_dict(data, lines_with_separation):
    df_dict = {}
    lines_with_separation.pop(0)
    previous_line = 0
    for line in lines_with_separation:#
#         print(data.iloc[previous_line:line,:])
        df_dict[previous_line] = data.iloc[previous_line:line,:]
        previous_line=line
    df_dict['last'] = data.iloc[previous_line:,:]
    return df_dict

def get_total_number_of_candidates_file(str_target, df_dict):
    total_per_file = []
    for key in df_dict:
        total = 0
        for index,row in df_dict[key].iterrows():
            if str_target in row['content']:
                number = row['content'].split(str_target)[1]
                total = total + int(number)
        total_per_file.insert(len(total_per_file), total)
    return total_per_file

def get_total_number_of_candidates_file_by_dataframe(str_target, df):
    total = 0
    for index,row in df.iterrows():
        if str_target in row['content']:
            number = row['content'].split(str_target)[1]
            number = number.split(".")[0]
            total = total + int(number)
    return total

def get_sum_of_file_by_dataframe(str_target, df):
    total = 0
    for index,row in df.iterrows():
        if str_target in row['content']:
            number = row['content'].split(str_target)[1]
            number = number.split(".")[0]
            total = total + int(number)
    return total

def get_max_number_of_file_by_dataframe(str_target, df):
    total = 0
    for index,row in df.iterrows():
        if str_target in row['content']:
            number = row['content'].split(str_target)[1]
            number = int(number.split(".")[0])
            total = max(total, number)
    return total

def get_min_number_of_file_by_dataframe(str_target, df):
    total = 99999
    for index,row in df.iterrows():
        if str_target in row['content']:
            number = row['content'].split(str_target)[1]
            number = int(number.split(".")[0])
            total = min(total, number)
    return total

def get_total_number_of_ms(str_target, df):
    total = 0
    for index,row in df.iterrows():
        if str_target in row['content']:
            number = row['content'].split(str_target)[1]
            number = number.split(" milliseconds")[0]
            total = total + float(number)
    return total

def get_last_number_of_ms(str_target, df):
    total = 0
    for index,row in df.iterrows():
        if str_target in row['content']:
            number = row['content'].split(str_target)[1]
            number = number.split(" milliseconds")[0]
            total = float(number)
    return total

def get_first_number(str_target, df):
    total = 0
    for index,row in df.iterrows():
        if str_target in row['content']:
            number = row['content'].split(str_target)[1]
            number = number.split(" ")[0]
            return float(number)
    return total
    
def get_sum_of_file_by_dataframe(str_target, df):
    total = 0
    for index,row in df.iterrows():
        if str_target in row['content']:
            number = row['content'].split(str_target)[1]
            number = number.split(".")[0]
            total = total + int(number)
    return total
    
def get_count_of_file_by_dataframe(str_target, df):
    total = 0
    for index,row in df.iterrows():
        if str_target in row['content']:
            total = total + 1
    return total

# ----------------------------------------------------------------------------------
# def split_string(string, delimiter):
#     return str(string.split(delimiter)[1])  