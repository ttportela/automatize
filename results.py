'''
Created on Feb, 2021

@author: Tarlis Portela
'''
# --------------------------------------------------------------------------------
# import os
# # import sys
# # import numpy as np
# import pandas as pd
# import glob2 as glob
# from datetime import datetime
from .main import importer, display
importer(['S', 'glob', 'datetime'], globals())
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
    if set(name) & set(['*', '#', 'AT', 'at', 'RF', 'S', 's']): 
        list_stats = list_stats + [
            ['ACC (RF)',             'ACC',     'RF'],
        ]
    if set(name) & set(['*', '#', 'AT', 'SVM', 'S']): 
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
        
    if set(name) & set(['*', '#', 'D']):
        list_stats = list_stats + [
            ['Date', 'endDate', ''],
        ]

    return list_stats
# --------------------------------------------------------------------------------

# def results2df(res_path, prefix, modelfolder='model', isformat=True):
#     filelist = []
#     filesList = []

#     # 1: Build up list of files:
#     print("Looking for result files in " + os.path.join(res_path, prefix, '**', '*.txt' ))
#     for files in glob.glob(os.path.join(res_path, prefix, '**', '*.txt' )):
#         fileName, fileExtension = os.path.splitext(files)
#         filelist.append(fileName) #filename without extension
#         filesList.append(files) #filename with extension
    
#     # 2: Create and concatenate in a DF:
# #     ct = 1
#     df = pd.DataFrame()

#     cols = []

#     df[' '] = ['Candidates', 'Movelets', 'ACC (MLP)', 'ACC (RF)', 'ACC (SVM)' , 'Time (Movelets)', 'Time (MLP)', 'Time (RF)', 'Time (SVM)', 'Trajs. Compared', 'Trajs. Pruned', 'Date']
#     df['Dataset'] = ""
#     df['Dataset'][0] = prefix
# #     df = df[['Dataset',' ']]
#     for ijk in filesList:
#         method = os.path.basename(ijk)[:-4]
#         cols.append(method)
        
#         path = os.path.dirname(ijk)
#         df[method] = addResults(df, ijk, path, method, modelfolder, isformat)
      
#     print("Done.")
#     cols.sort()
#     cols = ['Dataset',' '] + cols
#     return df[cols]

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
#     from automatize.results import printLatex
    
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

#         return df
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
    
#     import os
#     import pandas as pd
#     from IPython.display import display
# #     from automatize.results import printLatex, kFoldResults
    
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
    
#     from automatize.results import format_stats
    if isformat:
        for column in df.columns[2:n]:
            df[column] = format_stats(df, column, list_stats)
        for column in df.columns[n:]:
            df[column] = df[column].map(lambda x: '{:.2f}%'.format(x))

    if istodisplay:
        display(df)
        
    return df

# ------------------------------------------------------------
# def kFoldResults(res_path, prefix, method, modelfolder='model', isformat=True, list_stats=[]):
#     filelist = []
#     filesList = []

#     # 1: Build up list of files:
#     print("Looking for result files in " + os.path.join(res_path, prefix, 'run*', method, method+'.txt' ))
#     for files in glob.glob(os.path.join(res_path, prefix, 'run*', method, method+'.txt' )):
#         fileName, fileExtension = os.path.splitext(files)
#         filelist.append(fileName) #filename without extension
#         filesList.append(files) #filename with extension
    
#     # 2: Create and concatenate in a DF:
# #     ct = 1
#     df = pd.DataFrame()

#     cols = []

#     df[' '] = ['Candidates', 'Movelets', 'ACC (MLP)', 'ACC (RF)', 'ACC (SVM)' , 'Time (Movelets)', 'Time (MLP)', 'Time (RF)', 'Time (SVM)', 'Trajs. Compared', 'Trajs. Pruned', 'Date']
#     df['Dataset'] = ""
#     df['Dataset'][0] = prefix
            
#     for ijk in filesList:
#         path = os.path.dirname(ijk)
#         run = os.path.basename(os.path.abspath(os.path.join(path, '..')))
        
#         cols.append(run)
#         df[run] = addResults(df, ijk, path, method, modelfolder, False)
        
#         if containErrors(ijk):
#             print('*** Warning: '+method+' contains errors ***')
        
    
#     # ---
#     if len(list_stats) > 0:
#         # Add Stats Cols
# #         ncols = len(df.columns)-2
#         j = 12
#         i = 0
#         for x in list_stats:
#             df.loc[j+i] = [x[0], ''] + ([0] * len(cols))
#             i += 1
            
#         # Add stats:
#         for ijk in filesList:
#             i = 0
#             run = os.path.basename(os.path.abspath(os.path.join(os.path.dirname(ijk), '..')))
#             stats = get_stats(ijk, list_stats)
#             for key, value in stats.items():
#                 df[run][j+i] = value
#                 i += 1
    
#     # ---
#     df[method] = df.loc[:, cols].mean(axis=1)
#     df[method].loc[11] = -1
    
#     if isformat:
#         for column in cols:
#             df[column] = format_col(df, column)

#         df[method] = format_col(df, method)
        
#     cols = ['Dataset',' '] + cols + [method]
#     return df[cols]


# def containErrors(file):
#     return open(file, 'r').read().find('java.') > -1

# def addResults(df, resfile, path, method, modelfolder='model', isformat=True):
#     print("Loading " + method + " results from: " + path)
#     data = read_csv(resfile)
    
#     try:
# #         mk = '%a %b %d %H:%M:%S CET %Y'
#         dtstr = data.iloc[-1]['content']
# #         dtstr = data.iloc[-2]['content'] if dtstr == '' else dtstr
# #         dt = datetime.strptime(dtstr, mk)
#         import dateutil.parser
#         dt = dateutil.parser.parse(dtstr).timestamp()
# #         dt = dt.strftime("%d/%m/%y-%H:%M:%S")
#     except ValueError:
#         dt = -1
    
#     total_can = get_sum_of_file_by_dataframe("Number of Candidates: ", data)
#     total_mov = get_sum_of_file_by_dataframe("Total of Movelets: ", data)
#     trajs_looked = get_sum_of_file_by_dataframe("Trajs. Looked: ", data)
#     trajs_ignored = get_sum_of_file_by_dataframe("Trajs. Ignored: ", data)
    
#     if os.path.exists(os.path.join(path, 'npoi_results.txt')):
#         #'POI' in path or 'NPOI' in path or 'WNPOI' in path or 'POIFREQ' in path:
#         time = get_total_number_of_ms("Processing time: ", read_csv(os.path.join(path, 'npoi_results.txt')))
#     else:
#         time = get_total_number_of_ms("Processing time: ", data)

#     mlp_acc = getACC_MLP(path, method, modelfolder) * 100
#     rf_acc  = getACC_RF(path, modelfolder) * 100
#     svm_acc = getACC_SVM(path, modelfolder) * 100

#     mlp_t = getACC_time(path, 'MLP', modelfolder)
#     rf_t  = getACC_time(path, 'RF', modelfolder)
#     svm_t = getACC_time(path, 'SVM', modelfolder)

#     if isformat:
#         total_can = '{:,}'.format(total_can) if total_can > 0 else "-"
#         total_mov = '{:,}'.format(total_mov) if total_mov > 0 else "-"
        
#         mlp_acc = "{:.3f}".format(mlp_acc) if mlp_acc > 0 else "-"
#         rf_acc  = "{:.3f}".format(rf_acc)  if rf_acc  > 0 else "-"
#         svm_acc = "{:.3f}".format(svm_acc) if svm_acc > 0 else "-"
        
# #         time  = '%dh%dm%ds' % printHour(time)  if time  > 0 else "-"
# #         mlp_t = '%dh%dm%ds' % printHour(mlp_t) if mlp_t > 0 else "-"
# #         rf_t  = '%dh%dm%ds' % printHour(rf_t)  if rf_t  > 0 else "-"
# #         svm_t = '%dh%dm%ds' % printHour(svm_t) if svm_t > 0 else "-"
        
#         time   = format_hour(time)
#         mlp_t  = format_hour(mlp_t)
#         rf_t   = format_hour(rf_t)
#         svm_t  = format_hour(svm_t)

#         trajs_looked  = '{:,}'.format(trajs_looked)  if trajs_looked  > 0 else "-"
#         trajs_ignored = '{:,}'.format(trajs_ignored) if trajs_ignored > 0 else "-"
        
#         dt = format_date(dt)
        
#     return (total_can, total_mov, mlp_acc, rf_acc, svm_acc, 
#                   time, mlp_t, rf_t, svm_t, 
#                   trajs_looked, trajs_ignored, dt)

# def format_col(df, method):
#     line = [format_cel(df, method, 0, '{val:,}'),
#             format_cel(df, method, 1, '{val:,}'), 
#             format_celf(df, method, 2, '{val:.3f}'),
#             format_celf(df, method, 3, '{val:.3f}'),
#             format_celf(df, method, 4, '{val:.3f}'),
#             format_celh(df, method, 5, '%dh%02dm%02ds'),
#             format_celh(df, method, 6, '%dh%02dm%02ds'),
#             format_celh(df, method, 7, '%dh%02dm%02ds'),
#             format_celh(df, method, 8, '%dh%02dm%02ds'),
#             format_cel(df, method, 9, '{val:,}'),
#             format_cel(df, method, 10, '{val:,}'),
#             format_date(df.at[11,method])]
    
#     for i in range(12, df.shape[0]):
#         try:
#             line.append(format_cel(df, method, i, '{val:,}'))
#         except TypeError:
#             line.append(df.at[i,method])
    
#     return line

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
    
    return line
    
def format_cel(df, method, row, pattern):
    value = int(df.at[row,method])
    return format_float(value, pattern) #pattern.format(df.at[row,method]) 
    
def format_celf(df, method, row, pattern):
    value = float(df.at[row,method])
    value = format_float(value, pattern) #pattern.format(df.at[row,method]) 
    return value
    
def format_celh(df, method, row, pattern):
    return format_hour(df.at[row,method])

def format_float(value, pattern='{val:.3f}'):
    if value > 0:
        return pattern.format(val=value)
    else: 
        return "-"

def format_date(ts):
#     from main import importer
    importer(['datetime'], globals())
    
#     from datetime import datetime
    try:
        return datetime.fromtimestamp(ts).strftime("%d/%m/%y-%H:%M:%S") if ts > -1 else '-'
    except TypeError:
        return ts

def format_hour(millis):
    if millis > 0:
        hours, rem = divmod(millis, (1000*60*60))
        minutes, rem = divmod(rem, (1000*60))
        seconds, rem = divmod(rem, 1000)
#         seconds = (rem / 1000.0)
#         hours, minutes, seconds = printHour(millis) 
        value = ''
        if hours > 0:
            value = value + ('%dh' % hours)
        if minutes > 0:
            value = value + (('%02dm' % minutes) if value != '' else ('%dm' % minutes))
        if seconds > 0:
            value = value + (('%02ds' % seconds) if value != '' else ('%ds' % seconds))
        if value == '':
            value = value + (('%02.3fs' % (rem/1000)) if value != '' else ('%.3fs' % (rem/1000)))
        return value
    else: 
        return "-"

# ----------------------------------------------------------------------------------
def getACC_time(path, label, modelfolder='model'):
    acc = 0.0
#     print(path)
    if getPOISFile(path, modelfolder) and label == 'MLP':
        res_file = getPOISFile(path, modelfolder)
        if res_file:
            data = read_csv(res_file)
            acc = get_first_number("Classification Time: ", data)
    elif getTECFile(path, modelfolder):
        data = pd.read_csv(getTECFile(path, modelfolder), index_col=0)
        data = data.set_index('classifier')
        acc = float(data['time']['EnsembleClassifier']) if 'time' in data.columns else 0
    else:
        data = getACC_data(path, 'classification_times.csv', modelfolder)
        if data is not None:
            acc = data[label][0]
    return acc

def getACC_RF(path, modelfolder='model'):
    acc = 0
    data = getACC_data(path, 'model_approachRF300_history.csv', modelfolder)
    if data is not None:
        acc = data['1'].iloc[-1]
    return acc

def getACC_SVM(path, modelfolder='model'):
    acc = 0
    data = getACC_data(path, 'model_approachSVC_history.csv', modelfolder)
    if data is not None:
        acc = data.loc[0].iloc[-1]
    return acc

def getACC_MLP(path, method, modelfolder='model'):
    acc = 0
#     print(path, method, modelfolder)
    if "MARC" in method:
        res_file = getMARCFile(path, method, modelfolder)
        if res_file:
            data = pd.read_csv(res_file)
            acc = data['test_acc'].iloc[-1]
    elif getPOISFile(path, modelfolder):
        res_file = getPOISFile(path, modelfolder)
        if res_file:
            data = read_csv(res_file)
            acc = get_first_number("Acc: ", data)
    elif getTECFile(path, modelfolder):
        data = pd.read_csv(getTECFile(path, modelfolder), index_col=0)
        data = data.set_index('classifier')
        acc = data['accuracy']['EnsembleClassifier']
    else:
        data = getACC_data(path, 'model_approach2_history_Step5.csv', modelfolder)
        if data is not None:
            acc = data['val_accuracy'].iloc[-1]
    return acc

def getACC_data(path, approach_file, modelfolder='model'):
#     from main import importer
#     importer(['S'], locals())
    
    res_file = os.path.join(path, modelfolder, approach_file)
    if os.path.isfile(res_file):
        data = pd.read_csv(res_file)
        return data
    else:
        return None

def printHour(millis):
    millis = int(millis)
    seconds=(millis/1000)%60
    seconds = int(seconds)
    minutes=(millis/(1000*60))%60
    minutes = int(minutes)
    hours=(millis//(1000*60*60))

#     print ("%dh%dm%ds" % (hours, minutes, seconds))
    return (hours, minutes, seconds)
# ----------------------------------------------------------------------------------
# def getMARCFile(path, method, modelfolder):
# #     from main import importer
# #     importer(['S'], locals())
    
#     res_file = os.path.join(path, method + '.csv')
#     if not os.path.isfile(res_file):
#         res_file = os.path.join(path, modelfolder,  modelfolder + '_results.csv')
#     if not os.path.isfile(res_file):
#         res_file = glob.glob(os.path.join(path, '**', method+'*' + '_results.csv'), recursive=True)[0]
#         print(res_file)
#     if not os.path.isfile(res_file):
#         return False
#     return res_file
def getMARCFile(path, method, modelfolder):
    res_file = os.path.join(path, method + '.csv')
#     print('getMARCFile', path, method, modelfolder)
    if not os.path.isfile(res_file):
        res_file = os.path.join(path, modelfolder,  modelfolder + '_results.csv')
    if not os.path.isfile(res_file):
        res_file = glob.glob(os.path.join(path, '**', method+'*' + '_results.csv'), recursive=True)
        if len(res_file) > 0 and os.path.isfile(res_file[0]):
            res_file = res_file[0]
        else:
            return False
    return res_file

def getTECFile(path, modelfolder):
    res_file = glob.glob(os.path.join(path, modelfolder, 'model_approachEnsemble_history.csv'), recursive=True)
    if len(res_file) > 0 and os.path.isfile(res_file[0]):
        return res_file[0]
    else:
        return False

def getPOISFile(path, modelfolder):
#     from main import importer
#     importer(['S'], locals())
    
    res_file = os.path.join(path, 'poifreq_results.txt')
    if not os.path.isfile(res_file):
        res_file = os.path.join(path, modelfolder, 'poifreq_results.txt')
    if not os.path.isfile(res_file):
        return False
    return res_file
# --------------------------------------------------------------------------------->   
def read_csv(file_name):
#     from main import importer
#     importer(['S'], locals())
    
#     file_name = DIR_V1 + "results/"+file_name + '.txt'
    data = pd.read_csv(file_name, header = None, delimiter='-=-', engine='python', on_bad_lines='skip') #error_bad_lines=False, warn_bad_lines=False
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

# def get_min_number_of_file_by_dataframe(str_target, df):
#     total = 99999
#     for index,row in df.iterrows():
#         if str_target in row['content']:
#             number = row['content'].split(str_target)[1]
#             number = int(number.split(".")[0])
#             total = min(total, number)
#     return total

def split_string(string, delimiter):
    return str(string.split(delimiter)[1])  

def get_stats(resfile, path, method, list_stats, modelfolder='model', show_warnings=True):
#     from main import importer
#     importer(['S'], locals())
    
    stats = []
    
#     print("Loading " + method + " results from: " + path)
    if os.path.exists(os.path.join(path, os.path.basename(path)+'.txt')):
        data = read_csv(os.path.join(path, os.path.basename(path)+'.txt'))
    else:
        data = read_csv(resfile)
    
    for x in list_stats:
        ssearch = x[2]+": "
#         print('[DEBUG]',ssearch)
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
        
        elif x[1] == 'time':
            stats.append( get_total_number_of_ms(ssearch, data) )
        
        elif x[1] == 'accTime':
#             print(path, x[2], modelfolder)
            stats.append( getACC_time(path, x[2], modelfolder) )
        
        elif x[1] == 'totalTime':
            timeRun = get_total_number_of_ms(x[2].split('|')[0]+": ", data) 
            timeAcc = getACC_time(path, x[2].split('|')[1], modelfolder) 
            if show_warnings and timeRun <= 0:
                print('*** Warning "totalTime": No Run Time for '+resfile+'.')
            if show_warnings and timeAcc <= 0:
                print('*** Warning "totalTime": No ACC Time for '+resfile+'.')
            stats.append(timeRun + timeAcc)
        
        elif x[1] == 'ACC':
#             print(path, method, modelfolder)
            if x[2] in ['MLP', '']:
                acc = getACC_MLP(path, method, modelfolder)
            elif x[2] == 'RF':
                acc = getACC_RF(path, modelfolder)
            elif x[2] == 'SVM':
                acc = getACC_SVM(path, modelfolder)
                
            if show_warnings and acc <= 0:
                print('*** Warning "ACC": not found for '+resfile+'.')
            
            stats.append( acc * 100 )
        
        elif x[1] == 'endDate':
            try:
#                 from main import importer
                importer(['dateparser'], globals())
    
                dtstr = data.iloc[-1]['content']
#                 import dateutil.parser
                stats.append( dateparser.parse(dtstr).timestamp() )
            except ValueError:
                stats.append( -1 )
                
    return stats

# --------------------------------------------------------------------------------->  
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

# ----------------------------------------------------------------------------------
# def printProcess(prefix, dir_path):
#     file = os.path.join(prefix, dir_path)
#     res_file = os.path.join(RES_PATH, file + '.txt')
    
#     data = read_csv(res_file)
#     total_can = get_sum_of_file_by_dataframe("Number of Candidates: ", data)
#     total_mov = get_sum_of_file_by_dataframe("Total of Movelets: ", data)
#     trajs_looked = get_sum_of_file_by_dataframe("Trajs. Looked: ", data)
#     trajs_ignored = get_sum_of_file_by_dataframe("Trajs. Ignored: ", data)
#     time = get_total_number_of_ms("Processing time: ", data)
    
#     print('# <=====================================================>')
#     print('# '+file)
#     print("# Number of Candidates: " + str(total_can))
#     print("# Total of Movelets:    " + str(total_mov))
#     print("# Processing time:      " + str(time) + ' ms -- %d:%d:%d' % printHour(time))
#     print('# --')
    
#     acc  = getACC_SVM(prefix, method) * 100
#     if acc is not 0:
#         print("# SVM ACC:    " + acc)
    
#     acc  = getACC_RF(prefix, method) * 100
#     if acc is not 0:
#         print("# Random Forest ACC:    " + acc)
        
#     acc  = getACC_MLP(prefix, method) * 100
#     if acc is not 0:
#         print("# Neural Network ACC:   " + acc)
        
    
#     print('# --')
#     print("# Total of Trajs. Looked: " + str(trajs_looked))
#     print("# Total of Trajs. Ignored:   " + str(trajs_ignored))
#     print("# Total of Trajs.:    " + str(trajs_looked+trajs_ignored))
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
    filesList = filesList + findFiles('classification_times.csv')
    filesList = filesList + findFiles('*_results.txt')
    filesList = filesList + findFiles('model_approachEnsemble_history.csv')
    return filesList

def decodeURL(ijk):
    rpos = ijk.find('run')
    path = ijk[:ijk.find(os.path.sep, rpos+5)]
    method = ijk[rpos+5:ijk.rfind(os.path.sep)]

    if ijk.endswith('poifreq_results.txt'):
        file = ijk
        method = method[method.find(os.path.sep)+1:]
    elif ijk.endswith('model_approachEnsemble_history.csv'):
        file = ijk
        method = method[:method.find(os.path.sep)] 
    else:
        file = glob.glob(os.path.join(path, '**', '*.txt'), recursive=True)[0]
        method = method[:method.find(os.path.sep)]

    run = path[rpos:rpos+4]
    run = (run)[3:]

    method, subset = method.split('-')[:2]

    prefix = os.path.basename(path[:rpos-1])

    model = os.path.dirname(ijk)
    model = model[model.rfind(os.path.sep)+1:]
    
    if ijk.endswith('model_approachEnsemble_history.csv'):
        method += '_'+model.split('_')[-1]

    random = '1' if '-' not in model else model.split('-')[-1]

    return run, random, method, subset, prefix, model, path, file

def organizeResults(filesList, sub_set=None):
    results = {}
    for ijk in filesList:
        run, random, method, subset, prefix, model, path, file = decodeURL(ijk)
        
        is_POIF = ijk.endswith('poifreq_results.txt')
        is_TEC  = ijk.endswith('approachEnsemble_history.csv')
        
        var = sub_set if sub_set and is_POIF else subset
        
        # is this forced?
        if is_POIF and sub_set:
            method += '_'+subset
            subset = sub_set
        
        if sub_set and var != sub_set:
            continue
            
        dataset = prefix +'-'+ var
        mname   = method +'-'+ subset #(var if sub_set and is_POIF else subset)
        
        if dataset not in results.keys():
            results[dataset] = {}
        if mname not in results[dataset].keys():
            results[dataset][mname] = []
        results[dataset][mname].append([run, random, method, subset, prefix, model, path, file])

    return results

def history(res_path): #, prefix, method, list_stats=STATS(['S']), modelfolder='model', isformat=True):
    
    importer(['S', 'glob', 'STATS', 'get_stats', 'containErrors', 'np'], globals())
    histres = pd.DataFrame(columns=['#','timestamp','dataset','subset','run','random','method','runtime', 'classifier','accuracy','cls_runtime','error','file'])

    filesList = getResultFiles(res_path)
    
    list_stats = STATS(['#'])
    list_stats_ind = [p[x] for p in list_stats for x in range(len(p))]
            
    for ijk in filesList:
        run, random, method, subset, prefix, model, path, file = decodeURL(ijk)
        
        stats = get_stats(file, path, method+'-'+subset, list_stats, modelfolder=model)
        def gstati(x):
            return stats[list_stats_ind.index(x) // 3]
        def gstat(x):
            return stats[list_stats.index(x)]
        def getrow(run, random, method, subset, prefix, model, path, file, result):
            return {
                '#': 0,
                'timestamp': gstati('endDate'),
                'dataset': prefix,
                'subset': subset,
                'run': run,
                'random': random,
                'method': method,
                'runtime': result[2] if ijk.endswith('model_approachEnsemble_history.csv') else gstati('time'),
                'classifier': result[0],
                'accuracy': result[1],
                'cls_runtime': result[2],
                'error': containErrors(file) or containWarnings(file),
                'file': ijk,
            }
        
        if ijk.endswith('model_approachEnsemble_history.csv'):
            data = pd.read_csv(file, index_col=0)
            data = data.set_index('classifier')
            for index, row in data[:-1].iterrows():
                classifier = ['#'+index, row['accuracy'] * 100, row['time']]
                aux_hist = getrow(run, random, method, subset, prefix, model, path, file, classifier)
                histres = pd.concat([histres, pd.DataFrame([aux_hist])])
                
            classifier = [model.split('-')[0].split('_')[-1], data['accuracy'][-1] * 100, data['time'][-1]]
            aux_hist = getrow(run, random, method, subset, prefix, model, path, file, classifier)
            histres = pd.concat([histres, pd.DataFrame([aux_hist])])
        else:
            for x in ['MLP', 'RF', 'SVM']:
                acc = gstat(['ACC ('+x+')', 'ACC', x])
                if acc > 0:
                    classifier = [x, acc] + [ gstat(['Time ('+x+')', 'accTime', x],) ]
                    aux_hist = getrow(run, random, method, subset, prefix, model, path, file, classifier)
                    histres = pd.concat([histres, pd.DataFrame([aux_hist])])
        
    # ---
    # Post treatment:
    histres['name']   = histres['method'] + '-' + histres['subset'] + '-' + histres['classifier']
    histres['key'] = histres['dataset'] + '-' + histres['subset'] + '-' + histres['run']
    # Ordering / Renaming:
    histres.reset_index(drop=True, inplace=True)

    return histres

def runningProblems(ijk):
    e1 = containErrors(ijk)
    e2 = containWarnings(ijk)
    e3 = containTimeout(ijk)
    s = False
    if e1 or e2 or e3:
        s = ('[ERROR]' if e1 else '[  -  ]')+('[WARN.]' if e2 else '[  -  ]')+('[T.OUT]' if e3 else '[  -  ]')
    return s

def check_run(res_path, show_warnings=False):
    
    importer(['S', 'glob', 'STATS', 'get_stats', 'check_run', 'format_hour', 'np'], globals())
    filesList = []

    def findFiles(x):
        search = os.path.join(res_path, '**', x)
        fl = []
        for files in glob.glob(search, recursive=True):
            fileName, fileExtension = os.path.splitext(files)
            fl.append(files) #filename with extension
        return fl
            
    filesList = filesList + findFiles('*.txt')
    
    def decode_url(ijk):
        rpos = ijk.find('run')
        path = ijk[:ijk.find(os.path.sep, rpos+5)]
        method = ijk[rpos+5:ijk.rfind(os.path.sep)]

        if ijk.endswith('poifreq_results.txt'):
            method = method[method.find(os.path.sep):]

        run = path[rpos:rpos+4]
        run = (run)[3:]
        
        method, subset = method.split('-')[:2]
        
        prefix = os.path.basename(path[:rpos-1])

        return run, method, subset, prefix, path
            
    filesList.sort()
    for ijk in filesList:
        run, method, subset, prefix, path = decode_url(ijk)
        e = runningProblems(ijk)
        if e:
            print('[*] NOT OK:\t', method, '\t', prefix, '\t', run, '\t', subset, '\t', e)
        else:
            res = get_stats(ijk, path, method, STATS(['AccTT']), show_warnings=show_warnings)
            print('        OK:\t', method, '\t', prefix, '\t', run, '\t', subset, 
                  '\t ACC:', format_float(res[0]), '\t t:', format_hour(res[1]))


def compileResults(res_path, sub_set=None, list_stats=STATS(['S']), isformat=True, k=False):
    importer(['S', 'glob'], globals())
    
    results = organizeResults(getResultFiles(res_path), sub_set)
    
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
            for run, random, method, subset, prefix, model, path, file in results[dataset][mname]:
                run_cols.append(run)
                df[run] = get_stats(file, path, method, list_stats, model)

                e = runningProblems(file)
                if e:
                    partial_result = True
                    print('*** Warning: '+mname+'-'+run+' contains errors > ' + e)
            # ---
            if k and len(run_cols) != k:
                partial_result = True
            
            df[method] = df.loc[:, run_cols].mean(axis=1)
            if list_stats[-1][1] == 'enddate':
                df[method].loc[df.index[-1]] = -1

#             display(df)
            print('Adding:', dataset, '\t', len(run_cols), 'runs', '\t', mname)
            if isformat:
                for column in run_cols:
                    df[column] = format_stats(df, column, list_stats)

                df[method] = format_stats(df, method, list_stats)
                
                if partial_result:
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
    table = table.reset_index(drop=True)
    return table