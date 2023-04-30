# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Feb, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
from .main import importer, display
importer(['S', 'glob', 'datetime', 're', 'itertools'], globals())

from .inc.script_def import getSubset
from .inc.io.results_read import ResultConfig
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
    elif set(name) & set(['m', 's', 'MC']): # Movelets & Candidates  Simple
        list_stats = list_stats + [
            ['Candidates',           'sum',   'Number of Candidates'],
            ['Movelets',             'sum',   'Total of Movelets'],
        ]
        
     # ACC All
    if set(name) & set(['*', '#', 'AT', 'at', 'ACC', 'MLP', 'N', 'S', 's', 'AccTT']):
        list_stats = list_stats + [
            ['ACC (NN)',            'ACC',     'MLP'],
        ]
    if set(name) & set(['*', 'AT', 'at', 'ACC', 'RF', 'S', 's']): 
        list_stats = list_stats + [
            ['ACC (RF)',             'ACC',     'RF'],
        ]
    if set(name) & set(['*', 'AT', 'ACC', 'SVM', 'S']): 
        list_stats = list_stats + [
            ['ACC (SVM)',            'ACC',     'SVM'],
        ]
        
    # F-SCORE
    if set(name) & set(['*', 'F1']):
        list_stats = list_stats + [
            ['F-Score (NN)',          'F1',     'MLP'],
        ]
    if set(name) & set(['*', 'F1']):
        list_stats = list_stats + [
            ['F-Score (RF)',          'F1',     'RF'],
        ]
    if set(name) & set(['*', 'F1']):
        list_stats = list_stats + [
            ['F-Score (SVM)',         'F1',     'SVM'],
        ]
    
    # TIME
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
        if set(name) & set(['*', 'MSG']): # the only not boolean
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
        if set(name) & set(['*', 'isMsg']):
            list_stats = list_stats + [
                ['Messages', 'msg', 'isMsg'],
            ]
        
    if set(name) & set(['*', '#', 'D']):
        list_stats = list_stats + [
            ['Date', 'endDate', ''],
        ]

    return list_stats

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

    # Second add the remaining:
    filesList = filesList + findFiles('*-*.txt') # NN / RF / SVM
    filesList = filesList + findFiles('classification_times.csv') # NN / RF / SVM
    filesList = filesList + findFiles('poifreq_results.txt') # POI-F / POI-FS
    #filesList = filesList + findFiles('MARC-*.txt') # MARC
    filesList = filesList + findFiles('model_approachEnsemble_history.csv') # TEC
    filesList = filesList + findFiles('TEC*/*.txt') # TEC
    
    filesList = list(set(filesList))
    
    filesList = list(filter(lambda file: 'POI' not in os.path.basename(file).split('-')[0], filesList))
    
    filesList.sort()
    
    return filesList

def instantiateResults(filesList, subsets=None):
    results = {}
    for ijk in filesList:
        m = ResultConfig.instantiate(ijk)
        
        if not m or (subsets and m.subset not in subsets):
            continue
            
        dataset = m.prefix +'-'+ m.subset 
        mname   = m.method +'-'+ m.subset 
        
        if dataset not in results.keys():
            results[dataset] = {}
            
        if mname not in results[dataset].keys():
            results[dataset][mname] = set()
        results[dataset][mname].add(m)

    return results

def check_run(res_path, show_warnings=False):
    
    #importer(['S', 'glob', 'STATS', 'get_stats', 'format_hour', 'np'], globals())
    
    filesList = getResultFiles(res_path)
    lr = list(set(map(lambda ijk: ResultConfig.instantiate(ijk), filesList)))
    lr.sort()
    
    def adj(s, size=15):
        return s.ljust(size, ' ')
    
    SEP = ' ' #'\t'
    #filesList.sort()
    for m in lr:
        #run, random, method, subset, subsubset, prefix, model, path, file, statsf = decodeURL(ijk)
        e = m.runningProblemsStr()
        
        res = m.metrics(STATS(['AccTT']), show_warnings=show_warnings)
        #res = get_stats([file, statsf], path, method, STATS(['AccTT']), model, show_warnings=show_warnings)
        line = '[' + adj(format_float(res[0]),6) +']['+ format_hour(res[1])+']'
        
        if m.runningErrors():
            print('[*] NOT OK:'+SEP, adj(m.method, 20), SEP, adj(m.prefix), SEP, adj(m.run,3), SEP, adj(m.subset), SEP, e, line)
        else:
            print('        OK:'+SEP, adj(m.method, 20), SEP, adj(m.prefix), SEP, adj(m.run,3), SEP, adj(m.subset), SEP, e, line)


def history(res_path): #, prefix, method, list_stats=STATS(['S']), modelfolder='model', isformat=True):
    
    importer(['S', 'glob', 'STATS', 'get_stats', 'containErrors', 'np'], globals())
    histres = pd.DataFrame(columns=['#','timestamp','dataset','subset','subsubset','run','random','method','classifier', \
                                    #'metric:accuracy', 'metric:f1_score', \
                                    'runtime','cls_runtime','totaltime','candidates','movelets','error','file'])

    filesList = getResultFiles(res_path)
    lr = list(set(map(lambda ijk: ResultConfig.instantiate(ijk), filesList)))
    lr.sort()

    D  = STATS(['D'])
    MC = STATS(['MC'])
            
    for m in lr:
        def getrow(config, result):
            
            aux = m.metrics(MC)
            
            met_dict = m.allMetrics(classifier=result[0])
            
            met_dict.update( {
                '#': 0,
                'timestamp': m.metrics(D)[0], #gstati('endDate'),
                'dataset': config.prefix,
                'subset': config.subset,
                'run': config.run,
                'subsubset': config.subsubset,
                'random': config.random,
                'method': config.method,
                'classifier': result[0],
                #'metric:accuracy': result[1],
                'runtime': config.runtime(),
                'cls_runtime': result[2],
                'totaltime': config.totaltime() if result[0] in ['NN', 'MLP'] else config.totaltime(classifier=result[0]),
                'candidates': aux[0],
                'movelets': aux[1],
#                'error': config.containErrors() or config.containWarnings() or config.containTimeout(),
                'error': config.containErrors() or config.containTimeout(),
                'file': config.statsf,
            } )
            return met_dict
        
        for classifier in m.classification():
            aux_hist = getrow(m, classifier)
            histres = pd.concat([histres, pd.DataFrame([aux_hist])])

    # ---
    # Post treatment:
    histres['name']   = histres['method'].map(str) + '-' + histres['subsubset'].map(str) + '-' + histres['classifier'].map(str)
    histres['key'] = histres['dataset'].map(str) + '-' + histres['subset'].map(str) + '-' + histres['run'].map(str)
    
    histres.sort_values(['dataset', 'subset', 'run', 'method', 'classifier'], inplace=True)
    
    # Ordering / Renaming:
    histres.reset_index(drop=True, inplace=True)
    histres['#'] = histres.index

    return histres


def compileResults(res_path, subsets=['specific'], list_stats=STATS(['S']), isformat=True, k=False, return_ranks=False, verbose=False):
    importer(['S', 'glob'], globals())
    
    results = instantiateResults(getResultFiles(res_path), subsets)

    cols = []
    
    table = pd.DataFrame()
    ranks = pd.DataFrame()
    for dataset in results.keys():
        data = pd.DataFrame()
        rank = pd.DataFrame()
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
            for m in results[dataset][mname]:
                run_cols.append(m.run)
                df[m.run] = m.metrics(list_stats, show_warnings=True if verbose else False)#get_stats([file, statsf], path, method, list_stats, model)

                e = m.unfinished()
                if e:
                    partial_result = True
                    if verbose:
                        print('*** Warning: '+mname+'-'+m.run+' contains problems > ' + m.runningProblemsStr())
            # ---
            if k and len(run_cols) != k:
                partial_result = True
            method = mname.split('-')[0]
            df[method] = summaryRuns(df, run_cols, list_stats)
            
            if verbose and verbose > 1:
                print('Adding:', dataset, '\t', len(run_cols), 'run(s)', '\t', mname)
            
            if return_ranks:
                rank[method] = df[[method]]
            
            if isformat:
                for column in run_cols:
                    df[column] = format_stats(df, column, list_stats)

                df[method] = format_stats(df, method, list_stats)
                
                if partial_result: # and not ('MARC' in method or 'POI' in method or 'TEC' in method):
                    df[method] = df[method].add('*')
    
            # ---
            if not len(data.columns) > 0:
                data = df[['Dataset', ' ']].copy()
            if method in df.columns:
                data[method] = df[[method]]
            else:
                data[method] = '' # Probably never happen ?
        # ---
        table = pd.concat([table, data], axis=0)
        if return_ranks:
            ranks = pd.concat([ranks, rank], axis=0)
            
    # ---
    table.fillna(value='', inplace=True)
    table = table.reset_index(drop=True)

    if return_ranks:
        ranks = ranks.reset_index(drop=True)
        ranks = resultRanks(ranks, list_stats)
        return table, ranks
    else:
        return table
    
# ----------------------------------------------------------------------------------
def resultsDiff(df, ref_cols=[2], list_stats=STATS(['S']), isformat=True, verbose=True):
    n = len(df.columns)
    for ref in ref_cols:
        for col in range(2, n):
            #if col not in ref_cols:
            if col != ref:
                a = df.iloc[:,ref]
                b = df.iloc[:,col]
                name = str(df.columns[ref])+'-'+str(df.columns[col]) +' ('+ str(ref)+'-'+str(col)+')'
                df[name] = ((a-b)*100/a) * -1 #((b-a) / b * 100.0)
    
#     from PACKAGE_NAME.results import format_stats
    if isformat:
        for column in df.columns[2:n]:
            df[column] = format_stats(df, column, list_stats)
        for column in df.columns[n:]:
            df[column] = df[column].map(lambda x: '{:.2f}%'.format(x))

    if verbose:
        display(df)
        
    return df

def resultRanks(df, list_stats=STATS(['S'])):
    rank = pd.DataFrame()
    df = df.copy()
    df = df.drop(['Dataset',' '], axis=1, errors='ignore')
    df = df.replace('', np.nan).replace('-', np.nan).replace('-*', np.nan)
    stats = itertools.cycle(list_stats)
    for i in range(len(df)):
        stat = next(stats)
        row = pd.DataFrame(df.iloc[i,:]).T
        if 'time' in stat[1] or stat[1] in ['totalTime', 'time', 'accTime', 'sum', 'max', 'min', 'count']: # Smaller First
            row = row.astype(float, errors='ignore').div(1000).astype(int, errors='ignore')
            row = row.rank(1, ascending=True)
        elif 'ACC' in stat[1] or stat[1] in ['ACC']: # Bigger First
            row = row.astype(float, errors='ignore').round(decimals=3)
            row = row.rank(1, ascending=False)
        else: # 'msg', 'endDate'
            row = row.rank(1, ascending=False)
        rank = pd.concat([rank, row])

    return rank

# --------------------------------------------------------------------------------->  
def results2latex(df, cols=None, ajust=9, clines=[], resize=True, doubleline=True, mark_results=2, ranks=None):
    if not cols:
        cols = df.columns
       
    def visibleCols(cols):
        ls = []
        for i in cols:
            ls.extend(i if type(i) == list else [i])
        return ls
    #def countCols(cols):
    #    ct = 0
    #    for c in cols:
    #        if type(c) == list:
    #            ct += len(c)
    #        else:
    #            ct += 1
    #    return ct
    def styleValue(ln, col, ajust, df, ranks, mark_results):
        value = df.at[ln,col] if col in df.columns else '-'
        value = str(value)
        if ranks is not None and mark_results > 0 and col in ranks.columns:
            marks = list(filter(lambda x:  x == x, list(ranks.loc[ln,:].unique())))
            marks.sort()
            if ranks.at[ln,col] in marks and marks.index(ranks.at[ln,col]) < mark_results:
                value = '\\'+chr(marks.index(ranks.at[ln,col])+65)+'st{'+value+'}'
        value = value.rjust(ajust, ' ') + ' '
        return value
       
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
        
        if ' ' in df.columns:
            line = '&'+ str(df.at[l,df.columns[1]]).rjust(15, ' ') + ' '
        else:
            line = ''
            
        for g in cols:
            if type(g) == list:
                g1 = g
            else:
                g1 = [g]
            for c in g1:
                c1 = searchColumn(df, c)   
                line = line + '& '+ styleValue(l, c1, ajust, df, ranks, mark_results)
        line = line + '\\\\'
        return line
    # ---

    n_ds = len(df['Dataset'].unique())-1 #TODO ?
    n_rows = int(int(len(df)) / n_ds)
    n_idx = 1 if n_rows == 1 else 2
    
    visible = visibleCols(cols) 
    methods = list(filter(lambda x: x not in ['Dataset', ' '], visible))
    n_cols = len(visible)
    
    if ranks is not None and mark_results > 0:
        for col in methods:
            if col not in ranks.columns:
                ranks[col] = np.nan
        ranks = ranks[methods]
    
    if n_rows > 1 and (' ' in df.columns and ' ' not in cols):
        visible = [' '] + visible
    if 'Dataset' in df.columns and 'Dataset' not in cols:
        visible = ['Dataset'] + visible
        
    for col in visible:
        if col not in df.columns:
            df[col] = '-'
    df = df[visible]
    
    df = df.fillna('-')
    
    # ---
    s = ('\\begin{table*}[!ht]\n')
    s += ('\\centering\n')
    s += (('' if resize else '%')+'\\resizebox{\columnwidth}{!}{\n')
    
    if n_rows == 1:
        s += ('\\begin{tabular}{|c|'+('r|'*n_cols)+'}\n')
        #if ' ' in df.columns:
        #    df.drop(' ', inplace=True, axis=1)
    else:
        s += ('\\begin{tabular}{|c|r||'+('r|'*n_cols)+'}\n')
    s += ('\\hline\n')
    # ---
    
    if n_cols > len(cols): # 1st header, We have column groups
        colg = []
        for i in range(len(cols)):
            if type(cols[i]) == list:
                colg.append('\\multicolumn{'+str(len(cols[i]))+'}{c|}{'+cols[i][0].replace('_', '-')+'}')
            else:
                colg.append(cols[i])
        s += ((' & '.join(colg)) + ' \\\\\n')
        
    colg = []
    for i in range(len(cols)): # 2nd header
        if type(cols[i]) == list:
            colg = colg + [x.replace('_', '-') for x in cols[i]]
        else:
            colg.append( ' ' ) #cols[i].replace('_', '-'))
    s += ((' & '.join(colg)) + ' \\\\\n')
    # ---
    
    for k in range(0, int(len(df)), n_rows): # rows
        if doubleline or k == 0:
            s += ('\n\\hline\n')
            
        if n_rows == 1:
            s += (df.at[k,'Dataset'].replace('_', ' ')+'\n')
        else:
            s += ('\\multirow{'+str(n_rows)+'}{2cm}{'+df.at[k,'Dataset'].replace('_', ' ')+'}\n')
            
        for j in range(0, n_rows):
            s += (getLine(df, cols[n_idx:], k+j, ajust)+'\n')
            if j in clines:
                s += ('\\cline{2-'+str(n_cols)+'}\n')
                
        if doubleline:
            s += ('\\hline\n')
    # ---
    
    s += ('\\hline\n')
    s += ('\\end{tabular}'+('}' if resize else '%}')+'\n')
    s += ('\\caption{Results for xxx dataset.}\n')
    s += ('\\label{tab:results_xxx}\n')
    s += ('\\end{table*}\n')
    
    print('Done.')
    return s

def results2xlsx(df, cols=None, seplines=[], mark_results=2, ranks=None, filename='results.xlsx'):
    if not cols:
        cols = df.columns
       
    def visibleCols(cols):
        ls = []
        for i in cols:
            ls.extend(i if type(i) == list else [i])
        return ls

    datasets = df['Dataset'].replace('', np.nan).dropna().unique()
    n_ds = len(datasets) #-1
    n_rows = int(int(len(df)) / n_ds)
    n_idx = 1 if n_rows == 1 else 2
    
    visible = visibleCols(cols) 
    methods = list(filter(lambda x: x not in ['Dataset', ' '], visible))
    n_cols = len(visible)
    
    if ranks is not None and mark_results > 0:
        for col in methods:
            if col not in ranks.columns:
                ranks[col] = np.nan
        ranks = ranks[methods]
    
    if n_rows > 1 and (' ' in df.columns and ' ' not in cols):
        visible = [' '] + visible
    if 'Dataset' in df.columns and 'Dataset' not in cols:
        visible = ['Dataset'] + visible
        
    for col in visible:
        if col not in df.columns:
            df[col] = '-'
    df = df[visible]
    
    df = df.fillna('-')
    
    # ---
    sheet_name = 'Results'
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    pd.DataFrame().to_excel(writer, sheet_name=sheet_name, index=False)
    
    workbook  = writer.book
    worksheet = writer.sheets[sheet_name]
    
    firstc_format = workbook.add_format({'bold': True, 'border': 1, 'align': 'center', 'valign': 'vcenter', 'text_wrap': True,
                                         'fg_color': '#4285f4', 'color': '#FFFFFF'})
    header_format = workbook.add_format({'bold': True, 'border': 1, 'align': 'center', 'valign': 'vcenter', 
                                         'fg_color': '#34a853', 'color': '#FFFFFF'})
    
    first_format  = workbook.add_format({'top': 1, 'left': 1, 'right': 1, 'align': 'right', 'valign': 'vcenter', 'fg_color': '#a4c2f4'})
    odd_format    = workbook.add_format({'border': 0,'left': 1,'right': 1, 'align': 'right',  'valign': 'vcenter', 'fg_color': '#c9daf8'})
    even_format   = workbook.add_format({'border': 0,'left': 1,'right': 1, 'align': 'right',  'valign': 'vcenter', 'fg_color': '#a4c2f4'})
    last_format   = workbook.add_format({'bottom': 1,'left': 1,'right': 1, 'align': 'right',  'valign': 'vcenter', 'fg_color': '#ffff00'})
    
    mark_formats = [
        workbook.add_format({'bold': True}),
        workbook.add_format({'italic': True, 'underline': True}),
        workbook.add_format({'italic': True})
    ]
    err_format = workbook.add_format({'color': '#ea4335'})
    
    if mark_results > len(mark_formats):
        mark_results = len(mark_formats)
    
    # First two columns
    first = 1 if n_cols > len(cols) else 0
    j = 0
    if 'Dataset' in df.columns:
        worksheet.merge_range(0, j, first, j, 'Dataset', firstc_format)
        i = first+1
        for ds in range(n_ds):
            worksheet.merge_range(i, j, i+n_rows-1, j, datasets[ds].replace('(specific)', ''), firstc_format)
            i = i+n_rows
            if ds+1 in seplines:
                i += 1
        worksheet.set_column(j, j, 10)
        j += 1
    if ' ' in df.columns:
        worksheet.merge_range(0, j, first, j, ' ', firstc_format)
        i = first+1
        for ds in range(n_ds):
            for r in range(n_rows):
                l = (ds * n_rows) + r
                if r == 0:
                    style = first_format
                elif r == n_rows-1:
                    style = last_format
                elif (r % 2) == 0:
                    style = even_format
                else:
                    style = odd_format
                worksheet.write(i, j, df.at[l,' '], style)
                i += 1
            if ds+1 in seplines:
                i += 1
        worksheet.set_column(j, j, 14)
        j += 1
    startc = j
    
    # First Line, group headers (if it has):
    if n_cols > len(cols):
        for i in range(len(cols)):
            if type(cols[i]) == list:
                worksheet.merge_range(0, j, 0, j+len(cols[i])-1, cols[i][0].replace('_', '-'), header_format)
                j += len(cols[i])
            else:
                worksheet.write(0, j, cols[i].replace('_', '-'), header_format)
                j += 1
            j += 1
    
    # Second Line, headers:
    j = startc
    for i in range(len(cols)):
        if type(cols[i]) == list:
            for ii in range(len(cols[i])):
                worksheet.write(first, j, cols[i][ii].replace('_', '-'), header_format)
                worksheet.set_column(j, j, 14)
                j += 1
        else:
            worksheet.write(first, j, cols[i].replace('_', '-'), header_format)
            worksheet.set_column(j, j, 14)
            j += 1
        worksheet.set_column(j, j, 1)
        j += 1
    
    #--
    def getValue(ln, col):
        value = df.at[ln,col] if col in df.columns else '-'
        value = str(value)
        style = None
        if ranks is not None and mark_results > 0 and col in ranks.columns:
            marks = list(filter(lambda x:  x == x, list(ranks.loc[ln,:].unique())))
            marks.sort()
            if ranks.at[ln,col] in marks and marks.index(ranks.at[ln,col]) < mark_results:
                style = mark_formats[marks.index(ranks.at[ln,col])]
        if '*' in value:
            style = err_format
        return value, style
    
    def write(wl, wc, l, c, style):
        def searchColumn(col):
            if '*' in col:
                import fnmatch
                filtered = fnmatch.filter(df.columns, col)
                for c1 in filtered:
                    if df.at[l,c1] != '-' and df.at[l,c1]:
                        return c1
                return filtered[0] if len(filtered) > 0 else col
            return col
         
        value, special = getValue(l, searchColumn(c))
        if special:
            #worksheet.write(wl, wc, value, style)
            worksheet.write_rich_string(wl, wc, ' ', special, value, ' ', style)
        else:
            worksheet.write(wl, wc, value, style)
    #---
    
    # The rest of data:
    i = first+1
    for ds in range(n_ds):
        for r in range(n_rows):
            l = (ds * n_rows) + r
            if r == 0:
                style = first_format
            elif r == n_rows-1:
                style = last_format
            elif (r % 2) == 0:
                style = even_format
            else:
                style = odd_format
                
            j = startc
            for jj in range(len(cols)):
                if type(cols[jj]) == list:
                    for jjj in range(len(cols[jj])):
                        write(i, j, l, cols[jj][jjj], style)
                        j += 1
                else:
                    write(i, j, l, cols[jj], style)
                    j += 1
                j += 1
                
            i += 1
        if ds+1 in seplines:
            i += 1
       
    # --
    workbook.close()
    print(filename + ', Done.')
    
#def df2Latex(df, ajust=9, clines=[]):
#    n_cols = (len(df.columns)-2)
#    n_ds = len(df['Dataset'].unique()) -1
#    n_rows = int(int(len(df)) / n_ds)
#    
#    s = '\\begin{table*}[!ht]\n'
#    s += '\\centering\n'
#    s += '\\resizebox{\columnwidth}{!}{\n'
#    s += '\\begin{tabular}{|c|r||'+('r|'*n_cols)+'}\n'
#    s += '\\hline\n'
##     print('\\hline')
#    s += (' & '.join(df.columns)) + ' \\\\\n'
#    
#    for k in range(0, int(len(df)), n_rows):
#        s += '\n\\hline\n'
#        s += '\\hline\n'
#        s += '\\multirow{'+str(n_rows)+'}{2cm}{'+df.at[k,'Dataset']+'}\n'
#        for j in range(0, n_rows):
#            s += latexLine(df, k+j, ajust) + '\n'
#            if j in clines:
#                s += '\\cline{2-'+str(n_cols+2)+'}\n'
#    
##     print('\\hline')
#    s += '\\hline\n'
#    s += '\\end{tabular}}\n'
#    s += '\\caption{Results for xxx dataset.}\n'
#    s += '\\label{tab:results_xxx}\n'
#    s += '\\end{table*}\n'
#    return s
#    
#def latexLine(df, l, ajust=12):
#    line = '&'+ str(df.at[l,df.columns[1]]).rjust(15, ' ') + ' '
#    for i in range(2, len(df.columns)):
#        line = line + '& '+ str(df.at[l,df.columns[i]]).rjust(ajust, ' ') + ' '
#    line = line + '\\\\'
#    return line

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
def summaryRuns(df, run_cols, list_stats):    
    stats = []
    for i in range(len(list_stats)):
        if list_stats[i][1] == 'endDate' or list_stats[i][1] == 'max': # For maximuns
            val = -1
            for rc in run_cols:
                val = max(val, df.at[i, rc])
            stats.append(val)
        elif list_stats[i][1] == 'min': # For minimuns
            val = float("inf")
            for rc in run_cols:
                val = min(val, df.at[i, rc])
            stats.append(val if val != float("inf") else 0)
        elif list_stats[i][2] == 'msg': # For texts
            val = ''
            for rc in run_cols:
                e = df.at[i, rc]
                val = val + (str(e) if e else '_') + (',' if rc != run_cols[-1] else '')
            stats.append(val)
        elif list_stats[i][1] == 'msg': # For booleans
            val = False
            for rc in run_cols:
                val = True if val or df.at[i, rc] else False
            stats.append(val)
        else: # For mean
            val = 0
            ct = 0
            for rc in run_cols:
                val += df.at[i, rc]
                ct += 1
            stats.append(val / ct)
    
    return stats

#def runningProblems(ijk):
#    e1 = containErrors(ijk)
#    e2 = containWarnings(ijk)
#    e3 = containTimeout(ijk)
#    s = False
#    if e1 or e2 or e3:
#        s = ('[ER]' if e1 else '[--]')+('[WN]' if e2 else '[--]')+('[TC]' if e3 else '[--]')
#    return s

## --------------------------------------------------------------------------------->   
#def read_csv(file_name):
##     from main import importer
##     importer(['S'], locals())
#
#    # Check Py Version:
#    from inspect import signature
#    if ('on_bad_lines' in signature(pd.read_csv).parameters):
#        data = pd.read_csv(file_name, header = None, delimiter='-=-', engine='python', on_bad_lines='skip')
#    else:
#        data = pd.read_csv(file_name, header = None, delimiter='-=-', engine='python', error_bad_lines=False, warn_bad_lines=False)
#    data.columns = ['content']
#    return data
#
#def get_lines_with_separator(data, str_splitter):
#    lines_with_separation = []
#    for index,row in data.iterrows():#
#        if str_splitter in row['content']:
##             print(row)
#            lines_with_separation.insert(len(lines_with_separation), index)
#    return lines_with_separation
#
#def get_titles(data):
#    titles = []
#    for index,row in data.iterrows():#
#        if "Loading train and test data from" in row['content']:
#            titles.insert(len(titles), row['content'])
#    return titles
#
#def split_df_to_dict(data, lines_with_separation):
#    df_dict = {}
#    lines_with_separation.pop(0)
#    previous_line = 0
#    for line in lines_with_separation:#
##         print(data.iloc[previous_line:line,:])
#        df_dict[previous_line] = data.iloc[previous_line:line,:]
#        previous_line=line
#    df_dict['last'] = data.iloc[previous_line:,:]
#    return df_dict
#
#def get_total_number_of_candidates_file(str_target, df_dict):
#    total_per_file = []
#    for key in df_dict:
#        total = 0
#        for index,row in df_dict[key].iterrows():
#            if str_target in row['content']:
#                number = row['content'].split(str_target)[1]
#                total = total + int(number)
#        total_per_file.insert(len(total_per_file), total)
#    return total_per_file
#
#def get_total_number_of_candidates_file_by_dataframe(str_target, df):
#    total = 0
#    for index,row in df.iterrows():
#        if str_target in row['content']:
#            number = row['content'].split(str_target)[1]
#            number = number.split(".")[0]
#            total = total + int(number)
#    return total
#
#def get_sum_of_file_by_dataframe(str_target, df):
#    total = 0
#    for index,row in df.iterrows():
#        if str_target in row['content']:
#            number = row['content'].split(str_target)[1]
#            number = number.split(".")[0]
#            total = total + int(number)
#    return total
#
#def get_max_number_of_file_by_dataframe(str_target, df):
#    total = 0
#    for index,row in df.iterrows():
#        if str_target in row['content']:
#            number = row['content'].split(str_target)[1]
#            number = int(number.split(".")[0])
#            total = max(total, number)
#    return total
#
#def get_min_number_of_file_by_dataframe(str_target, df):
#    total = 99999
#    for index,row in df.iterrows():
#        if str_target in row['content']:
#            number = row['content'].split(str_target)[1]
#            number = int(number.split(".")[0])
#            total = min(total, number)
#    return total
#
#def get_total_number_of_ms(str_target, df):
#    total = 0
#    for index,row in df.iterrows():
#        if str_target in row['content']:
#            number = row['content'].split(str_target)[1]
#            number = number.split(" milliseconds")[0]
#            total = total + float(number)
#    return total
#
#def get_last_number_of_ms(str_target, df):
#    total = 0
#    for index,row in df.iterrows():
#        if str_target in row['content']:
#            number = row['content'].split(str_target)[1]
#            number = number.split(" milliseconds")[0]
#            total = float(number)
#    return total
#
#def get_first_number(str_target, df):
#    total = 0
#    for index,row in df.iterrows():
#        if str_target in row['content']:
#            number = row['content'].split(str_target)[1]
#            number = number.split(" ")[0]
#            return float(number)
#    return total
#    
#def get_sum_of_file_by_dataframe(str_target, df):
#    total = 0
#    for index,row in df.iterrows():
#        if str_target in row['content']:
#            number = row['content'].split(str_target)[1]
#            number = number.split(".")[0]
#            total = total + int(number)
#    return total
#    
#def get_count_of_file_by_dataframe(str_target, df):
#    total = 0
#    for index,row in df.iterrows():
#        if str_target in row['content']:
#            total = total + 1
#    return total

# ----------------------------------------------------------------------------------
# def split_string(string, delimiter):
#     return str(string.split(delimiter)[1])  

# OLDER FUNCTIONS: TODO Remake
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
