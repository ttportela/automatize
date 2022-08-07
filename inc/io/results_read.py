# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Aug, 2022
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
import sys, os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from main import importer #, display
importer(['S', 'glob', 're'], globals())

class ResultConfig:
    MARC = 'MARC'
    POIS = 'POIS'
    TEC  = 'TEC'
    MC   = 'MC'
    TC   = 'TC'
    
    # STATIC methods
    def getType(file):
        if file.endswith('poifreq_results.txt'):
            return ResultConfig.POIS
        elif 'MARC' in file:
            return ResultConfig.MARC
        elif (file.endswith('model_approachEnsemble_history.csv') or 
                               'TEC' in os.path.basename(os.path.dirname(file)).split('-')[0]):
            return ResultConfig.TEC
        elif os.path.basename(file).startswith('eval_') or \
             os.path.basename(os.path.dirname(file)).split('-')[0] in ['TRF', 'TXGB', 'TULVAE', 'BITULER', 'DST']:
            return ResultConfig.TC
        else: # Other type of files can be 
            return ResultConfig.MC

    def instantiate(file):
        m_type = ResultConfig.getType(file)
        if m_type == ResultConfig.MC:
            return MC(file)
        elif m_type == ResultConfig.MARC:
            return MARC(file)
        elif m_type == ResultConfig.POIS:
            return POIS(file)
        elif m_type == ResultConfig.TEC:
            return TEC(file)
        elif m_type == ResultConfig.TC:
            return TC(file)
        else:
            return None
    
    # Abstract methods:
    def decodeURL(self, ijk):
        raise NotYetImplemented('Implement abstract method decodeURL')
    def acc(self):
        raise NotYetImplemented('Implement abstract method acc')
    def runtime(self):
        raise NotYetImplemented('Implement abstract method runtime')
    def clstime(self):
        raise NotYetImplemented('Implement abstract method clstime')
    def totaltime(self):
        raise NotYetImplemented('Implement abstract method totaltime')
        
    def classification(self):
        return [ ['NN', self.acc()*100, self.clstime()] ]
        
    def metrics(self, list_stats, show_warnings=True):
        return get_stats(self, list_stats, show_warnings=show_warnings)
    
    # Class generic methods
    def __init__(self, file):
        #self.type = self.getMethod(file)
        self.run, self.random, self.method, self.subset, self.subsubset, self.prefix, self.model, \
        self.path, self.file, self.statsf = self.decodeURL(file)

        if not self.random.isdigit():
            self.random = 1
    
    @property
    def tipe(self):
        return type(self).__name__
    
    def __hash__(self):
        return hash((str(self.file), str(self.statsf)))
    
    def __eq__(self, other):
        return other and self.__hash__() == other.__hash__()
    
    def __lt__(self, other):
        if not other:
            return -1
        return str(self.statsf) < str(other.statsf)
    
    def __str__ (self):
        return ' '.join(['type:',     self.tipe, 
                        'run:',       self.run, 
                        'random:',    str(self.random), 
                        'method:',    self.method, 
                        'subset:',    self.subset, 
                        'subsubset:', self.subsubset, 
                        'prefix:',    self.prefix, 
                        'model:',     str(self.model) + '\n',
                        'path:',      str(self.path) + '\n', 
                        'file:',      str(self.file) + '\n', 
                        'statsf:',    str(self.statsf)])
    
    def isType(self, tipe):
        return tipe and self.tipe == tipe
    
    # ------------------------------------------------------------
    def containErrors(self):
        if self.file is None:
            return False
        txt = open(self.file, 'r').read()
        return txt.find('Error: ') > -1 or txt.find('Traceback') > -1 
    def containWarnings(self):
        if self.file is None:
            return False
        txt = open(self.file, 'r').read()
        return txt.find('UndefinedMetricWarning:') > -1 or txt.find('Could not load dynamic library') > -1
    def containTimeout(self):
        if self.file is None:
            return False
        txt = open(self.file, 'r').read()
        return txt.find('Processing time:') < 0
    
    def runningProblems(self):
        e1 = self.containErrors()
        e2 = self.containWarnings()
        e3 = self.containTimeout()
        s = False
        if e1 or e2 or e3:
            s = ('[ER]' if e1 else '[--]')+('[WN]' if e2 else '[--]')+('[TC]' if e3 else '[--]')
        return s
    
    def read_log(self, file=None):
        file = file if file else self.file
        if not file:
            return None
        # Check Py Version:
        from inspect import signature
        if ('on_bad_lines' in signature(pd.read_csv).parameters):
            data = pd.read_csv(file, header = None, delimiter='-=-', engine='python', on_bad_lines='skip')
        else:
            data = pd.read_csv(file, header = None, delimiter='-=-', engine='python', error_bad_lines=False, warn_bad_lines=False)
        data.columns = ['content']
        return data
    
        
class MARC(ResultConfig):
    
    def decodeURL(self, ijk):
        rpos = ijk.find('run')
        path = ijk[:ijk.find(os.path.sep, rpos+5)]

        if ijk.endswith('_results.csv'):
            statsf = ijk
            files = glob.glob(os.path.join(path, '*-*.txt'))
            ijk = files[0] if len(files) > 0 else None
        else:
            files = glob.glob(os.path.join(path, '*_results.csv'))
            if len(files) > 0:
                statsf = files[0]
            else:
                statsf = None

        model = os.path.dirname(statsf if statsf else ijk)
        model = model[model.rfind(os.path.sep)+1:]

        files = glob.glob(os.path.join(path, '**', '*_results.csv'), recursive=True)
        statsf = files[0] if len(files) > 0 else None

        method = path[path.rfind(os.path.sep)+1:]
        subset = method.split('-')[-1]
        method = method.split('-')[0]

        run = path[rpos:rpos+4]
        run = (run)[3:]

        prefix = os.path.basename(path[:rpos-1])

        subsubset = subset

        random = '1' if '-' not in model else model.split('-')[-1]

        return run, random, method, subset, subsubset, prefix, model, path, ijk, statsf
    
    def acc(self):
        if not hasattr(self, '_acc'):
            self._acc = 0
            if self.statsf:
                data = pd.read_csv(self.statsf)
                self._acc = data['test_acc'].iloc[-1]

        return self._acc
    
    def runtime(self):
        if not hasattr(self, '_time'):
            self._time = get_last_number_of_ms('Processing time: ', self.read_log()) 
        return self._time 
    def clstime(self):
        return self.runtime()
    def totaltime(self):
        return self.runtime()

    
class POIS(ResultConfig):
        
    def decodeURL(self, ijk):
        rpos = ijk.find('run')
        path = ijk[:ijk.find(os.path.sep, rpos+5)]

        model = os.path.dirname(ijk)
        model = model[model.rfind(os.path.sep)+1:]

        statsf = ijk
        if ijk.endswith('poifreq_results.txt'):
            files = glob.glob(os.path.join(path, os.path.basename(path) + '.txt'))
            ijk = files[0] if len(files) > 0 else None

        method = path[path.rfind(os.path.sep)+1:]
        subset = method.split('-')[-1]
        method = method.split('-')[0]

        run = path[rpos:rpos+4]
        run = (run)[3:]

        prefix = os.path.basename(path[:rpos-1])

        subsubset = model.split('-')[1] 
        method = method+ '_' + subsubset[re.search(r"_\d", subsubset).start()+1:]

        random = '1' if model.count('-') <= 1 else model.split('-')[-1]

        return run, random, method, subset, subsubset, prefix, model, path, ijk, statsf 
    
    def acc(self):
        if not hasattr(self, '_acc'):
            self._acc = 0
            if self.statsf:
                data = self.read_log(self.statsf)
                self._acc = get_first_number("Acc: ", data)

        return self._acc
    
    def runtime(self):
        if not hasattr(self, '_time'):
            self._time = get_first_number("Processing time: ", self.read_log())
        return self._time 
    
    def clstime(self):
        return get_first_number("Classification Time: ", self.read_log(self.statsf))
    
    def totaltime(self, show_warnings=True):
        timeRun = self.runtime()
        timeAcc = self.clstime()
        if show_warnings and (timeRun <= 0 or timeAcc <= 0):
            print('*** Warning ***', 'timeRun:', timeRun, 'timeAcc:', timeAcc, 'for ' + str(self.statsf) + '.')
        return timeRun + timeAcc

    
class TEC(ResultConfig):
        
    def decodeURL(self, ijk):
        rpos = ijk.find('run')
        path = ijk[:ijk.find(os.path.sep, rpos+5)]

        if ijk.endswith('model_approachEnsemble_history.csv'):
            statsf = ijk
            files = glob.glob(os.path.dirname(statsf) + '.txt')
            ijk = files[0] if len(files) > 0 else None  
        else:
            files = glob.glob(os.path.join(ijk.replace('.txt', ''), 'model_approachEnsemble_history.csv'))
            statsf = files[0] if len(files) > 0 else None
        
        if statsf:
            model = os.path.basename(os.path.dirname(statsf))
        else:
            model = os.path.basename(ijk)[:-4]

        method = path[path.rfind(os.path.sep)+1:]
        subset = method.split('-')[-1]
        method = method.split('-')[0]

        run = path[rpos:rpos+4]
        run = (run)[3:]

        prefix = os.path.basename(path[:rpos-1])

        subsubset = subset

        method += '_' + model.split('_')[-1]

        random = '1' if '-' not in model else model.split('-')[-1]

        return run, random, method, subset, subsubset, prefix, model, path, ijk, statsf
    
    def readMetric(self, classifier='TEC', metric='accuracy'):
        val = 0
        if self.statsf:
            data = pd.read_csv(self.statsf, index_col=0)
            data = data.set_index('classifier')
            
            if classifier == 'TEC':
                key = 'TEC' if 'TEC' in data.index else 'EnsembleClassifier' # TODO 'EnsembleClassifier' deprecated
            elif classifier == 'TEC.NN':
                key = 'modelets_nn'
            #elif classifier == 'TEC.MLP':
            #    key = 'movelets_mlp'
            else:
                key = classifier.replace('TEC.', '').lower()
            
            if key in data.index and metric in data.columns:
                val = data[metric][key]

        return val
    
    def accTEC(self):
        if not hasattr(self, '_acc'):
            self._acc = self.readMetric(metric='accuracy')
        return self._acc
    
    def acc(self, classifier='TEC'):
        if classifier in ['TEC', 'MLP']:
            return self.accTEC()
        else:
            return self.readMetric(classifier, metric='accuracy')
    
    def runtime(self):
        if not hasattr(self, '_time'):
            self._time = float(self.readMetric('TEC', metric='time'))
        return self._time
    
    def clstime(self, classifier='TEC'):
        if classifier == 'TEC':
            return self.runtime()
        else:
            return float(self.readMetric(classifier, metric='time'))
        
    def totaltime(self):
        return self.runtime()
    
    def classification(self):
        ls = [ ['NN', self.acc(), self.clstime()] ]
        for c in ['TEC.NN', 'TEC.MLP', 'TEC.POI', 'TEC.NPOI', 'TEC.WNPOI', 'TEC.MARC', 'TEC.RF', 'TEC.RFHP']:
            time = self.clstime(c)
            if time > 0:
                ls.append(['#'+c, self.acc(c), time])
                
        return ls

    
class MC(ResultConfig):
        
    def decodeURL(self, ijk):
        rpos = ijk.find('run')
        path = ijk[:ijk.find(os.path.sep, rpos+5)]

        if ijk.endswith('classification_times.csv'):
            statsf = ijk
            files = glob.glob(os.path.join(path, '*-*.txt'))
            ijk = files[0] if len(files) > 0 else None
        else:
            files = glob.glob(os.path.join(path, '*', 'classification_times.csv'))
            if len(files) > 0:
                statsf = files[0]
            else:
                statsf = None

        if statsf:
            model = os.path.dirname(statsf)
            model = model[model.rfind(os.path.sep)+1:]
        else:
            model = 'model'
            
        method = path[path.rfind(os.path.sep)+1:]
        subset = method.split('-')[-1]
        method = method.split('-')[0]

        run = path[rpos:rpos+4]
        run = (run)[3:]

        prefix = os.path.basename(path[:rpos-1])

        subsubset = subset

        random = '1' if '-' not in model else model.split('-')[-1]

        return run, random, method, subset, subsubset, prefix, model, path, ijk, statsf
    
    def containErrors(self):
        txt = open(self.file, 'r').read()
        return txt.find('java.') > -1 or txt.find('heap') > -1 or txt.find('error') > -1
    def containWarnings(self):
        txt = open(self.file, 'r').read()
        return txt.find('Empty movelets set') > -1
    def containTimeout(self):
        txt = open(self.file, 'r').read()
        return txt.find('[Warning] Time contract limit timeout.') > -1
    
    def read_approach(self, classifier):
        if classifier == 'RF':
            approach_file = 'model_approachRF300_history.csv'
        elif classifier == 'SVC':
            approach_file = 'model_approachSVC_history.csv'
        elif classifier == 'MLP':
            approach_file = 'model_approach2_history_Step5.csv'
        else:
            approach_file = 'classification_times.csv'
        
        res_file = os.path.join(self.path, self.model, approach_file)
        if os.path.isfile(res_file):
            data = pd.read_csv(res_file)
            return data
        else:
            return None

    def accRF(self):
        acc = 0
        data = self.read_approach('RF')
        if data is not None:
            acc = data['1'].iloc[-1]
        return acc

    def accSVM(self):
        acc = 0
        data = self.read_approach('SVC')
        if data is not None:
            acc = data.loc[0].iloc[-1]
        return acc

    def accMLP(self):
        if not hasattr(self, '_acc'):
            self._acc = 0
            data = self.read_approach('MLP')
            if data is not None:
                self._acc = data['val_accuracy'].iloc[-1]
        return self._acc
    
    def acc(self, classifier='MLP'):
        if classifier in ['MLP', 'NN']:
            return self.accMLP()
        elif classifier == 'RF':
            return self.accRF()
        elif classifier == 'SVM':
            return self.accSVM()
        else:
            return 0
    
    def runtime(self):
        if not hasattr(self, '_time'):
            self._time = get_last_number_of_ms('Processing time: ', self.read_log()) 
        return self._time
    
    def clstime(self, classifier='MLP'):
        data = self.read_approach(None)
        if data is not None:
            return data[classifier][0]
        return 0
    
    def totaltime(self, classifier='MLP', show_warnings=True):
        timeRun = self.runtime()
        timeAcc = self.clstime(classifier)
        if show_warnings and (timeRun <= 0 or timeAcc <= 0):
            print('*** Warning ***', 'timeRun:', timeRun, 'timeAcc:', timeAcc, classifier, 'for '+str(self.statsf)+'.')
        return timeRun + timeAcc
    
    def classification(self):
        ls = [ ['NN', self.acc(), self.clstime()] ]
        for c in ['RF', 'SVM']:
            time = self.clstime(c)
            if time > 0:
                ls.append([c, self.acc(c), time])
                
        return ls
    
class TC(ResultConfig): # TODO
        
    def decodeURL(self, ijk):
        rpos = ijk.find('run')
        path = ijk[:ijk.find(os.path.sep, rpos+5)]

        model = os.path.dirname(ijk)
        model = model[model.rfind(os.path.sep)+1:]

        if self.type not in [self.POIF, self.TEC, self.MARC]:
        #(self.isMethod(ijk, 'POIF') or self.isMethod(ijk, 'TEC') or self.isMethod(ijk, 'MARC')):
            files = glob.glob(os.path.join(path, '*', 'classification_times.csv'), recursive=True)
            statsf = files[0] if len(files) > 0 else None
            model = 'model'
        elif self.type == self.TEC:
            files = glob.glob(os.path.join(ijk.replace('.txt', ''), 'model_approachEnsemble_history.csv'), recursive=True)
            statsf = files[0] if len(files) > 0 else None  
            model = os.path.basename(ijk)[:-4]
        elif self.type == self.MARC:
            files = glob.glob(os.path.join(path, '**', '*_results.csv'), recursive=True)
            statsf = files[0] if len(files) > 0 else None
        else:
            statsf = ijk

        method = path[path.rfind(os.path.sep)+1:]
        subset = method.split('-')[-1]
        method = method.split('-')[0]

        run = path[rpos:rpos+4]
        run = (run)[3:]

        prefix = os.path.basename(path[:rpos-1])

        #if statsf:
        #    model = os.path.dirname(statsf)
        #    model = model[model.rfind(os.path.sep)+1:]

        if self.type == self.POIF:
            subsubset = model.split('-')[1] 
            method = method+ '_' + subsubset[re.search(r"_\d", subsubset).start()+1:]
        else:
            subsubset = subset

        if self.type == self.TEC:
            method += '_' + model.split('_')[-1]

        if self.type == self.POIF:
            random = '1' if model.count('-') <= 1 else model.split('-')[-1]
        else:
            random = '1' if '-' not in model else model.split('-')[-1]

        if not random.isdigit():
            random = 1

        return run, random, method, subset, subsubset, prefix, model, path, ijk, statsf



# ----------------------------------------------------------------------------------
def get_stats(config, list_stats, show_warnings=True):
#     from main import importer
#     importer(['S'], locals())

    resfile = config.file
    stsfile = config.statsf
    path = config.path
    method = config.method
    modelfolder = config.model

    stats = []
        
    fdata = config.read_log()
    #sdata = read_csv(stsfile)
    
    for x in list_stats:
        ssearch = x[2]+": "
        if x[1] == 'max':
            stats.append( get_max_number_of_file_by_dataframe(ssearch, fdata) )
        
        elif x[1] == 'min':
            stats.append( get_min_number_of_file_by_dataframe(ssearch, fdata) )
        
        elif x[1] == 'sum':
            stats.append( get_sum_of_file_by_dataframe(ssearch, fdata) )
        
        elif x[1] == 'count':
            stats.append( get_count_of_file_by_dataframe(ssearch, fdata) )
        
        elif x[1] == 'mean':
            a = get_sum_of_file_by_dataframe(ssearch, fdata)
            b = get_count_of_file_by_dataframe(ssearch, fdata)
            if b > 0:
                stats.append( a / b )
            else:
                stats.append( 0 )
        
        elif x[1] == 'first':
            stats.append( get_first_number(ssearch, fdata) )        
        elif x[1] == 'time':
            stats.append( config.runtime() )
        
        elif x[1] == 'accTime':
            if config.tipe == ResultConfig.MC or config.tipe == ResultConfig.TEC:
                acc = config.clstime(classifier=x[2])
            else:
                acc = config.clstime() 
                
        elif x[1] == 'totalTime':
            if config.tipe == ResultConfig.MC or config.tipe == ResultConfig.POIS:
                stats.append( config.totaltime(show_warnings=show_warnings) )
            else:
                stats.append( config.totaltime() )
        
        elif x[1] == 'ACC':
            if config.tipe == ResultConfig.MC or config.tipe == ResultConfig.TEC:
                acc = config.acc(classifier=x[2])
            else:
                acc = config.acc() 
                
            stats.append( acc * 100 )
        
        elif x[1] == 'msg':
            e = False
            if x[2] == 'msg':
                e = config.runningProblems()
            if x[2] == 'isMsg':
                e = config.runningProblems() != False
            elif x[2] == 'err':
                e = config.containErrors()
            elif x[2] == 'warn':
                e = config.containWarnings()
            elif x[2] == 'TC':
                e = config.containTimeout()
            
            stats.append(e)
            
        elif x[1] == 'endDate':
            try:
                importer(['dateparser'], globals())
    
                dtstr = fdata.iloc[-1]['content']
                stats.append( dateparser.parse(dtstr).timestamp() )
            except ValueError:
                stats.append( -1 )
                
    return stats

# ----------------------------------------------------------------------------------
def get_lines_with_separator(data, str_splitter):
    if data is None:
        return -1
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
    if df is None:
        return 0
    total = 0
    for index,row in df.iterrows():
        if str_target in row['content']:
            number = row['content'].split(str_target)[1]
            number = number.split(".")[0]
            total = total + int(number)
    return total

def get_sum_of_file_by_dataframe(str_target, df):
    if df is None:
        return 0
    total = 0
    for index,row in df.iterrows():
        if str_target in row['content']:
            number = row['content'].split(str_target)[1]
            number = number.split(".")[0]
            total = total + int(number)
    return total

def get_max_number_of_file_by_dataframe(str_target, df):
    if df is None:
        return 0
    total = 0
    for index,row in df.iterrows():
        if str_target in row['content']:
            number = row['content'].split(str_target)[1]
            number = int(number.split(".")[0])
            total = max(total, number)
    return total

def get_min_number_of_file_by_dataframe(str_target, df):
    if df is None:
        return 0
    total = 99999
    for index,row in df.iterrows():
        if str_target in row['content']:
            number = row['content'].split(str_target)[1]
            number = int(number.split(".")[0])
            total = min(total, number)
    return total

def get_total_number_of_ms(str_target, df):
    if df is None:
        return 0
    total = 0
    for index,row in df.iterrows():
        if str_target in row['content']:
            number = row['content'].split(str_target)[1]
            number = number.split(" milliseconds")[0]
            total = total + float(number)
    return total

def get_last_number_of_ms(str_target, df):
    if df is None:
        return 0
    total = 0
    for index,row in df.iterrows():
        if str_target in row['content']:
            number = row['content'].split(str_target)[1]
            number = number.split(" milliseconds")[0]
            total = float(number)
    return total

def get_first_number(str_target, df):
    if df is None:
        return 0
    total = 0
    for index,row in df.iterrows():
        if str_target in row['content']:
            number = row['content'].split(str_target)[1]
            number = number.split(" ")[0]
            return float(number)
    return total
    
def get_sum_of_file_by_dataframe(str_target, df):
    if df is None:
        return 0
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