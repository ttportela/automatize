'''
Created on Jun, 2020

@author: Tarlis Portela
'''
# --------------------------------------------------------------------------------
# RUNNER
# import os
# import pandas as pd
# import glob2 as glob
# from datetime import datetime
# # from IPython.utils import io
from .main import importer #, display
importer(['S'], globals())
automatize_scripts = 'automatize/scripts'
# --------------------------------------------------------------------------------

def k_Movelets(k, data_folder, res_path, prefix, folder, descriptor, version=None, ms = False, Ms = False, extra=False, 
        java_opts='', jar_name='HIPERMovelets', n_threads=1, prg_path='./', print_only=False, keep_folder=2, 
        pyname='python3', timeout=None, impl=3):
#     from ..main import importer
#     importer(['S'], locals())
        
    if isinstance(k, int):
        k = range(1, k+1)
    
    for x in k:
        subpath_data = os.path.join(data_folder, 'run'+str(x))
        subpath_rslt = os.path.join(res_path,    prefix, 'run'+str(x))
        Movelets(subpath_data, subpath_rslt, None, folder, descriptor, version, ms, Ms, extra, 
        java_opts, jar_name, n_threads, prg_path, print_only, keep_folder, pyname, timeout, impl)

# --------------------------------------------------------------------------------
def Movelets(data_folder, res_path, prefix, folder, descriptor, version=None, ms = False, Ms = False, extra=False, 
        java_opts='', jar_name='HIPERMovelets', n_threads=1, prg_path='./', print_only=False, keep_folder=2,
        pyname='python3', timeout=None, impl=3):
#     from ..main import importer
#     importer(['S'], locals())
    
#     print('# --------------------------------------------------------------------------------------')
    print('# ' + res_path + ' - ' +folder)
    print('# --------------------------------------------------------------------------------------')
    
    if prefix != None:
        res_folder = os.path.join(res_path, prefix, folder)
    else:
        res_folder = os.path.join(res_path, folder)
    mkdir(res_folder, print_only)
    
    program = os.path.join(prg_path, jar_name+'.jar')
    outfile = os.path.join(res_folder, folder+'.txt')
    
    CMD = '-nt %s' % str(n_threads)
    
    if os.path.sep not in descriptor:
        descriptor = os.path.join(data_folder, descriptor)
    
    if impl == 1:
        CMD = CMD + ' -q LSP -p false'
    elif impl == 2:
        CMD = CMD + ' -ed true -samples 1 -sampleSize 0.5 -medium "none" -output "discrete" -lowm "false"'
    elif impl > 2 and version:
        CMD = CMD + ' -version ' + version

    CMD = 'java '+java_opts+' -jar "'+program+'" -curpath "'+data_folder+'" -respath "'+res_folder+'" -descfile "'+ descriptor + '.json" ' + CMD
    
    if timeout:
        CMD = 'timeout '+timeout+' '+CMD
    
    if ms != False:
        CMD = CMD + ' -ms '+str(ms)
    elif impl > 1:
        CMD = CMD + ' -ms -1'
        
    if Ms != False:
        CMD = CMD + ' -Ms '+str(Ms)
        
    if extra != False:
        CMD = CMD + ' ' + extra
        
#     if PVT:
#         CMD = CMD + ' -pvt true -lp false -pp 10 -op false'
        
    if os.name == 'nt':
        CMD = CMD +  ' >> "'+outfile +  '"'
    else:
        CMD = CMD +  ' 2>&1 | tee -a "'+outfile+'" '
        
#     print('# --------------------------------------------------------------------------------------')
    execute(CMD, print_only)
    
    dir_path = "MASTERMovelets"
    if keep_folder >= 1: # keep_folder = 1 or 2
        if impl > 1:
            mergeAndMove(res_folder, dir_path, prg_path, print_only, pyname)
        else:
            prg = os.path.join(prg_path, automatize_scripts, 'MoveDatasets.py')
            execute(pyname+' "'+prg+'" "'+res_folder+'"', print_only)
    
    if keep_folder <= 1: # keep_folder = 0 or 1, 1 for both
        execute('rm -R "'+os.path.join(res_folder, dir_path)+'"', print_only)
        
#     print('# --------------------------------------------------------------------------------------')
#     print()
    
# --------------------------------------------------------------------------------------
def execute(cmd, print_only=False):
    if print_only:
        print(cmd)
        print()
    else:
        print(os.popen(cmd).read())
#         os.system(cmd)
    
def mkdir(folder, print_only=False):
#     from ..main import importer
#     importer(['S'], locals())
    
    cmd = 'md' if os.name == 'nt' else 'mkdir -p'
    if not os.path.exists(folder):
        if print_only:
            execute(cmd+' "' + folder + '"', print_only)
        else:
            os.makedirs(folder)

def move(ffrom, fto, print_only=False):
    execute('mv "'+ffrom+'" "'+fto+'"', print_only)
    
def getResultPath(mydir):
    for dirpath, dirnames, filenames in os.walk(mydir):
        if not dirnames:
            dirpath = os.path.abspath(os.path.join(dirpath,".."))
            return dirpath
    
def moveResults(dir_from, dir_to, print_only=False):
#     from ..main import importer
#     importer(['S'], locals())
    
    csvfile = os.path.join(dir_from, "train.csv")
    move(csvfile, dir_to, print_only)
    csvfile = os.path.join(dir_from, "test.csv")
    move(csvfile, dir_to, print_only)
    
def mergeClasses(res_folder, prg_path='./', print_only=False, pyname='python3'):
#     from ..main import importer
#     importer(['S'], locals())
    
    dir_from = getResultPath(res_folder)
    
    if print_only:
        dir_from = res_folder

    prg = os.path.join(prg_path, automatize_scripts, 'MergeDatasets.py')
    execute(pyname+' "'+prg+'" "'+res_folder+'"', print_only)
#     execute('python3 "'+prg+'" "'+res_folder+'" "test.csv"', print_only)

    return dir_from
    
def mergeAndMove(res_folder, folder, prg_path='./', print_only=False, pyname='python3'):
    dir_from = mergeClasses(res_folder, prg_path, print_only, pyname)
    
    if not print_only and not dir_from:
        print("Nothing to Merge. Abort.")
        return

# --------------------------------------------------------------------------------------
def mergeDatasets(dir_path, file='train.csv'):
#     from ..main import importer
    importer(['S', 'glob'], globals())
    
    files = [i for i in glob.glob(os.path.join(dir_path, '*', '**', file))]

    print("Loading files - " + file)
    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f).drop('class', axis=1) for f in files[:len(files)-1]], axis=1)
    combined_csv = pd.concat([combined_csv, pd.read_csv(files[len(files)-1])], axis=1)
    #export to csv
    print("Writing "+file+" file")
    combined_csv.to_csv(os.path.join(dir_path, file), index=False)
    
    print("Done.")

# --------------------------------------------------------------------------------------
def countMovelets(dir_path):
#     from ..main import importer
#     importer(['S'], locals())
    
    ncol = 0
    print(os.path.join(dir_path, "**", "train.csv"))
    for filenames in glob.glob(os.path.join(dir_path, "**", "train.csv"), recursive = True):
#         print(filenames)
        with open(filenames, 'r') as csv:
            first_line = csv.readline()

        ncol += first_line.count(',')# + 1 
    return ncol

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
def k_MARC(k, data_folder, res_path, prefix, folder, train="train.csv", test="test.csv",
            EMBEDDING_SIZE=100, MERGE_TYPE="concatenate", RNN_CELL="lstm",
            prg_path='./', print_only=False, pyname='python3', extra_params=None):
#     from ..main import importer
#     importer(['S'], locals())
        
    if isinstance(k, int):
        k = range(1, k+1)
    
    for x in k:
        subpath_data = os.path.join(data_folder, 'run'+str(x))
        subpath_rslt = os.path.join(res_path,    prefix, 'run'+str(x))
        MARC(subpath_data, subpath_rslt, None, folder, train, test,
            EMBEDDING_SIZE, MERGE_TYPE, RNN_CELL,
            prg_path, print_only, pyname, extra_params)
    
def MARC(data_folder, res_path, prefix, folder, train="train.csv", test="test.csv",
            EMBEDDING_SIZE=100, MERGE_TYPE="concatenate", RNN_CELL="lstm",
            prg_path='./', print_only=False, pyname='python3', extra_params=None):
#     from ..main import importer
    importer(['S', 'datetime'], globals())
        
#     print("# ---------------------------------------------------------------------------------")
    print("# MARC: " + res_path + ' - ' +folder)
    print("# ---------------------------------------------------------------------------------")
#     print('echo MARC - ' + res_path + ' - ' +folder)
    print()
    
    if prefix != None:
        res_folder = os.path.join(res_path, prefix, folder)
    else:
        res_folder = os.path.join(res_path, folder)
#     res_folder = os.path.join(res_path, prefix, folder)
    mkdir(res_folder, print_only)
    
    TRAIN_FILE   = os.path.join(data_folder, train)
    TEST_FILE    = os.path.join(data_folder, test)
    DATASET_NAME = folder
    RESULTS_FILE = os.path.join(res_folder, folder + "_results.csv")
    OUTPUT_FILE  = '"' + os.path.join(res_folder, folder+'.txt') + '"'
    
#     mkdir(os.path.join(res_path, prefix), print_only)
        
    PROGRAM = os.path.join(prg_path, 'multi_feature_classifier.py')
    CMD = pyname+' "'+PROGRAM+'" "' + TRAIN_FILE + '" "' + TEST_FILE + '" "' + RESULTS_FILE + '" "' + DATASET_NAME + '" ' + str(EMBEDDING_SIZE) + ' ' + MERGE_TYPE + ' ' + RNN_CELL + ((' ' + extra_params) if extra_params else '')
    
    if os.name == 'nt':
        tee = ' >> '+OUTPUT_FILE 
    else:
        tee = ' 2>&1 | tee -a '+OUTPUT_FILE
        
    CMD = CMD + tee    
        
    if print_only:
        print('ts=$(date +%s%N)')
        print(CMD)
        print('tt=$((($(date +%s%N) - $ts)/1000000))')
        print('echo "Processing time: $tt milliseconds\\r\\n"' + tee)
    else:
        print(CMD)
        time = datetime.now()
        out = os.popen(CMD).read()
        time = (datetime.now()-time).total_seconds() * 1000

        f=open(OUTPUT_FILE, "a+")
        f.write(out)
        f.write("Processing time: %d milliseconds\r\n" % (time))
        f.close()

        print(captured.stdout)
        print("Done. " + str(time) + " milliseconds")
    print("# ---------------------------------------------------------------------------------")
    
def k_POIFREQ(k, data_folder, res_path, prefix, dataset, sequences, features, method='npoi', pyname='python3', \
              print_only=False, doclass=True):
#     from ..main import importer
#     importer(['S'], locals())
        
    if isinstance(k, int):
        k = range(1, k+1)
    
    for x in k:
        subpath_data = os.path.join(data_folder, 'run'+str(x))
        subpath_rslt = os.path.join(res_path,    prefix, 'run'+str(x))
#         print(subpath_data, subpath_rslt, None, dataset, sequences, features, py_name, print_only, doclass)
        POIFREQ(subpath_data, subpath_rslt, None, dataset, sequences, features, method, pyname, print_only, doclass)
        
def POIFREQ(data_folder, res_path, prefix, dataset, sequences, features, method='npoi', pyname='python3', print_only=False, doclass=True, or_methodsuffix=None):
#     from ..main import importer
#     importer(['S'], locals())
        
    ds_var = or_methodsuffix if or_methodsuffix else dataset
    result_name =  ('_'.join(features)) +'_'+ ('_'.join([str(n) for n in sequences]))
    folder = method.upper()+'-'+result_name +'-'+ ds_var
    
#     print("# ---------------------------------------------------------------------------------")
    print("# "+method.upper()+": " + res_path + ' - ' +folder)
    print("# ---------------------------------------------------------------------------------")
    print()
    
    if prefix != None:
        res_folder = os.path.join(res_path, prefix, folder)
    else:
        res_folder = os.path.join(res_path, folder)
        
    mkdir(res_folder, print_only)
    
    if print_only:
        outfile = os.path.join(res_folder, folder+'.txt')
    
        # RUN:
        CMD = pyname + " automatize/pois/POIS.py "
        CMD = CMD + "\""+method+"\" "
        CMD = CMD + "\""+(','.join([str(n) for n in sequences]))+"\" "
        CMD = CMD + "\""+(','.join(features))+"\" "
        CMD = CMD + "\""+dataset+"\" "
        CMD = CMD + "\""+data_folder+"\" "
        CMD = CMD + "\""+res_folder+"\""
        
        if os.name == 'nt':
            CMD = CMD +  ' >> "'+outfile+'"'
        else:
            CMD = CMD +  ' 2>&1 | tee -a "'+outfile+'"'
        
        execute(CMD, print_only)
        
        result_file = os.path.join(res_folder, method+'_'+result_name)#+'_'+ds_var)
        
        # Classification:
        if doclass:
            for s in sequences:
                pois = ('_'.join(features))+'_'+str(s)
                print(pyname+' automatize/pois/POIS-Classifier.py "'+method+'" "'+pois+'" "'+res_folder+'" "'+method.upper()+'-'+pois+'"')
                
            pois = ('_'.join(features))+'_'+('_'.join([str(n) for n in sequences]))
            print(pyname+' automatize/pois/POIS-Classifier.py "'+method+'" "'+pois+'" "'+res_folder+'" "'+method.upper()+'-'+pois+'"')
        
        return result_file
    else:
#         from ..main import importer
        importer(['poifreq'], globals())
    
#         from automatize.ensemble_models.poifreq import poifreq
        return poifreq(sequences, dataset, features, data_folder, res_folder, method=method, doclass=doclass)

def k_Ensemble(k, data_path, results_path, prefix, ename, methods=['movelets','poifreq'], \
             modelfolder='model', save_results=True, print_only=False, pyname='python3', \
             descriptor='', sequences=[1,2,3], features=['poi'], dataset='specific', num_runs=1,\
             movelets_line=None, poif_line=None, movelets_classifier='nn'):
#     from ..main import importer
#     importer(['S'], locals())
        
    if isinstance(k, int):
        k = range(1, k+1)
    
    for x in k:
        subpath_data = os.path.join(data_path, 'run'+str(x))
        subpath_rslt = os.path.join(results_path, prefix, 'run'+str(x))
        
        Ensemble(subpath_data, subpath_rslt, prefix, ename, methods, \
             modelfolder, save_results, print_only, pyname, \
             descriptor, sequences, features, dataset, num_runs,\
             movelets_line.replace('&N&', str(x)), poif_line.replace('&N&', str(x)), movelets_classifier)
    
def Ensemble(data_path, results_path, prefix, ename, methods=['master','npoi','marc'], \
             modelfolder='model', save_results=True, print_only=False, pyname='python3', \
             descriptor='', sequences=[1,2,3], features=['poi'], dataset='specific', num_runs=1,\
             movelets_line=None, poif_line=None, movelets_classifier='nn'):
    
    ensembles = dict()
    for method in methods:
        if method == 'poi' or method == 'npoi' or method == 'wnpoi':
            if poif_line is None:
                prefix = ''
                core_name = POIFREQ(data_path, results_path, prefix, 'specific', sequences, features, \
                                    print_only=print_only, doclass=False, pyname=pyname)
                ensembles['npoi'] = core_name
            else:
                ensembles['npoi'] = poif_line
            
        elif method == 'marc':
            ensembles['marc'] = data_path
            
        elif method == 'rf':
            ensembles['rf'] = data_path
            
        elif method == 'rfhp':
            ensembles['rfhp'] = data_path
            
        else: # the method is 'movelets':
            if movelets_line is None:
#                 from automatize.run import Movelets
                mname = method.upper()+'L-'+dataset
                prefix = ''
                Movelets(data_path, results_path, prefix, mname, descriptor, Ms=-3, \
                         extra='-T 0.9 -BU 0.1 -version '+method, \
                         print_only=print_only, jar_name='TTPMovelets', n_threads=4, java_opts='-Xmx60G', pyname=pyname)
                ensembles['movelets_'+movelets_classifier] = os.path.join(results_path, prefix, mname)
            else:
                ensembles['movelets_'+movelets_classifier] = movelets_line
#                 movelets_classifier IS nn OR mlp
                     
    if print_only:
        if num_runs == 1:
            CMD = pyname + " "+automatize_scripts+"/Classifier-Ensemble.py "
            CMD = CMD + "\""+data_path+"\" "
            CMD = CMD + "\""+os.path.join(results_path, ename)+"\" "
            CMD = CMD + "\""+str(ensembles)+"\" "
            CMD = CMD + "\""+dataset+"\" "
            CMD = CMD + "\""+modelfolder+"\" "
            CMD = CMD + ' 2>&1 | tee -a "'+os.path.join(results_path, ename, modelfolder+'.txt')+'" '
            print(CMD)
            print('')
        else:
            for i in range(1, num_runs+1): # TODO: set a different random number in python
                print('# Classifier Ensemble run-'+str(i))
#                 print('mkdir -p "'+os.path.join(results_path, postfix, modelfolder))
                CMD = pyname + " "+automatize_scripts+"/Classifier-Ensemble.py "
                CMD = CMD + "\""+data_path+"\" "
                CMD = CMD + "\""+os.path.join(results_path, ename)+"\" "
                CMD = CMD + "\""+str(ensembles)+"\" "
                CMD = CMD + "\""+dataset+"\" "
                CMD = CMD + "\""+modelfolder+'-'+str(i)+"\" "
                CMD = CMD + ' 2>&1 | tee -a "'+os.path.join(results_path, ename, modelfolder+'-'+str(i)+'.txt')+'" '
#                 CMD = CMD + " 2>&1 | tee -a \""+os.path.join(results_path, postfix, modelfolder, 'EC_results-'+modelfolder+'-'+str(i)+'.txt')+"\" "
                print(CMD)
                print('')
    else:
#         from ..main import importer
        importer(['ClassifierEnsemble'], globals())
        
        return ClassifierEnsemble(data_path, results_path, ensembles, dataset, save_results, modelfolder)