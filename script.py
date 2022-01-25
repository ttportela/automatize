'''
Created on Feb, 2021

@author: Tarlis Portela
'''
from .main import importer #, display
importer(['S'], globals())

def gensh(method, datasets, params=None):
#     from automatize.run import Movelets, MARC, k_MARC
#     import os, sys
    importer(['sys'], globals())
    
    method, params, mname, runopts, islog, pname, THREADS, GIG, data_folder, res_path, prog_path = configMethod(method, params)
    # -----------------------------
    TEST_PATH   = (params['folder'] if 'folder' in params else mname)+'-5fold'+ '_' + THREADS + 'T_' + GIG+'G'
    DESC_PATH   = os.path.join(data_folder, 'descriptors')
    results     = os.path.join(res_path, TEST_PATH)
    # -----------------------------
    
    print_only = True
    f_name = ''

    for key in datasets:
        ds = key.split(sep='.')[0]
        desc = key.split(sep='.')[1]
        for var in datasets[key]:
                        
            prefix      = ds #.capitalize()
            data        = os.path.join(data_folder, ds)
            json        = os.path.join(DESC_PATH, desc)
            
            os.makedirs(params['sh_folder'], exist_ok=True)        
                        
            scrpt = 'run-'+mname+'-'+ds+'-'+var+'-'+THREADS + 'T'+'.sh'

            f_name += 'sh ' + scrpt + '\n'
            print('sh ' + scrpt)
            
            orig_stdout = sys.stdout
            f = open(os.path.join(params['sh_folder'], scrpt), 'w')
            sys.stdout = f
            print('#!/bin/bash')
            
            json = json + '_' + var
            
            var = prefix if ('useds' in params and params['useds']) else var
            
            print('BASE="'+params['root']+'"')
            printRun(method, data, results, prog_path, prefix, mname, var, json, params, runopts, islog, print_only, 
                     pyname=pname)
                
            print("# END - By Tarlis Portela")
            sys.stdout = orig_stdout
            f.close()
    return f_name

# -----------------------------------------------------------------------------------
def configMethod(method, params):
    if params is None:
        params = {'root': '../', \
                  'k': [1, 2, 3, 4, 5], \
                  'sh_folder': 'scripts', \
                  'threads': 4, \
                  'gig': 60, \
                  'call_exit': False \
                 }
    runopts = ''    
    def replacesuffix(method):
        return (method.replace('+Log', '').replace('+TF50', '').replace('+TR50', '').replace('+TF75', '').replace('+TR75', '')\
                        .replace('+TF', '').replace('+TR', '').replace('-2', '').replace('-3', '').replace('-4', ''))
    
    
    if 'hiper' in method:
        mname = 'HT' if 'hipert' in method else 'H'
        runopts = '-version ' + replacesuffix(method) + ' '

        if 'pivots' in method:
            mname += 'p'
        if 'ce' in method:
            mname += 'ce'
        if 'random' in method:
            mname += 'r'
        if 'entropy' in method:
            mname += 'en'
                   
    elif 'super' in method:
        mname = 'S2'
        if 'class' in method:
            mname = 'SC'
        runopts = '-version ' + replacesuffix(method) + ' '
                        
    elif 'ultra' in method:
        mname = 'U'
        runopts = '-version ' + replacesuffix(method) + ' '    
    elif 'random' in method:
        mname = 'R'
        runopts = '-version ' + replacesuffix(method) + ' ' 
    elif 'pivots' in method:
        mname = 'M2p'
        runopts = '-version ' + replacesuffix(method) + ' '
    elif 'indexed' in method:
        mname = 'IX'
        runopts = '-version ' + replacesuffix(method) + ' '
    elif 'master' in method:
        mname = 'M2'
        runopts = '-version ' + replacesuffix(method) + ' '
    elif 'poi' in method:
        mname = method.upper()+'-'+('_'.join(params['features']))+'_'+('_'.join([str(n) for n in params['sequences']]))
        
    elif 'MMp' in method:   
        mname = 'MMp'
        runopts = '-pvt true -lp false -pp 10 -op false'
    elif 'MM' in method:   
        mname = 'MM'
        runopts = ''
    elif 'SM' in method:   
        mname = 'SM'
        runopts = ''
    else:
        mname = method.replace('+Log', '')
       
            
    if '+TF' in method:
        mname += 'f'
        if '+TF50' in method:
            mname += 'TF50'
            runopts += '-TF 0.5 '
        elif '+TF75' in method:
            mname += 'TF75'
            runopts += '-TF 0.75 '
        else:
            runopts += '-TF ' + ('0.5' if 'super' in method else '0.9') + ' '
    elif '+TR' in method:
        if '+TR50' in method:
            mname += 'TR50'
            runopts += '-TR 0.5 '
        elif '+TR75' in method:
            mname += 'TR75'
            runopts += '-TR 0.75 '
        else:
            runopts += '-TR ' + ('0.5' if 'super' in method else '0.9') + ' '
       

    if '+Log' in method:
        islog = -3
        mname += 'L'
    else:
        islog=False


    if '-2' in method:
        mname += 'D2'
        if 'SM' in method:
            runopts += '-Al true '
        else:
            runopts += '-mnf -2 '
        
    if '-3' in method:
        mname += 'D3'
        runopts += '-mnf -3 '
    if '-4' in method:
        mname += 'D4'
        runopts += '-mnf -4 '
       
    if 'samples' in params.keys():
        runopts += '-fold '+str(params['samples'])+' '
    
    if 'runopts' in params.keys():
#         if '-TF' in params['runopts']:
#             mname += 'f'
        if not ( method.startswith('MM') or method.startswith('SM') ): 
            runopts += params['runopts'] + ' '
    
    if 'suffix' in params.keys():
        mname += params['suffix']
    
    pname   = params['pyname'] if 'pyname' in params else 'python3'
    THREADS = params['threads'] if 'threads' in params else 4
    GIG     = params['gig'] if 'gig' in params else 30

    data_folder = params['data_folder'] if 'data_folder' in params else os.path.join('${BASE}', 'data')
    res_path    = params['res_path'] if 'res_path' in params else os.path.join('${BASE}', 'results')
    prog_path   = os.path.join('${BASE}', 'programs')
        
    return method, params, mname, runopts, islog, pname, str(THREADS), str(GIG), data_folder, res_path, prog_path
    # -----------------------------
    
def printRun(method, data, results, prog_path, prefix, mname, var, json, params, runopts, islog, print_only, check_done=True, doacc=True, pyname='python3'):
#     import os, sys
#     from automatize.run import Movelets, MARC, POIFREQ, Ensemble#k_MARC, k_Ensemble
    importer(['methods'], globals())
    automatize_scripts = 'automatize/scripts'
        
    if 'k' in params and params['k']:
        k = params['k']
    else:
        k = None
        
    if isinstance(k, int):
        k = range(1, k+1)
    
    call_exit = params['call_exit']

    THREADS = str(params['threads'])
    GIG     = str(params['gig'])
    
    if k:
        data = os.path.join(data, '${RUN}')
        results = os.path.join(results, prefix, '${RUN}')
    else:
        results = os.path.join(results, prefix)
    
    if k:
        print('for RUN in '+ ' '.join(['"run'+str(x)+'"' for x in list(k)]) )
        print('do')
    print('DIR="'+results+'"')
    print('if [ -d "${DIR}/'+mname+'-'+var+'" ]; then')
    print('   echo "${DIR}/'+mname+'-'+var+'... [OK]"')
    print('else')
    
    results = '${DIR}'
    dsvar = 'specific' if '_ts' not in data else prefix
    if 'univariate_ts' in data:
        runopts += '-inprefix "' + prefix + '" '
    
#     if method == 'TEC' or method == 'TEC2':
    if 'TEC' in method:
        if 'ensemble_methods' in params:
            ensemble_methods = params['ensemble_methods']
        else:
            ensemble_methods = [['MML'], ['npoi']]
#         movelets_method = 'MML' if 'ensemble_methods' not in params else params['ensemble_methods'][0]
#         pois_method = 'npoi' if 'ensemble_methods' not in params else params['ensemble_methods'][1]
        print('mkdir -p "${DIR}/'+mname+'-'+var+ '"')
        print('')
        
        for movelets_method in ensemble_methods[0]:
            for pois_method in ensemble_methods[1]:
        
                methods = ['movelets_nn', 'npoi', 'marc']
                if method == 'TEC2':
                    methods = ['movelets_nn', 'marc']

                pois = ('_'.join(params['features']))+'_'+('_'.join([str(n) for n in params['sequences']]))
                poif_line     = os.path.join('${DIR}', pois_method.upper()+'-'+pois+'-'+var, pois_method+'_'+pois) #+'_'+var)
                movelets_line = os.path.join('${DIR}', movelets_method+'-'+var) # IF DIFFERENT METHOD, CHANGE modelfolder NAME!!

                metsuff = movelets_method+ (pois_method if method != 'TEC2' else '')
                
                Ensemble(data, results, prefix, mname+'-'+var, methods=methods, \
                     modelfolder='model_'+metsuff, save_results=True, print_only=print_only, pyname=pyname, \
                     descriptor='', sequences=params['sequences'], features=params['features'], dataset='specific', num_runs=1,\
                     movelets_line=movelets_line, poif_line=poif_line)

    prefix = None
    if method == 'MARC':
        train_file = dsvar+"_train.csv" if '_ts' not in data else dsvar+"_TRAIN.ts"
        test_file  = dsvar+"_test.csv"  if '_ts' not in data else dsvar+"_TEST.ts"
        MARC(data, results, prefix, mname+'-'+var, print_only=print_only, prg_path=os.path.join(prog_path, 'automatize','marc'), 
             pyname=pyname, extra_params=GIG+' '+THREADS, train=train_file, test=test_file)

    if 'poi' in method: #method == 'npoi' or method == 'poi' or method == 'wnpoi':
        POIFREQ(data, results, prefix, dsvar, params['sequences'], params['features'], method, print_only=print_only, pyname=pyname, or_methodsuffix='specific')
        
    if 'SM' in method:                    
        Movelets(data, results, prefix, mname+'-'+var, json, \
                   Ms=islog, extra=runopts, n_threads=THREADS, prg_path=prog_path, \
                   print_only=print_only, jar_name='SUPERMovelets', java_opts='-Xmx'+GIG+'G', pyname=pyname)
    if 'super' in method:                    
        Movelets(data, results, prefix, mname+'-'+var, json+'_hp', \
                   Ms=islog, extra=runopts, n_threads=THREADS, prg_path=prog_path, \
                   print_only=print_only, jar_name='TTPMovelets', java_opts='-Xmx'+GIG+'G', pyname=pyname)

    if 'hiper' in method or 'ultra' in method or 'random' in method or 'indexed' in method or method == 'pivots': 
        Movelets(data, results, prefix, mname+'-'+var, json+'_hp', Ms=islog, extra=runopts, n_threads=THREADS, 
        prg_path=prog_path, print_only=print_only, jar_name='TTPMovelets', java_opts='-Xmx'+GIG+'G', pyname=pyname)

    if 'MM' in method or 'MMp' in method:
        Movelets(data, results, prefix, mname+'-'+var,  json, Ms=islog, n_threads=THREADS, extra=runopts,
        prg_path=prog_path, print_only=print_only, jar_name='MASTERMovelets', java_opts='-Xmx'+GIG+'G', pyname=pyname)
    if 'master' in method:                    
        Movelets(data, results, prefix, mname+'-'+var, json+'_hp', \
                   Ms=islog, extra=runopts, n_threads=THREADS, prg_path=prog_path, \
                   print_only=print_only, jar_name='TTPMovelets', java_opts='-Xmx'+GIG+'G', pyname=pyname)


    if 'samples' in params and not(method == 'MARC' or 'poi' in method or 'TEC' in method):
        print('# --------------------------------------------------------------------------------------')
        print('for FOLD in '+ ' '.join(['"run'+str(x)+'"' for x in range(1, params['samples']+1)]) )
        print('do')
        print(pyname+' '+automatize_scripts+'/MergeDatasets.py "'+results+'/${FOLD}/'+mname+'-'+var+'"') #MERGE
        if doacc :
            print(pyname+' '+automatize_scripts+'/Classifier-MLP_RF.py "'+results+'/${FOLD}" "'+mname+'-'+var+'"') #MLP_RF
        print('done')
        print('# --------------------------------------------------------------------------------------')
        print()
        
    elif doacc and not(method == 'MARC' or 'poi' in method or 'TEC' in method):
        print('# --------------------------------------------------------------------------------------')
        print(pyname+' '+automatize_scripts+'/Classifier-MLP_RF.py "'+results+'" "'+mname+'-'+var+'"') #MLP_RF
        print()
        
    print('echo "${DIR}/'+mname+'-'+var+' => Done."')
    if call_exit:
        print('exit 1')
    print('fi')
    if k:
        print('done')
    print('# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ')
    
# --------------------------------------------------------------------------------------
def sh2bat(path, folder, root=None, root_win=None, py_name='python'):
    # Convert sh to bat:
    importer(['glob'], globals())

    dict_strs = {
        '# ': ':: ',
        '#!/bin/bash': '',
        'ts=$(date +%s%N)':'SET ts=%time%',
        'tt=$((($(date +%s%N) - $ts)/1000000))':'SET tt=%time%-%ts%',
        'Processing time: $tt':'Processing time: %tt%',
        'DIR="': 'SET DIRET=',
        'BASE="': 'SET BASE=',
        'RUN="': 'SET RUN=',
        '"\nSET': '\nSET',
        '"\nif': '\nif',
        'if [ -d ': 'IF EXIST ',
        ' ]; then': ' (',
        'else': ') ELSE (',
        'fi\n': ')\n',
#         '# ': 'ECHO ',
        'mkdir -p': 'md',
        '2>&1 | tee -a': '>>',
        'rm -R': 'ECHO FIM-',
        '/': '\\',
        'python3': py_name,
        'for RUN in ': 'FOR %%RUN IN (',
        'done' : ')',
        '\ndo' : ') DO (',
        '" "run' : ',run',
        '${DIR}': '%DIRET%',
        '${BASE}': '%BASE%',
        '${RUN}': '%RUN%',
#         '"': '',
    }
    
    if root and root_win:
        dict_strs[root] = root_win
    
    os.makedirs(os.path.join(path, folder+'_win'), exist_ok=True)
    for file in glob.glob(os.path.join(path, folder, '*.sh')):
        file_to = os.path.join(path, folder+'_win', os.path.basename(file)[:-3]+'.bat')
        fileReplace(dict_strs, file, file_to)

def fileReplace(dict_strs, file_from, file_to): 
    # Read in the file
    with open(file_from, 'r') as f :
        filedata = f.read()

    # Replace the target string
    for key, value in dict_strs.items():
        filedata = filedata.replace(key, value)

    # Write the file out again        
    with open(file_to, 'w') as f:
        f.write(filedata)