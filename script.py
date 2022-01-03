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
                        
            scrpt = 'run5-'+mname+'-'+ds+'-'+var+'-'+THREADS + 'T'+'.sh'

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
        return (method.replace('+Log', '').replace('+TF', '').replace('+TR', '')\
                        .replace('-2', '').replace('-3', '').replace('-4', ''))
    
    
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
    elif 'indexed' in method:
        mname = 'IX'
        runopts = '-version ' + replacesuffix(method) + ' '
        
    elif 'master' in method:
        mname = 'M2'
            
        runopts = '-version ' + replacesuffix(method) + ' '
    elif 'poi' in method:
        mname = 'NPOI_'+('_'.join(params['features']))+'_'+('_'.join([str(n) for n in params['sequences']]))
        
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
        runopts += '-TF ' + ('0.5' if 'super' in method else '0.9') + ' '
    elif '+TR' in method:
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
        
    k = params['k']
    if isinstance(k, int):
        k = range(1, k+1)
    
    call_exit = params['call_exit']

    THREADS = str(params['threads'])
    GIG     = str(params['gig'])
    
    data = os.path.join(data, '${RUN}')
    results = os.path.join(results, prefix, '${RUN}')
    
    print('for RUN in '+ ' '.join(['"run'+str(x)+'"' for x in list(k)]) )
    print('do')
    print('DIR="'+results+'"')
    print('if [ -d "$DIR/'+mname+'-'+var+'" ]; then')
    print('   echo "${DIR}/'+mname+'-'+var+'... [OK]"')
    print('else')
    
    results = '${DIR}'
    
    if method == 'TEC' or method == 'TEC2':
        movelets_method = 'M2L'
        pois_method = 'NPOI'
        
        methods = ['movelets_nn', 'npoi', 'marc']
        if method == 'TEC2':
            methods = ['movelets_nn', 'marc']
            
        pois = ('_'.join(params['features']))+'_'+('_'.join([str(n) for n in params['sequences']]))
        poif_line     = os.path.join('${DIR}', pois_method+'_'+pois+'-'+var)
        movelets_line = os.path.join('${DIR}', movelets_method+'-'+var) # IF DIFFERENT METHOD, CHANGE modelfolder NAME!!
        
        metsuff = movelets_method+pois_method
        
        Ensemble(data, results, prefix, mname+'-'+var, methods=methods, \
             modelfolder='model_'+metsuff, save_results=True, print_only=print_only, pyname=pyname, \
             descriptor='', sequences=params['sequences'], features=params['features'], dataset='specific', num_runs=1,\
             movelets_line=movelets_line, poif_line=poif_line)

    prefix = None
    if method == 'MARC':
        MARC(data, results, prefix, mname+'-'+var, print_only=print_only, prg_path=os.path.join(prog_path, 'automatize','marc'), 
             pyname=pyname, extra_params=GIG+' '+THREADS)

    if 'poi' in method: #method == 'npoi' or method == 'poi' or method == 'wnpoi':
        POIFREQ(data, results, prefix, var, params['sequences'], params['features'], method, print_only=print_only, pyname=pyname)
        
    if 'SM' in method:                    
        Movelets(data, results, prefix, mname+'-'+var, json, \
                   Ms=islog, extra=runopts, n_threads=THREADS, prg_path=prog_path, \
                   print_only=print_only, jar_name='SUPERMovelets', java_opts='-Xmx'+GIG+'G', pyname=pyname)
    if 'super' in method:                    
        Movelets(data, results, prefix, mname+'-'+var, json+'_hp', \
                   Ms=islog, extra=runopts, n_threads=THREADS, prg_path=prog_path, \
                   print_only=print_only, jar_name='TTPMovelets', java_opts='-Xmx'+GIG+'G', pyname=pyname)

    if 'hiper' in method or 'ultra' in method or 'random' in method or 'indexed' in method: 
        Movelets(data, results, prefix, mname+'-'+var, json+'_hp', Ms=islog, extra=runopts, n_threads=THREADS, 
        prg_path=prog_path, print_only=print_only, jar_name='TTPMovelets', java_opts='-Xmx'+GIG+'G', pyname=pyname)

    if 'MM' in method:
        Movelets(data, results, prefix, mname+'-'+var,  json, Ms=islog, n_threads=THREADS, 
        prg_path=prog_path, print_only=print_only, jar_name='MASTERMovelets', java_opts='-Xmx'+GIG+'G', pyname=pyname)
    if 'master' in method:                    
        Movelets(data, results, prefix, mname+'-'+var, json+'_hp', \
                   Ms=islog, extra=runopts, n_threads=THREADS, prg_path=prog_path, \
                   print_only=print_only, jar_name='TTPMovelets', java_opts='-Xmx'+GIG+'G', pyname=pyname)


    if doacc and not(method == 'MARC' or 'poi' in method or 'TEC' in method):
        print('# --------------------------------------------------------------------------------------')
        print(pyname+' '+automatize_scripts+'/Classifier-All.py "'+results+'" "'+mname+'-'+var+'"') #MLP_RF
        print()
        
    print('echo "${DIR}/'+mname+'-'+var+' => Done."')
    if call_exit:
        print('exit 1')
    print('fi')
    print('done')
    print('# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ')
    
# --------------------------------------------------------------------------------------
def sh2bat(root, root_win, path, folder):
    # Convert sh to bat:
#     import os
#     import glob2 as glob
    importer(['glob'], globals())

    dict_strs = {
        '# --------------------------------------------------------------------------------------': '',
        'DIR=': 'SET DIRET=',
        'if [ -d "$DIR" ]; then': 'IF EXIST %DIRET% (',
        '${DIR}': '%DIRET%',
        'else': ') ELSE (',
        'fi\n': ')\n',
        '#!/bin/bash': '',
        '# ': 'ECHO ',
        'mkdir -p': 'md',
        '2>&1 | tee -a': '>>',
        'rm -R': 'ECHO FIM-',
        root: root_win,
        '/': '\\',
        'python3': 'python'
    }
    
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