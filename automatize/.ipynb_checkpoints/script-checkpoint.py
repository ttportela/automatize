# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Feb, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
from .main import importer, pyshel, PACKAGE_NAME, DEFAULT_MC, DEFAULT_TC #, display
importer(['S'], globals())

def gensh(method, datasets, params=None, check_done=True, doacc=True):
#     from run import Movelets, MARC, k_MARC
#     import os, sys
    importer(['sys'], globals())
    
    method, params, mname, runopts, islog, pname, THREADS, GIG, data_folder, res_path, prog_path = configMethod(method, params)
    # -----------------------------
    TEST_PATH   = (params['folder'] if 'folder' in params else mname)+'-' + THREADS + 'T_' + GIG+'G'
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
            scrpt = scrpt.replace('/', '_')

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
                     check_done, doacc, pname)
                
            print("# Automatize - END generated script")
            sys.stdout = orig_stdout
            f.close()
    return f_name

# -----------------------------------------------------------------------------------
def oneof(method, ls):
    return any([x for x in ls if x in method])

def trimsuffix(method):
#     end = min(method.find('+') if '+' in method else len(method), method.find('-') if '-' in method else len(method))
    end = method.find('+') if '+' in method else len(method)
    return method[:end] #.replace('-2', '').replace('-3', '').replace('-4', '')
#     return (method.replace('+Log', '').replace('+TF50', '').replace('+TR50', '').replace('+TF75', '').replace('+TR75', '')\
#                     .replace('+TF', '').replace('+TR', '').replace('-2', '').replace('-3', '').replace('-4', ''))

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
    
    
    if 'hiper' in method:
        mname = 'H'
#         runopts = '-version ' + trimsuffix(method) + ' '

#         if '-pivots' in method:
#             mname += 'p'
# #         if 'ce' in method:
# #             mname += 'ce'
# #         if 'random' in method:
# #             mname += 'r'
# #         if 'entropy' in method:
# #             mname += 'en'
                        
    elif 'ultra' in method:
#         if 'ultra-wp' in method:
#             mname = 'Uwp'
#         else:
        mname = 'U'
#         runopts = '-version ' + trimsuffix(method) + ' '    
    elif 'random' in method:
        mname = 'R'
#         runopts = '-version ' + trimsuffix(method) + ' ' 

    elif 'super' in method:
        mname = 'S2'
#         if 'class' in method:
#             mname = 'SC'
#         runopts = '-version ' + trimsuffix(method) + ' '

    elif 'master' in method:
        mname = 'M2'
#         runopts = '-version ' + trimsuffix(method) + ' '
    elif 'pivots' in method:
        mname = 'M2p'
#         runopts = '-version ' + trimsuffix(method) + ' '

    elif 'indexed' in method:
        mname = 'IX'
#         runopts = '-version ' + trimsuffix(method) + ' '

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
        
    elif 'TEC' in method or 'MARC' in method or oneof(method, ['Movelets', 'Dodge', 'Xiao', 'Zheng']):
        mname = trimsuffix(method)
        
    elif 'TC-' in method:   
        mname = ''
        runopts = ''
    else:
        mname = trimsuffix(method).capitalize() # method.replace('+Log', '')
#         runopts = '-version ' + trimsuffix(method) + ' '

    if '-' in trimsuffix(method):
        suff = trimsuffix(method).split('-', 1)[1:]
        for s in suff:
            if s == 'pivots':
                mname += 'p'
#             elif s == 'ce':
#                 mname += 'ce'
            elif s == 'random':
                mname += 'r'
            elif s == 'entropy':
                mname += 'en'
            elif s == 'class':
                mname += 'C'
            else:
                mname += s


#     if '+TF' in method:
#         mname += 'f'
    if '+TF' in method:
        mname += method[method.find('+TF')+1:method.find('+TF')+5]
        runopts += '-TF 0.' + method[method.find('+TF')+3:method.find('+TF')+5] + ' '
    elif '+TR' in method:
        mname += method[method.find('+TR')+1:method.find('+TR')+5]
        runopts += '-TR 0.' + method[method.find('+TR')+3:method.find('+TR')+5] + ' '
    elif '+T' in method:
        mname += 'TF'+method[method.find('+T')+2:method.find('+T')+4]
        runopts += '-TF 0.' + method[method.find('+T')+2:method.find('+T')+4] + ' '
        
    if '+Log' in method:
        islog = -3
        mname += 'L'
    else:
        islog=False


    if '+2' in method:
        mname += 'D2'
        if 'SM' in method:
            runopts += '-Al true '
        else:
            runopts += '-mnf -2 '
        
    if '+3' in method:
        mname += 'D3'
        runopts += '-mnf -3 '
    if '+4' in method:
        mname += 'D4'
        runopts += '-mnf -4 '
        
    if '+Ms' in method:
        Ms = int(method[method.find('+Ms')+3:method.find('+Ms')+5])
        mname += 'S'+str(Ms)
        runopts += '-Ms ' + str(Ms) + ' '
        
    if '+ms' in method:
        Ms = int(method[method.find('+ms')+3:method.find('+ms')+5])
        mname += 's'+str(Ms)
        runopts += '-ms ' + str(Ms) + ' '
       
    if 'samples' in params.keys():
        runopts += '-fold '+str(params['samples'])+' '
    
    if 'runopts' in params.keys():
#         if '-TF' in params['runopts']:
#             mname += 'f'
        if not ( method.startswith('MM') or method.startswith('SM') ): 
            runopts += params['runopts'] + ' '
            
    if 'timeout' in params.keys() and not oneof(method, ['MM', 'SM', 'Movelets', 'Dodge', 'Xiao', 'Zheng']):
        runopts += '-TC ' + params['timeout'] + ' '
    
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
#     from run import Movelets, MARC, POIFREQ, Ensemble#k_MARC, k_Ensemble
    importer(['methods'], globals())
#     package_scripts = PACKAGE_NAME+'/scripts'
        
    if 'k' in params and params['k']:
        k = params['k']
    else:
        k = None
        
    if isinstance(k, int):
        k = range(1, k+1)
    
    call_exit = params['call_exit'] if 'call_exit' in params else False

    THREADS = str(params['threads'])
    GIG     = str(params['gig'])
    
    
    print('DATAPATH="'+data+'"')
    data = '${DATAPATH}'
    print('RESPATH="'+results+'"')
    results = '${RESPATH}'
    print('MNAME="'+mname+'-'+var+'"')
    MNAME = '${MNAME}'
    print()
    
    if k:
        data = os.path.join(data, '${RUN}')
    else:
        print('RUN="run0"')
        #results = os.path.join(results, prefix.replace('/', '_'), 'run0')
    results = os.path.join(results, prefix.replace('/', '_'), '${RUN}')
    
    if k:
        print('for RUN in '+ ' '.join(['"run'+str(x)+'"' for x in list(k)]) )
        print('do')
        
#     print('FOLDER="'+results+'"')
#     results = '${FOLDER}'
    print('DIR="'+results+'"')
#     print('DIR="'+ os.path.join('${FOLDER}', mname+'-'+var) +'"')
    results = '${DIR}'
    DIRF = os.path.join(results, MNAME)
#     print('if [ -d "${DIR}/'+mname+'-'+var+'" ]; then')
#     print('   echo "${DIR}/'+mname+'-'+var+'... [OK]"')
    if check_done:
        print('if [ -d "'+DIRF+'" ]; then')
        print('   echo "'+DIRF+' ... [OK]"')
        print('else')
        print()
    
#     dsvar = 'specific' if '_ts' not in data else prefix
    dsvar = var if '_ts' not in data else prefix
    
    if 'univariate_ts' in data:
        runopts += '-inprefix "' + prefix + '" '
        
    def mkdir(folder):
        print('md' if os.name == 'nt' else 'mkdir -p', '"'+folder+'"')
        print('')
    
#     if method == 'TEC' or method == 'TEC2':
    if 'TEC' in method:
        if 'ensemble_methods' in params:
            ensemble_methods = params['ensemble_methods']
        else:
            ensemble_methods = [['MML'], ['npoi']]
#         movelets_method = 'MML' if 'ensemble_methods' not in params else params['ensemble_methods'][0]
#         pois_method = 'npoi' if 'ensemble_methods' not in params else params['ensemble_methods'][1]
#         print('mkdir -p "${DIR}/'+mname+'-'+var+ '"')

        mkdir(DIRF)
#        print('mkdir -p "'+DIRF+'"')
#        print('')
        
        for movelets_method in ensemble_methods[0]:
            for pois_method in ensemble_methods[1]:
        
                methods = ['movelets_nn', 'npoi', 'marc']
                if method == 'TEC2':
                    methods = ['movelets_nn', 'marc']

                pois = ('_'.join(params['features']))+'_'+('_'.join([str(n) for n in params['sequences']]))
                poif_line     = os.path.join('${DIR}', pois_method.upper()+'-'+pois+'-'+var, pois_method+'_'+pois)
                # IF DIFFERENT METHOD, CHANGE modelfolder NAME!!
                movelets_line = os.path.join('${DIR}', movelets_method+'-'+var) 

                metsuff = movelets_method+ (pois_method if method != 'TEC2' else '')
                
                Ensemble(data, results, prefix, MNAME, methods=methods, \
                     modelfolder='model_'+metsuff, save_results=True, print_only=print_only, pyname=pyname, prg_path=prog_path, \
                     descriptor='', sequences=params['sequences'], features=params['features'], dataset=dsvar, num_runs=1,\
                     movelets_line=movelets_line, poif_line=poif_line)
    elif 'TC-' in method:
#        if 'rounds' in params:
#            print('# --------------------------------------------------------------------------------------')
#            print('for ROUND in '+ ' '.join(['"'+str(x)+'"' for x in range(1, params['rounds']+1)]) )
#            print('do')
#            print(pyshel(DEFAULT_TC, prog_path, pyname)+' -ds "'+dsvar+'" -c "'+method.split('-')[1]+'" -r ${ROUND} "'+data+'" "'+results+'/rand${ROUND}"', '2>&1 | tee -a "'+outfile+'" ')
#            print('done')
#            print('# --------------------------------------------------------------------------------------')
#            print()
#
#        else:
        print('# --------------------------------------------------------------------------------------')
        mkdir(DIRF)
        print(pyshel(DEFAULT_TC, prog_path, pyname)+' -ds "'+dsvar+'" -c "'+method.split('-')[1]+'" "'+data+'" "'+results+'"', 
              '2>&1 | tee -a "'+os.path.join(DIRF, MNAME)+'.txt" ')
        print()
    else:
        prefix = None
        
        if method == 'MARC':
            train_file = dsvar+"_train.csv" if '_ts' not in data else dsvar+"_TRAIN.ts"
            test_file  = dsvar+"_test.csv"  if '_ts' not in data else dsvar+"_TEST.ts"
            
            MARC(data, results, prefix, MNAME, print_only=print_only, prg_path=prog_path, #os.path.join(prog_path, PACKAGE_NAME,'marc'), 
                 pyname=pyname, train=train_file, test=test_file) # , extra_params='--no-gpu -M '+GIG+' -T '+THREADS


        elif 'poi' in method: #method == 'npoi' or method == 'poi' or method == 'wnpoi':
            POIFREQ(data, results, prefix, dsvar, params['sequences'], params['features'], method, print_only=print_only, pyname=pyname, prg_path=prog_path, or_methodsuffix=dsvar if '_ts' not in data else 'specific', or_folder_alias=MNAME)
            

        elif 'SM' in method:
            timeout = params['timeout'] if 'timeout' in params.keys() else None
            Movelets(data, results, prefix, MNAME, json, \
                     Ms=islog, extra=runopts, n_threads=THREADS, prg_path=prog_path, \
                     print_only=print_only, jar_name='SUPERMovelets', java_opts='-Xmx'+GIG+'G', \
                     pyname=pyname, timeout=timeout, impl=2, \
                     keep_folder=(2 if 'keep_results' not  in params else params['keep_results']))

        elif 'MM' in method or 'MMp' in method:
            timeout = params['timeout'] if 'timeout' in params.keys() else None
            Movelets(data, results, prefix, MNAME, json, Ms=islog, n_threads=THREADS, extra=runopts, \
                     prg_path=prog_path, print_only=print_only, jar_name='MASTERMovelets', java_opts='-Xmx'+GIG+'G', \
                     pyname=pyname, timeout=timeout, impl=2, \
                     keep_folder=(2 if 'keep_results' not  in params else params['keep_results']))

        elif 'Movelets' in method:
            timeout = params['timeout'] if 'timeout' in params.keys() else None
            Movelets(data, results, prefix, MNAME, json, Ms=islog, n_threads=THREADS, extra=runopts, \
                     prg_path=prog_path, print_only=print_only, jar_name=trimsuffix(method), java_opts='-Xmx'+GIG+'G', \
                     pyname=pyname, timeout=timeout, impl=1, \
                     keep_folder=(2 if 'keep_results' not  in params else params['keep_results']))

        elif oneof(method, ['Dodge', 'Xiao', 'Zheng']):
            timeout = params['timeout'] if 'timeout' in params.keys() else None
            Movelets(data, results, prefix, MNAME, json, Ms=islog, n_threads=THREADS, extra=runopts, \
                    prg_path=prog_path, print_only=print_only, jar_name=trimsuffix(method), java_opts='-Xmx'+GIG+'G', \
                    pyname=pyname, timeout=timeout, impl=0, \
                    keep_folder=(2 if 'keep_results' not  in params else params['keep_results']))


        else: #if 'hiper' in method or 'ultra' in method or 'random' in method or 'indexed' in method or method == 'pivots':
            desc = None if 'use.mat' in params and params['use.mat'] else json+'_hp'
            Movelets(data, results, prefix, MNAME, desc, version=trimsuffix(method), \
                     Ms=islog, extra=runopts, n_threads=THREADS, prg_path=prog_path, print_only=print_only, \
                     jar_name='HIPERMovelets', java_opts='-Xmx'+GIG+'G', pyname=pyname, \
                     keep_folder=(2 if 'keep_results' not in params else params['keep_results']))


        if 'samples' in params and not(method == 'MARC' or 'poi' in method or 'TEC' in method):
            print('# --------------------------------------------------------------------------------------')
            print('for FOLD in '+ ' '.join(['"run'+str(x)+'"' for x in range(1, params['samples']+1)]) )
            print('do')
#             print(pyname+' '+package_scripts+'/MergeDatasets.py "'+results+'/${FOLD}/'+MNAME+'"') #MERGE
            print(pyshel('MergeDatasets', prog_path, pyname)+' "'+results+'/${FOLD}/'+MNAME+'"') #MERGE
            if doacc :
                if 'rounds' in params:
                    print('for ROUND in '+ ' '.join(['"'+str(x)+'"' for x in range(1, params['rounds']+1)]) )
                    print('do')
                    print(pyshel(DEFAULT_MC, prog_path, pyname)+' -c "MLP,RF" -r ${ROUND} -m "model_${FOLD}" "'+results+'/${FOLD}" "'+MNAME+'"')
                    print('done')
                    print()
                else:
                    print(pyshel(DEFAULT_MC, prog_path, pyname)+' -c "MLP,RF" "'+results+'/${FOLD}" "'+MNAME+'"')
            print('done')
            print('# --------------------------------------------------------------------------------------')
            print()

        elif doacc and not(method == 'MARC' or 'poi' in method or 'TEC' in method):
            print('# --------------------------------------------------------------------------------------')
#             print(pyname+' '+package_scripts+'/Classifier-MLP_RF.py "'+results+'" "'+MNAME+'"') #MLP_RF
            #print(pyshel(DEFAULT_MC, prog_path, pyname)+' "'+results+'" "'+MNAME+'" "MLP,RF"') #MLP_RF
            if 'rounds' in params:
                print('for ROUND in '+ ' '.join(['"'+str(x)+'"' for x in range(1, params['rounds']+1)]) )
                print('do')
                print(pyshel(DEFAULT_MC, prog_path, pyname)+' -c "MLP,RF" -r ${ROUND} -m "model_${ROUND}" "'+results+'/${ROUND}" "'+MNAME+'"')
                print('done')
                print()
            else:
                print(pyshel(DEFAULT_MC, prog_path, pyname)+' -c "MLP,RF" "'+results+'" "'+MNAME+'"')
            print()
        
#     print('echo "${DIR}/'+mname+'-'+var+' => Done."')
    print('echo "'+DIRF+' ... Done."')
    if call_exit:
        print('exit 1')
    if check_done:
        print('fi')
    if k:
        print('done')
    print('# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ')
    
# --------------------------------------------------------------------------------------
def scalability(methods, datasets, params=None,
    Ns=[100, 10],
    Ms=[10,  10],
    Ls=[8,   10],
    Cs=[2,   10],
    Ts=[1,    4],
    n=-1, m=-1, l=-1, c=-1, t=-1,
    fileprefix='scalability',
    fileposfix='train',
    memoryprofile=True,
    run_prefix=''):
    
    assert Ns[0] > 0, 'N > 0'
    assert Ms[0] > 0, 'M > 0'
    assert Cs[0] > 0, 'C > 0'
    assert Ls[0] > 0, 'L > 0'
    assert Ns[0] >= Cs[0], 'N >= C'
    
    importer(['sys', 'itertools'], globals())
    from automatize.generator import getScale, getMiddleE
    
    N = getScale(Ns[0], Ns[1])
    M = getScale(Ms[0], Ms[1])
    L = getScale(Ls[0], Ls[1])
    C = getScale(Cs[0], Cs[1])
    T = getScale(Ts[0], Ts[1])
    
    n = getMiddleE(N) if n == -1 else n
    m = getMiddleE(M) if m == -1 else m
    l = getMiddleE(L) if l == -1 else l
    c = getMiddleE(C) if c == -1 else c
    t = getMiddleE(T) if t == -1 else t
    
    class Comb():
        def __init__(self, name, n, m, l, c, t):
            self.name = name
            self.n = n
            self.m = m
            self.l = l
            self.c = c
            self.t = t

        def __hash__(self):
            return hash((str(self.n), str(self.m), str(self.l), str(self.c), str(self.t)))
        def __eq__(self, other):
            return other and self.__hash__() == other.__hash__()
        def __repr__ (self):
            return '_'.join([str(self.n),'trajectories',str(self.m),'points',str(self.l),'attrs',str(self.c),'labels',str(self.t),'T'])

        def before(self, x, y):
            return x - y < 0
        def __lt__(self, other):
            if not other:
                return False

            if  (self.before(self.l, other.l)) or \
                (self.l == other.l and self.before(self.n, other.n)) or \
                (self.l == other.l and self.n == other.n and self.before(self.m, other.m)) or \
                (self.l == other.l and self.n == other.n and self.m == other.m and self.before(self.c, other.c)) or \
                (self.l == other.l and self.n == other.n and self.m == other.m and self.c == other.c and self.before(other.t, self.t)):
                return True
            else:
                return False
            
    def combs(name, N, M, L, C, T):
        return list(map(lambda x: Comb(name, x[0], x[1], x[2], x[3], x[4]), 
                                 itertools.product(N, M, L, C, T)))

    ln = combs('N', N, [m], [l], [c], [t])
    lm = combs('M', [n], M, [l], [c], [t])
    lc = combs('C', [n], [m], [l], C, [t])
    ll = combs('L', [n], [m], L, [c], [t])
    lt = combs('T', [n], [m], [l], [c], T)

    lx = list(set(ln + lm + lc + ll + lt))
    lx.sort()    
    
    print_only = True
    doacc = False
    check_done = True
    params['use.mat'] = True
    params['keep_results'] = 0
    
    if memoryprofile:
        run_prefix = 'mprof run --multiprocess ' + run_prefix
    
    f_name = ''

    def printScript(name, method, key, dataset, lx, params):
        method, params, mname, runopts, islog, pname, THREADS, GIG, data_folder, res_path, prog_path = configMethod(method, params)
        # -----------------------------
        TEST_PATH   = (params['folder'] if 'folder' in params else 'Scalability')+'-' + GIG+'G'
        DESC_PATH   = os.path.join(data_folder, 'descriptors')
        results     = os.path.join(res_path, TEST_PATH)
        # -----------------------------
        
        torun = ''
        
        prefix      = dataset #.capitalize()
        data        = os.path.join(data_folder, key, dataset)
        #json        = os.path.join(DESC_PATH, desc)

        os.makedirs(params['sh_folder'], exist_ok=True)        

        scrpt = 'run-scalability_'+name+'-'+mname+'-'+dataset+'.sh'
        scrpt = scrpt.replace('/', '_')

        torun += run_prefix + 'sh ' + scrpt + '\n'
        #print('sh ' + scrpt)

        orig_stdout = sys.stdout
        f = open(os.path.join(params['sh_folder'], scrpt), 'w')
        sys.stdout = f
        print('#!/bin/bash')

        #json = json + '_' + var
        file = '"'+fileprefix+'_${FILE}_'+fileposfix+'"'
        params['extra'] = '-inprefix ' + file

        #var = prefix if ('useds' in params and params['useds']) else var

        print('BASE="'+params['root']+'"')

        print('for FILE in "'+ '" "'.join(map(lambda x: str(x), lx)) + '"')
        print('do')
        print('# ------------------------------------------------------------------')

        printRun(method, data, results, prog_path, prefix, mname, '${FILE}', None, params, 
                 runopts, islog, print_only, check_done, doacc, pname)

        print('done')
        print("# Automatize - END generated script")
        sys.stdout = orig_stdout
        f.close()
        
        return torun

    # ---
    for key in datasets:
        for dataset in datasets[key]:
            for name in ['N', 'M', 'C', 'L', 'T']:
                for method in methods:
                    l = [x for x in lx if x.name == name]
                    f_name += printScript(name, method, key, dataset, l, params)
                f_name += '\n'  
            f_name += '# ------------------------------------------------------------------\n'

    # ---
    f = open(os.path.join(params['sh_folder'], 'run-all-scalability.sh'),'w')
    print('#!/bin/bash', file=f)
    print(f_name, file=f)
    print('# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ', file=f)
    print("# Automatize - END generated script", file=f)
    f.close()
    return f_name

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
        'for ROUND in ': 'FOR %%ROUND IN (',
        'done' : ')',
        '\ndo' : ') DO (',
        '" "run' : ',run',
        '${DIR}': '%DIRET%',
        '${BASE}': '%BASE%',
        '${RUN}': '%RUN%',
        '${ROUND}': '%ROUND%',
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
        
# --------------------------------------------------------------------------------------        
def avgRuntimeMethods(df):
    df['key'] = df['dataset'] + '-' + df['subsubset'] + '-' + df['method']
    def getPair(df, key):
        return [key, df['total_time'].mean()]
        
    dtt = dict( map(lambda k: getPair(df[df['key'] == k], k) , df['key'].unique()) )
    return dtt

def howLong(params, datasets, methods, varParam=None, avg_runtimes=dict(), deftime=None):
    
    totalTime = 0
    counter   = 0
    
#    def discripts(prefix, params, datasets, methods, run_prefix='', varParam=None):
    if not deftime and len(avg_runtimes) > 1:
        deftime = sum(avg_runtimes.values()) / len(avg_runtimes.values())
        print('Average Runtime Set:', deftime)
    elif not deftime:
        deftime = 36000000
        
    def runtime(dsName, dsFeat, method, params):
        time = 0
        runs = 0
        if method.startswith('TEC') and 'ensemble_methods' in params.keys():
            for movelets_method in params['ensemble_methods'][0]:
                for pois_method in params['ensemble_methods'][1]:
                    mname = movelets_method+ (pois_method if method != 'TEC2' else '')
                    time += getRuntime(dsName, dsFeat, mname, params)
                    runs += 1
        elif 'poi' in method:
            for sequence in params['sequences']:
                for feature in params['features']:
                    mname = method.upper() +'_'+ str(sequence)
                    dsFeat = feature+'_'+str(sequence)
                    time += getRuntime(dsName, dsFeat, mname, params)
                    runs += 1
                    
            mname = method.upper() +'_'+ ('_'.join(map(str, params['sequences'])))
            dsFeat = ('_'.join(params['features'])) +'_'+ ('_'.join(map(str, params['sequences'])))
            time += getRuntime(dsName, dsFeat, mname, params)
            runs += 1
            #params['sequences'], params['features']
        else:
            mname = configMethod(method, params)[2]
            time = getRuntime(dsName, dsFeat, mname, params)
            runs += 1
        return time, runs if not varParam else runs*len(varParam[1])
    
    def getRuntime(dsName, dsFeat, method, params):
        key = dsName+'-'+dsFeat+'-'+method
        if key in avg_runtimes.keys():
            return avg_runtimes[key]
        else:
            #print('Default runtime for', key)
            return deftime
        
    for dsType, ds in datasets.items():
        for dsn in ds:
            dsFeat = 'generic' if '?' in dsn else 'specific'
            dsName = dsn.replace('?', '')
            for method in methods:
                key = dsName+'-'+dsFeat+'-'+method
                time, runs = runtime(dsName, dsFeat, method, params)
                totalTime += time
                counter += runs
    
    if params['k']:
        k = params['k'] if isinstance(params['k'], int) else len(params['k'])
        totalTime = k*totalTime
        counter   = k*counter
    
    from datetime import timedelta
    delta = timedelta(milliseconds=totalTime)
    print('Estimated Runtime:', delta)
    print('Number of Runs   :', counter)
    return totalTime