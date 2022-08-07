# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
# --------------------------------------------------------------------------------
# # ANALYSIS
from .main import importer #, display
importer(['S'], globals())

from .methods.movelet.classification import *

# --------------------------------------------------------------------------------
# from PACKAGE_NAME.Methods import Approach1, Approach2, ApproachRF, ApproachRFHP , ApproachMLP, ApproachDT, ApproachSVC
# --------------------------------------------------------------------------------
def ACC4All(res_path, res_folder, save_results = True, modelfolder='model', classifiers=['MLP', 'RF', 'SVM'], random_seed=1, data_path=''):
#     import os
# #     import sys
#     import numpy as np
# #     import pandas as pd
#     import glob2 as glob
# #     from datetime import datetime
#     from ..main import importer
    importer(['S', 'glob'], globals())

    filelist = []
    filesList = []
    
    for files in glob.glob(os.path.join(res_path, res_folder, "**", "*.txt")):
        fileName, fileExtension = os.path.splitext(files)
        method = os.path.basename(fileName)#[:-4]
        path = os.path.dirname(fileName)#[:-len(method)]
        todo = not os.path.exists( os.path.join(path, modelfolder) )
        empty = not os.path.exists( os.path.join(path, "train.csv") )
        if todo and not empty:
            ClassifyByMovelet(path, '', '', data_path, save_results, modelfolder, classifiers, random_seed)
        else:
            print(method + (" Done." if not empty else " Empty."))
            
# ----------------------------------------------------------------------------------
def ClassifyByMovelet(res_path, dataset, dir_path, data_path='', save_results = True, modelfolder='model', classifiers=['MLP', 'RF', 'SVM'], random_seed=1):
#     import os
# #     import sys
# #     import numpy as np
#     import pandas as pd
# #     import glob2 as glob
# #     from datetime import datetime
# #     def_random_seed(random_num, seed_num)
#     from ..main import importer
#     importer(['S'], locals())

    importer(['random'], globals())
    np.random.seed(seed=random_seed)
    #random.set_seed(random_seed)
    
    dir_path = os.path.join(res_path, dataset, dir_path)
    times = {'SVM': [0], 'RF': [0], 'MLP': [0], 'TEC': [0]}
    
    times_file = os.path.join(dir_path, modelfolder, "classification_times.csv")
    if os.path.isfile(times_file):
        times = pd.read_csv(times_file)
    
    if 'RF' in classifiers:
        times['RF']  = [MoveletClassifier_RF(dir_path, save_results, modelfolder)]
        
    if 'MLP' in classifiers:
        times['MLP'] = [MoveletClassifier_MLP(dir_path, save_results, modelfolder)]
        
    if 'SVM' in classifiers:
        times['SVM'] = [MoveletClassifier_SVM(dir_path, save_results, modelfolder)]
        
    if 'TEC' in classifiers:
        times['TEC'] = [MoveletClassifier_TEC(dir_path, data_path, save_results, modelfolder, random_seed=random_seed)]
        
#     t_svm = Classifier_SVM(dir_path, save_results, modelfolder)
#     t_rf  = Classifier_RF(dir_path, save_results, modelfolder)
#     t_mlp = Classifier_MLP(dir_path, save_results, modelfolder)
    
    # ------
    if (save_results) :
        if not os.path.exists(os.path.join(dir_path, modelfolder)):
            os.makedirs(os.path.join(dir_path, modelfolder))
        pd.DataFrame(times).to_csv(times_file)
            
# ----------------------------------------------------------------------------------
def ClassifyByTrajectory(res_path, data_path, prefix=None,
                         save_results=True, classifiers=['MARC', 'POIS', 'TRF', 'TXGB', 'TULVAE', 'BITULER', 'DST'],
                         random_seed=1, geohash=False):

    prefix = prefix if prefix else 'specific'

    if 'MARC' in classifiers:
        from automatize.methods.marc.marc_nn import marc
        train_file = os.path.join(data_path, prefix+'_train.csv')
        test_file = os.path.join(data_path, prefix+'_test.csv')
            
        marc('OURS', train_file, test_file, \
             os.path.join(res_folder, 'MARC-'+prefix+'_results.csv'), \
             prefix, 100, "concatenate", "lstm", random_seed=random_seed) # geohash always true
    
    if 'POIS' in classifiers:
        from automatize.methods.pois.poifreq import poifreq
        sequences = [1,2,3]
        dataset   = 'specific'
        features  = None
        poifreq(sequences, dataset, features, data_path, res_path, method='npoi', doclass=True)

    if 'TRF' in classifiers:
        from automatize.methods.rf.randomforrest import TrajectoryRF
        TrajectoryRF(data_path, res_path, prefix, save_results, random_state=random_seed)

    if 'TXGB' in classifiers:
        from automatize.methods.xgboost.XGBoost import TrajectoryXGBoost
        TrajectoryXGBoost(data_path, res_path, prefix, save_results, random_state=random_seed)

    if 'TULVAE' in classifiers:
        from automatize.methods.tuler.TULVAE import TrajectoryTULVAE
        TrajectoryTULVAE(data_path, res_path, prefix, save_results, random_state=random_seed)

    if 'BITULER' in classifiers:
        from automatize.methods.tuler.BITULER import TrajectoryBITULER
        TrajectoryBITULER(data_path, res_path, prefix, save_results, random_state=random_seed)

    if 'DST' in classifiers:
        from automatize.methods.deepest.DeepestST import TrajectoryDeepestST
        TrajectoryDeepestST(data_path, res_path, prefix, save_results, random_state=random_seed)
            
# ----------------------------------------------------------------------------------
#def MLP(res_path, prefix, dir_path, save_results = True, modelfolder='model'):
##     import os
##     from ..main import importer
##     importer(['os'], locals())
##     def_random_seed(random_num, seed_num)
#    
#    dir_path = os.path.join(res_path, prefix, dir_path)
#    t = Classifier_MLP(dir_path, save_results, modelfolder)
#    return t
#
#def RF(res_path, prefix, dir_path, save_results = True, modelfolder='model'):
##     import os
##     from ..main import importer
##     importer(['os'], locals())
##     def_random_seed(random_num, seed_num)
#    
#    dir_path = os.path.join(res_path, prefix, dir_path)
#    t = Classifier_RF(dir_path, save_results, modelfolder)
#    return t
#
#def SVM(res_path, prefix, dir_path, save_results = True, modelfolder='model'):
##     import os
##     from ..main import importer
##     importer(['os'], locals())
##     def_random_seed(random_num, seed_num)
#    
#    dir_path = os.path.join(res_path, prefix, dir_path)
#    t = Classifier_SVM(dir_path, save_results, modelfolder)
#    return t
#
#def TrajectoryRF(res_path, prefix, dir_path, save_results = True, modelfolder='model'):
##     import os
##     from ..main import importer
##     importer(['os'], locals())
##     def_random_seed(random_num, seed_num)
#    
#    dir_path = os.path.join(res_path, prefix, dir_path)
#    t = Classifier_SVM(dir_path, save_results, modelfolder)
#    return t