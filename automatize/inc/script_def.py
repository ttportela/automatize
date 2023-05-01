# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''

BASE_METHODS = [
    'MARC',
    'npoi',
#    'MM',
    'MM+Log',
    'MMp+Log',
    
    'ultra',
#    'ultra-random',
    'random+Log',
    
#    'SM',
#    'SM-2',
    'SM+Log',
    'SM-2+Log',
    
#    'hiper', 
    'hiper+Log', 
#    'hiper-pivots',  
    'hiper-pivots+Log',     
    
    'TC-TRF',
    'TC-TXGB',
    'TC-TULVAE',
    'TC-BITULER',
    'TC-DEEPEST',
]

METHODS_NAMES = {
    
    'Dodge': 'Dodge',
    'Xiao': 'Xiao',
    'Zheng': 'Zheng',
    'Movelets': 'Movelets', 
    
    'TRF': 'RF (DeepeST)',
    'TXGB': 'XGBoost (DeepeST)',
    'TULVAE': 'TULVAE (DeepeST)',
    'BITULER': 'BITULER (DeepeST)',
    'DEEPEST': 'DeepeST',
    
    'TC-TRF': 'RF (DeepeST)',
    'TC-TXGB': 'XGBoost (DeepeST)',
    'TC-TULVAE': 'TULVAE (DeepeST)',
    'TC-BITULER': 'BITULER (DeepeST)',
    'TC-DEEPEST': 'DeepeST',
    
    'poi':  'POI-F',
    'npoi': 'NPOI-F',
    'wpoi': 'WPOI-F',
    
    'POI_1':  'POI (1)',
    'POI_2':  'POI (2)',
    'POI_3':  'POI (3)',
    'POI_1_2_3':  'POI (1+2+3)',
    'NPOI_1':  'NPOI (1)',
    'NPOI_2':  'NPOI (2)',
    'NPOI_3':  'NPOI (3)',
    'NPOI_1_2_3':  'NPOI (1+2+3)',
    'WNPOI_1':  'WNPOI (1)',
    'WNPOI_2':  'WNPOI (2)',
    'WNPOI_3':  'WNPOI (3)',
    'WNPOI_1_2_3':  'WNPOI (1+2+3)',
    
    'MARC': 'MARC',
    
    'MM':   'MASTERMovelets',
    'MM+Log':  'MASTERMovelets-Log',
    'MMp':  'MASTERPivots',
    'MMp+Log': 'MASTERPivots-Log',
    'MML':  'MASTERMovelets-Log',
    'MMpL': 'MASTERPivots-Log',
    
    'SM': 'SUPERMovelets',
    'SM+Log': 'SUPERMovelets-Log',
    'SM-2': 'SUPERMovelets-λ',
    'SM+Log-2': 'SUPERMovelets-Log-λ',
    'SM-2+Log': 'SUPERMovelets-Log-λ',
    'SML': 'SUPERMovelets-Log',
    'SMD2': 'SUPERMovelets-λ',
    'SMD2L': 'SUPERMovelets-Log-λ',
    'SMLD2': 'SUPERMovelets-Log-λ',

    'hiper': 'HiPerMovelets', 
    'hiper+Log': 'HiPerMovelets-Log',
    'hiper-pivots': 'HiPerPivots', 
    'hiper-pivots+Log': 'HiPerPivots-Log',
    
    'H': 'HiPerMovelets τ=90%', 
    'HL': 'HiPerMovelets τ=90%',
    'HTR75': 'HiPerMovelets τ=75%', 
    'HTR75L': 'HiPerMovelets τ=75%',
    'HTR50': 'HiPerMovelets τ=50%', 
    'HTR50L': 'HiPerMovelets τ=50%',
    
    'Hp': 'HiPerPivots τ=90%', 
    'HpL': 'HiPerPivots τ=90%',
    'HpTR75': 'HiPerPivots τ=75%', 
    'HpTR75L': 'HiPerPivots τ=75%',
    'HpTR50': 'HiPerPivots τ=50%', 
    'HpTR50L': 'HiPerPivots τ=50%',
    
    'R': 'RandomMovelets',
    'random': 'RandomMovelets',
    'RL': 'RandomMovelets-Log', 
    'random+Log': 'RandomMovelets-Log',
    'U': 'UltraMovelets', 
    'ultra': 'UltraMovelets', 
#    'Ur': 'UltraMovelets-R',
}

METHODS_ABRV = {
    
    'Dodge': 'Dodge',
    'Xiao': 'Xiao',
    'Zheng': 'Zheng',
    'Movelets': 'Movelets', 
    
    'TRF': 'RF (D.)',
    'TXGB': 'XGBoost (D.)',
    'TULVAE': 'TULVAE (D.)',
    'BITULER': 'BITULER (D.)',
    'DEEPEST': 'DeepeST',
    
    'POI_1':  'POI (1)',
    'POI_2':  'POI (2)',
    'POI_3':  'POI (3)',
    'POI_1_2_3':  'POI (1+2+3)',
    'NPOI_1':  'NPOI (1)',
    'NPOI_2':  'NPOI (2)',
    'NPOI_3':  'NPOI (3)',
    'NPOI_1_2_3':  'NPOI (1+2+3)',
    'WNPOI_1':  'WNPOI (1)',
    'WNPOI_2':  'WNPOI (2)',
    'WNPOI_3':  'WNPOI (3)',
    'WNPOI_1_2_3':  'WNPOI (1+2+3)',
    
    'MARC': 'MARC',
    
    'MM':   'MM',
    'MM+Log':  'MM-Log',
    'MML':  'MM-Log',
    'MMp':  'MP',
    'MMp+Log': 'MP-Log',
    'MMpL': 'MP-Log',
    
    'SM': 'SM',
    'SM+Log': 'SM-Log',
    'SM-2': 'SM-λ',
    'SM+Log-2': 'SM-Log-λ',
    'SM-2+Log': 'SM-Log-λ',
    'SML': 'SM-Log',
    'SMD2': 'SM-λ',
    'SMD2L': 'SM-Log-λ',
    'SMLD2': 'SM-Log-λ',
    
    'H': 'HM τ=90%', 
    'HL': 'HM τ=90%',
    'HTR75': 'HM τ=75%', 
    'HTR75L': 'HM τ=75%',
    'HTR50': 'HM τ=50%', 
    'HTR50L': 'HM τ=50%',
    
    'Hp': 'HP τ=90%', 
    'HpL': 'HP τ=90%',
    'HpTR75': 'HP τ=75%', 
    'HpTR75L': 'HP τ=75%',
    'HpTR50': 'HP τ=50%', 
    'HpTR50L': 'HP τ=50%',
    
    'R': 'RM',
    'RL': 'RM-Log',
    'U': 'UM', 
}

CLASSIFIERS_NAMES = {
    '-':   'Self',
    'NN':  'Neural Network (NN)',
    'MLP': 'Neural Network (NN)',
    'RF':  'Random Forrest (RF)',
    'SVM': 'Support Vector Machine (SVM)',
}

METRICS_NAMES = {
    'f_score':       'F-Score',
    'f1_score':      'F-Measure',
    'accuracy':      'Accuracy',
    'accuracyTop5':  'Accuracy Top 5',
    'precision':     'Precision',
    'recall':        'Recall',
    'loss':          'Loss',
}

DESCRIPTOR_NAMES = {
    '*.*': None, 
    'multiple_trajectories.Gowalla':          'BrightkiteGowalla', 
    'multiple_trajectories.Brightkite':       'BrightkiteGowalla', 
    'multiple_trajectories.FoursquareNYC':    'FoursquareNYC', 
    'multiple_trajectories.FoursquareGlobal': 'FoursquareGlobal', 
    
    'raw_trajectories.*': 'RawTraj', 
    
    'semantic_trajectories.Promoters':     'GeneDS', 
    'semantic_trajectories.SJGS':          'GeneDS', 
    
    'univariate_ts.*': 'UnivariateTS', 
}

FEATURES_NAMES = {
    '*.*': ['poi'], 
    'multiple_trajectories.*': ['poi'], 
    'multiple_trajectories.FoursquareNYC?':    ['category'], 
    'multiple_trajectories.FoursquareGlobal?': ['category'], 
    
    'raw_trajectories.*':      ['lat_lon'], 
    
    'semantic_trajectories.Promoters':      ['sequence'], 
    'semantic_trajectories.SJGS':           ['sequence'], 
    
    'process.*':              ['Duration'],
    
    'univariate_ts.*':        ['dim0'],   # channel_1
    
    'multivariate_ts.*':                              ['dim0'],
    'multivariate_ts.ArticularyWordRecognition':      ['dim2'], # LL_z
    'multivariate_ts.AustralianSignLanguage':         ['dim17'], # right_thumb
    'multivariate_ts.CharacterTrajectories':          ['dim2'], # p
    'multivariate_ts.Cricket':                        ['dim5'], # channel_6
    'multivariate_ts.DuckDuckGeese':                  ['dim1244'], # channel_1245
    'multivariate_ts.Epilepsy':                       ['dim1'], # channel_2
    'multivariate_ts.EthanolConcentration':           ['dim2'], # channel_3
    'multivariate_ts.FaceDetection':                  ['dim105'], # channel_106
    'multivariate_ts.FingerMovements':                ['dim19'], # channel_20
    'multivariate_ts.GECCOWater':                     ['dim8'], # Tp
    'multivariate_ts.Heartbeat':                      ['dim58'], # channel_59
    'multivariate_ts.InsectWingbeat':                 ['dim1'], # channel_2
    'multivariate_ts.LSST':                           ['dim5'], # channel_6
    'multivariate_ts.MotorImagery':                   ['dim63'], # channel_64
    'multivariate_ts.NATOPS':                         ['dim4'], # channel_5
    'multivariate_ts.PEMS-SF':                        ['dim618'], # channel_619
    'multivariate_ts.PenDigits':                      ['dim1'], # channel_2
    'multivariate_ts.PhonemeSpectra':                 ['dim10'], # channel_11
    'multivariate_ts.SelfRegulationSCP1':             ['dim5'], # channel_6
    'multivariate_ts.SelfRegulationSCP2':             ['dim6'], # channel_7
    'multivariate_ts.StandWalkJump':                  ['dim1'], # channel_2
    'multivariate_ts.UWaveGestureLibrary':            ['dim2'], # channel_3
#     'multivariate_ts.ActivityRecognition':            ['dim0'], # x
#     'multivariate_ts.AtrialFibrillation':             ['dim0'], # channel_1
#     'multivariate_ts.BasicMotions':                   ['dim0'], # channel_1
#     'multivariate_ts.ERing':                          ['dim0'], # channel_1
#     'multivariate_ts.EigenWorms':                     ['dim0'], # channel_1
#     'multivariate_ts.FaciesRocks':                    ['dim0'], # Depth
#     'multivariate_ts.GrammaticalFacialExpression':    ['dim0'], # 0.0
#     'multivariate_ts.HandMovementDirection':          ['dim0'], # channel_1
#     'multivariate_ts.Handwriting':                    ['dim0'], # x
#     'multivariate_ts.JapaneseVowels':                 ['dim0'], # channel_1
#     'multivariate_ts.Libras':                         ['dim0'], # x
#     'multivariate_ts.RacketSports':                   ['dim0'], # channel_1
#     'multivariate_ts.SpokenArabicDigits':             ['dim0'], # channel_1
}

def getName(dic, dst=None, dsn=None):
    dst = (dst if dst else '*')
    dsn = (dsn if dsn else '*')
    if dst +'.'+ dsn in dic.keys():
        name = dic[dst +'.'+ dsn]
    elif dst +'.*' in dic.keys():
        name = dic[dst +'.*']
    elif '*.*' in dic.keys():
        name = dic['*.*']
        
    if not name:
        name = dsn 
    return name

def getDescName(dst, dsn):
    name = getName(DESCRIPTOR_NAMES, dst, dsn)
    if not name:
        name = dsn
    return name

def getFeature(dst, dsn):
    name = getName(FEATURES_NAMES, dst, dsn)
    if not name:
        name = ['poi']
    return name

def getSubset(dsn, feature):
    for key, value in FEATURES_NAMES.items():
        if dsn in key and feature in value:
            if '?' in key:
                return 'generic'
            
    return 'specific'

def readK(kRange):
    if ',' not in kRange and '-' not in kRange:
        k = range(1, int(kRange))
    else:
        k = []
        for x in kRange.split(','):
            k += [int(x)] if '-' not in x else list(range( int(x.split('-')[0]), int(x.split('-')[1])+1 ))
    return k

def metricName(code):
    code = code.replace('metric:', '')
    
    if code in METRICS_NAMES.keys():
        return METRICS_NAMES[code]
    
    name = code[0].upper()
    for c in code[1:]:
        if c.isupper():
            name += ' ' + c
        elif c.isdigit() and not name[-1].isdigit():
            name += ' ' + c
        elif c == '_':
            name += '-'
        else:
            name += c
    
    return name

def datasetName(dataset, subset):
    if subset == 'specific':
        return dataset
    else:
        return dataset + ' ('+subset+')'