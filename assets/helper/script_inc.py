BASE_METHODS = [
    'MARC',
    'npoi',
    'MM',
    
#     'ultra',
#     'random',
    
    'SM',
    'hiper', 
    'hiper-pivots',     
]

METHODS_NAMES = {

    'hiper': 'HiPerMovelets', 
    'hiper+Log': 'HiPerMovelets-Log',
    'hiper-pivots': 'HiPerMovelets-Pivots', 
    'hiper-pivots+Log': 'HiPerMovelets-Pivots-Log',
    'H': 'HiPerMovelets', 
    'HL': 'HiPerMovelets-Log',
    'Hp': 'HiPerMovelets-Pivots', 
    'HpL': 'HiPerMovelets-Pivots-Log',
    
    'SM': 'SUPERMovelets',
    'SM+Log': 'SUPERMovelets-Log',
    'SM-2': 'SUPERMovelets-λ',
    'SM+Log-2': 'SUPERMovelets-Log-λ',
    'SM-2+Log': 'SUPERMovelets-Log-λ',
    'SML': 'SUPERMovelets-Log',
    'SMD2': 'SUPERMovelets-λ',
    'SMD2L': 'SUPERMovelets-Log-λ',
    
    'MM':   'MASTERMovelets',
    'MM+Log':  'MASTERMovelets-Log',
    'MMp':  'MASTERMovelets-Pivots',
    'MMp+Log': 'MASTERMovelets-Pivots-Log',
    'MML':  'MASTERMovelets-Log',
    'MMpL': 'MASTERMovelets-Pivots-Log',
    
#     'ultra',
#     'random',
    
    'MARC': 'MARC',
    'poi':  'POI-F',
    'npoi': 'NPOI-F',
    'wpoi': 'WPOI-F',
    
    'Movelets': 'Movelets',
    'Dodge': 'Dodge',
    'Xiao': 'Xiao',
    'Zheng': 'Zheng',
    
}

CLASSIFIERS_NAMES = {
    'MLP': 'Neural Network',
    'RF':  'Random Forrest',
    'SVM': 'Support Vector Machine',
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
