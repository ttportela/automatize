import os
import glob2 as glob
from automatize.assets.config import DATA_PATH
from automatize.helper.script_inc import getDescName

DATASET_TYPES = {
    'multiple_trajectories':     'Multiple Aspect Trajectories', 
    'raw_trajectories':          'Raw Trajectories', 
    'semantic_trajectories':     'Semantic Trajectories', 
    'multivariate_ts':           'Multivariate Time Series', 
    'univariate_ts':             'Univariate Time Series',
}

SUBSET_TYPES = {
   '*.specific': 'Multiple',
   'multiple_trajectories.specific': 'Multiple Aspect',
   'raw_trajectories.specific':      'Raw',
   'semantic_trajectories.specific': 'Semantic',
   'multivariate_ts.specific':       'Multivariate',
   'univariate_ts.specific':         'Univariate',
    
   '*.raw':      'Spatio-Temporal',
    
   '*.spatial':  'Spatial',
   '*.generic':  'Generic',
   '*.category': 'Category',
   '*.poi':      'POI',
   '*.5dims':    '5-Dimensions',
}

def list_datasets(data_path=DATA_PATH):
#     files = []
#     for category in DATASET_TYPES.keys():
#         files_aux = glob.glob(os.path.join(data_path, category, '*', '*.md'))
#         files = files + files_aux
    
#     datasets = []
    
#     for f in files:
# #         tmp = os.path.dirname(f).split(os.path.sep)
#         name = os.path.basename(f).split('.')[0]
        
#         datasets.append(name)
        
#     return datasets
    datasetsdict = list_datasets_dict(data_path)
    datasets = {}
    
    for category, lsds in datasetsdict.items():
        for dataset, subsets in lsds.items():
            for ss in subsets:
                if ss == 'specific':
                    datasets[category+'.'+dataset] = dataset
                elif ss == 'generic':
                    datasets[category+'.'+dataset+'?'] = dataset + ' (generic)'

    return datasets

def list_datasets_dict(data_path=DATA_PATH):
    datasets_dict = {}
    for category in DATASET_TYPES.keys():
        files = glob.glob(os.path.join(data_path, category, '*', '*.md'))
        datasets_dict[category] = {}
    
        datasets = []
        for f in files:
            if f.endswith('-stats.md'):
                continue
            tmp = os.path.dirname(f).split(os.path.sep)
            name = os.path.basename(f).split('.')[0]

            datasets_dict[category][name] = list_subsets(name, category, f)
        
    return datasets_dict

def list_subsets(dataset, category, file, return_files=False):        
    subsets = set()
    desc_files = glob.glob(os.path.join(os.path.dirname(file), '..', 'descriptors', '*.json'))
    for f in desc_files:
#         print(os.path.basename(f).split('.')[0], os.path.dirname(f).split(os.path.sep))
        descName = os.path.basename(f) #.split('.')[0]
        descName = translateDesc(dataset, category, descName)
        if descName:
            if f.endswith('_hp.json') and not return_files:
                subsets.add(descName)
            elif return_files:
                subsets.add(f)
      
    subsets = list(subsets)
    subsets.sort()
    if 'specific' in subsets:
        subsets.remove('specific')
        subsets.insert(0, 'specific')

    return subsets

# ------------------------------------------------------------
def translateDesc(dataset, category, descName):        
    dst, dsn = descName.split('.')[0].split('_')[0:2]
    if dsn in ['allfeat', '5dims']:
        return False

    if getDescName(category, dataset) == dst:
        return dsn
    elif dataset in dst:
        return dsn
    return False

def translateCategory(dataset, category, descName=None):
    if descName:        
        if (category+'.'+descName) in SUBSET_TYPES.keys():
            return SUBSET_TYPES[category+'.'+descName]
        elif ('*.'+descName) in SUBSET_TYPES.keys():
            return SUBSET_TYPES['*.'+descName]
        else:
            return descName.capitalize()
        
    elif category in DATASET_TYPES.keys():
        return DATASET_TYPES[category]
    
    else:
        return category.split('_')[0].title()