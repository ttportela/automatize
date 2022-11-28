# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Jul, 2022
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
from .main import importer #, display
importer(['S', 'generator', 'tqdm'], globals(), {'preprocessing': ['writeFile', 'featuresJSON']})

#import math
#import random
#import itertools

#from .inc.io.converter import *
#from .preprocessing import
# --------------------------------------------------------------------------------
def scalerSamplerGenerator(
    Ns=[100, 10],
    Ms=[10,  10],
    Ls=[8,   10],
    Cs=[2,  10],
    random_seed=1,
    fileprefix='scalability',
    fileposfix='train',
    cols_for_sampling = ['space','time','day','rating','price','weather','root_type','type'],
    save_to=None,
    base_data=None,
    save_desc_files=True,
    outformats=['csv', 'mat']):
    '''
    [Function to generate trajectory datasets based on real data.]

    Args:
        Ns [int, int]: 
            Parameters to scale number of trajectories. 
            List of 2 values: starting number, number of elements (default [100, 10])
        Ms [int, int]: 
            Parameters to scale size of Trajectories. 
            List of 2 values: starting number, number of elements (default [10, 10])
        Ls [int, int]: 
            Parameters to scale number of attributes (* doubles the columns). 
            List of 2 values: starting number, number of elements (default [8, 10])
        Cs [int, int]: 
            Parameters to scale number of classes. 
            List of 2 values: starting number, number of elements (default [2, 10])
        random_seed [int]: 
            Random Seed (default 1)
        cols_for_sampling [list] or [dict]: 
            Columns to add in the generated dataset (default 'lat_lon,time,day,rating,price,weather,root_type,type')
            OR if use dictionary in the format: {'aspectName': 'type', 'aspectName': 'type'} when broviding base_data and saving .MAT
        save_to [str, bool]: 
            Destination folder to save or False, if not to save csv file (default False)
        fileprefix [str]: 
            Output filename prefix (default 'sample')
        fileposfix [str]: 
            Output filename postfix (default 'train')
        base_data [DataFrame]: 
            DataFrame of trajectoris to use as base for sampling data (default 'assets/examples/Foursquare_Example.csv')
        save_desc_files [bool]: 
            True if to save the .json description files or False, if not to save (default True)
        outformats [list]:
            Output file formats for saving (default ['csv', 'mat'])
    
    Returns:
        [pandas.DataFrame] the generated dataset
    '''
    
    assert Ns[0] > 0, 'N > 0'
    assert Ms[0] > 0, 'M > 0'
    assert Cs[0] > 0, 'C > 0'
    assert Ls[0] > 0, 'L > 0'
    assert Ns[0] >= Cs[0], 'N >= C'
    assert save_to, 'save_to param must be set.'
    
    # Random Seed
    np.random.seed(seed=random_seed)
    random.seed(random_seed)
    
    df, cols_for_sampling, desc_cols = getSamplingData(base_data, cols_for_sampling)
    
    pbar = tqdm(range(Ns[1] + Ms[1] + Ls[1] + Cs[1]))
    
    Ns = getScale(Ns[0], Ns[1])
    Ms = getScale(Ms[0], Ms[1])
    La = getScale(Ls[0], Ls[1])
    Cs = getScale(Cs[0], Cs[1])
    
    miN = getMiddleE(Ns)
    miM = getMiddleE(Ms)
    miL = len(cols_for_sampling) #getMiddleE(La)
    miC = getMiddleE(Cs)
    
    print('N ::', 'fix. value:', '\t', miN, '\tscale:\t', Ns)
    print('M ::', 'fix. value:', '\t', miM, '\tscale:\t', Ms)
    print('L ::', 'fix. value:', '\t', miL, '\tscale:\t', La)
    print('C ::', 'fix. value:', '\t', miC, '\tscale:\t', Cs)
    
    # 1 - Scale attributes (reshape columns), fixed trajectories, points, and classes:
    cols = cols_for_sampling.copy()
    prefix = fileprefix #+ '_L'
    for i in range(Ls[1]):        
        #if len(cols) == miL:
        #    cols_for_sampling = cols
            
        samplerGenerator(miN, miM, miC, random_seed, fileprefix, fileposfix, cols, save_to, df, outformats)
        pbar.update(1)
        
        if save_to and save_desc_files:
            featuresJSON(desc_cols, 1, file=os.path.join(save_to, '_'.join([fileprefix,str(len(cols)),'attrs'])+ ".json"))
            featuresJSON(desc_cols, 2, file=os.path.join(save_to, '_'.join([fileprefix,str(len(cols)),'attrs'])+ "_hp.json"))
        
        if i < Ls[1]-1:
            df_ = df[cols].copy()
            df_ = df_.add_suffix('_'+str(i+1))
            
            cols = cols + list(df_.columns)
            
            
            df = pd.concat([df, df_], axis=1)
    
    # 2 -  Scale trajectories, fixed points, attributes, and classes
    #prefix = fileprefix + '_N'
    for i in Ns:
        pbar.update(1)
        if i == miN:
            continue
        samplerGenerator(i,miM,miC,random_seed, fileprefix, fileposfix, cols_for_sampling, save_to, df, outformats)
    
    # 3 -  Scale points, fixed trajectories, attributes, and classes
    #prefix = fileprefix + '_M'
    for i in Ms:
        pbar.update(1)
        if i == miM:
            continue
        samplerGenerator(miN,i,miC,random_seed, fileprefix, fileposfix, cols_for_sampling, save_to, df, outformats)
    
    # 4 -  Scale classes, fixed trajectories, points, and attributes
    #prefix = fileprefix + '_C'
    for i in Cs:
        pbar.update(1)
        if i == miC:
            continue
        samplerGenerator(miN,miM,i,random_seed, fileprefix, fileposfix, cols_for_sampling, save_to, df, outformats)

def samplerGenerator(
    N=10,
    M=50,
    C=1,
    random_seed=1,
    fileprefix='sample',
    fileposfix='train',
    cols_for_sampling = ['space','time','day','rating','price','weather','root_type','type'],
    save_to=False,
    base_data=None,
    outformats=['csv', 'mat']):
    '''
    [Function to generate trajectories based on real data.]

    Args:
        N [int]: 
            Number of Trajectories (default 10)
        M [int]: 
            Size of Trajectories (default 50)
        C [int]: 
            Number of classes (default 1)
        random_seed [int]: 
            Random Seed (default 1)
        cols_for_sampling [list]: 
            Columns to add in the generated dataset (default 'lat_lon,time,day,rating,price,weather,root_type,type')
        save_to [str, bool]: 
            Destination folder to save or False, if not to save csv file (default False)
        fileprefix [str]: 
            Output filename prefix (default 'sample')
        fileposfix [str]: 
            Output filename postfix (default 'train')
        base_data [DataFrame]: 
            DataFrame of trajectoris to use as base for sampling data (default 'assets/examples/Foursquare_Example.csv')
        outformats [list]:
            Output file formats for saving (default ['csv', 'mat'])
    
    Returns:
        [pandas.DataFrame] the generated dataset
    '''
    
    assert N > 0, 'N > 0'
    assert M > 0, 'M > 0'
    assert C > 0, 'C > 0'
    assert N >= C, 'N >= C'
    
    # Random Seed
    np.random.seed(seed=random_seed)
    random.seed(random_seed)
    
    df, cols_for_sampling, desc_cols = getSamplingData(base_data, cols_for_sampling)
    
    #cols_for_sampling = ['lat_lon','time','day','price','weather','type']
    df_for_sampling = df[cols_for_sampling]
    
    # Number of Trajectories per class
    n = int(N / C)
    
    new_df = pd.concat( list(map(lambda j: sample_set(df_for_sampling, n, M, 'C'+str(j+1), j), range(C))) )
    
    if len(new_df['tid'].unique()) < N:
        df_ = sample_trajectory(df_for_sampling, M, N)
        df_['label'] = 'C'+str(C)
        new_df = pd.concat([new_df, df_])
    
    # Orders by tid, day e time
    new_df = new_df.sort_values(['tid','day','time'])
    # Reset indexes
    new_df.reset_index(drop=True, inplace=True)

    # Output file:
    if save_to:
        if not os.path.exists(save_to):
            os.makedirs(save_to)
            
        filename = '_'.join([fileprefix,
                             str(N),'trajectories',
                             str(M),'points',
                             str(len(cols_for_sampling)),'attrs',
                             str(C),'labels',
                             fileposfix])
        
        for outType in outformats:
            writeFile(save_to, new_df, filename, 'tid', 'label', ['tid', 'label']+cols_for_sampling, None, desc_cols, outType)
        #filename += '.csv'
        #new_df.to_csv( os.path.join(save_to, filename), index=False)

    return new_df

def getSamplingData(base_data, cols_for_sampling):
    if base_data is None:
        base_data = os.path.join(os.path.dirname(__file__), 'assets', 'examples', 'Foursquare_Example.csv')
        df = pd.read_csv(base_data).dropna()
        df = df.rename(columns={"lat_lon": "space"})
    
        cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['day'] = pd.Categorical(df['day'], categories=cats, ordered=True)
        
        desc_cols = {'space':'space2d', 'time':'time', 'day':'nominal', 'rating':'numeric', 'price':'numeric', 
                     'weather':'nominal', 'root_type':'nominal', 'type':'nominal'}
    else:
        df = base_data
        
        if type(cols_for_sampling) is dict:
            desc_cols = cols_for_sampling.copy()
            cols_for_sampling = list(cols_for_sampling.keys())
        else:
            desc_cols = None

    return df, cols_for_sampling, desc_cols

def scalerRandomGenerator(
    Ns=[100, 10],
    Ms=[10,  10],
    Ls=[8,   10],
    Cs=[2,  10],
    random_seed=1,
    fileprefix='scalability',
    fileposfix='train',
    attr_desc=None,
    save_to=None,
    save_desc_files=True,
    outformats=['csv', 'mat']):
    '''
    [Function to generate trajectory datasets based on real data.]

    Args:
        Ns [int, int]: 
            Parameters to scale number of trajectories. 
            List of 2 values: starting number, number of elements (default [100, 10])
        Ms [int, int]: 
            Parameters to scale size of Trajectories. 
            List of 2 values: starting number, number of elements (default [10, 10])
        Ls [int, int]: 
            Parameters to scale number of attributes (* doubles the columns). 
            List of 2 values: starting number, number of elements (default [8, 10])
        Cs [int, int]: 
            Parameters to scale number of classes. 
            List of 2 values: starting number, number of elements (default [2, 10])
        random_seed [int]: 
            Random Seed (default 1)
        attr_desc [list]: 
            Data type intervals to generate attributes as list of desciptive dicts (default 'default_types()')
        save_to [str, bool]: 
            Destination folder to save or False, if not to save csv file (default False)
        fileprefix [str]: 
            Output filename prefix (default 'sample')
        fileposfix [str]: 
            Output filename postfix (default 'train')
        save_desc_files [bool]: 
            True if to save the .json description files or False, if not to save (default True)
        outformats [list]:
            Output file formats for saving (default ['csv', 'mat'])
    
    Returns:
        [pandas.DataFrame] the generated dataset
    '''
    
    assert Ns[0] > 0, 'N > 0'
    assert Ms[0] > 0, 'M > 0'
    assert Cs[0] > 0, 'C > 0'
    assert Ls[0] > 0, 'L > 0'
    assert Ns[0] >= Cs[0], 'N >= C'
    assert save_to, 'save_to param must be set.'
    
    # Random Seed
    np.random.seed(seed=random_seed)
    random.seed(random_seed)
    
    pbar = tqdm(range(Ns[1] + Ms[1] + Ls[1] + Cs[1]))
    
    Ns = getScale(Ns[0], Ns[1])
    Ms = getScale(Ms[0], Ms[1])
    La = getScale(Ls[0], Ls[1])
    Cs = getScale(Cs[0], Cs[1])
    
    miN = getMiddleE(Ns)
    miM = getMiddleE(Ms)
    miL = Ls[0] #getMiddleE(La)
    miC = getMiddleE(Cs)
    
    print('N ::', 'fix. value:', '\t', miN, '\tscale:\t', Ns)
    print('M ::', 'fix. value:', '\t', miM, '\tscale:\t', Ms)
    print('L ::', 'fix. value:', '\t', miL, '\tscale:\t', La)
    print('C ::', 'fix. value:', '\t', miC, '\tscale:\t', Cs)
    
    if not attr_desc:
        attr_desc = default_types()[:Ls[0]]
    
    generators = instantiate_generators(attr_desc)
    
    # 1 - Scale attributes (reshape columns), fixed trajectories, points, and classes:
    #prefix = fileprefix #+ '_L'
    for i in La:        
        randomGenerator(miN, miM, i, miC, random_seed, fileprefix, fileposfix, cycleGenerators(i, generators), save_to, outformats)
        pbar.update(1)
        
        if save_to and save_desc_files:
            desc_cols = {g.name: g.descType() for g in cycleGenerators(i, generators)}
            
            featuresJSON(desc_cols, 1, file=os.path.join(save_to, '_'.join([fileprefix,str(i),'attrs'])+ ".json"))
            featuresJSON(desc_cols, 2, file=os.path.join(save_to, '_'.join([fileprefix,str(i),'attrs'])+ "_hp.json"))
    
    # 2 -  Scale trajectories, fixed points, attributes, and classes
    #prefix = fileprefix + '_N'
    for i in Ns:
        pbar.update(1)
        if i == miN:
            continue
        randomGenerator(i, miM, miL, miC, random_seed, fileprefix, fileposfix, generators, save_to, outformats)
    
    # 3 -  Scale points, fixed trajectories, attributes, and classes
    #prefix = fileprefix + '_M'
    for i in Ms:
        pbar.update(1)
        if i == miM:
            continue
        randomGenerator(miN, i, miL, miC, random_seed, fileprefix, fileposfix, generators, save_to, outformats)
    
    # 4 -  Scale classes, fixed trajectories, points, and attributes
    #prefix = fileprefix + '_C'
    for i in Cs:
        pbar.update(1)
        if i == miC:
            continue
        randomGenerator(miN, miM, miL, i, random_seed, fileprefix, fileposfix, generators, save_to, outformats)

def randomGenerator(
    N=10,
    M=50,
    L=10,
    C=10,
    random_seed=1,
    fileprefix='random',
    fileposfix='train',
    attr_desc=None,
    save_to=False,
    outformats=['csv', 'mat']):
    '''
    [Function to generate trajectories based on real data.]

    Args:
        N [int]: 
            Number of Trajectories (default 10)
        M [int]: 
            Size of Trajectories (default 50)
        L [int]: 
            Number of attributes (default 10)
        C [int]: 
            Number of classes (default 10)
        random_seed [int]: 
            Random Seed (default 1)
        attr_desc [list of dict]: 
            Data type intervals to generate attributes as list of desciptive dicts (default 'default_types()')
            OR a list of instances of AttributeGenerator
        save_to [str, bool]: 
            Destination folder to save or False, if not to save csv file (default False)
        fileprefix [str]: 
            Output filename prefix (default 'sample')
        fileposfix [str]: 
            Output filename postfix (default 'train')
        outformats [list]:
            Output file formats for saving (default ['csv', 'mat'])
    
    Returns:
        [pandas.DataFrame] the generated dataset
    '''
    
    assert N > 0, 'N > 0'
    assert M > 0, 'M > 0'
    assert L > 0, 'L > 0'
    assert C > 0, 'C > 0'
    assert N >= C, 'N >= C'
    
    # Random Seed
    np.random.seed(seed=random_seed)
    random.seed(random_seed)
    
    if not attr_desc:
        attr_desc = default_types()
        
    if isinstance(attr_desc[0], AttributeGenerator):
        generators = attr_desc
    else:
        generators = instantiate_generators(attr_desc)
    
    # Number of Trajectories per class
    n = int(N / C)

    new_df = pd.concat( list(map(lambda j: random_set(n, M, L, 'C'+str(j+1), j, generators), range(C))) )

    if len(new_df['tid'].unique()) < N:
        df_ = random_trajectory(M, L, N, generators)
        df_['label'] = 'C'+str(C)
        new_df = pd.concat([new_df, df_])
    
    # Orders by tid, day e time
    #new_df = new_df.sort_values(['tid','day','time'])
    # Reset indexes
    new_df.reset_index(drop=True, inplace=True)

    # Output file:
    if save_to:
        if not os.path.exists(save_to):
            os.makedirs(save_to)
            
        filename = '_'.join([fileprefix,
                             str(N),'trajectories',
                             str(M),'points',
                             str(L),'attrs',
                             str(C),'labels',
                             fileposfix])
        
        desc_cols = {g.name: g.descType() for g in generators}
        
        for outType in outformats:
            writeFile(save_to, new_df, filename, 'tid', 'label', list(new_df.columns), None, desc_cols, outType)
        #filename += '.csv'
        #new_df.to_csv( os.path.join(save_to, filename), index=False)

    return new_df
    
    print('Not implemented.')

# --------------------------------------------------------------------------------
def default_types():
    return [
        {'name': 'space',    'atype': 'space',    'method': 'grid_cell',    'interval': [(0.0,1000.0), (0.0,1000.0)]},
        {'name': 'time',     'atype': 'time',     'method': 'random',       'interval': [0, 1440]},
        {'name': 'n1',       'atype': 'numeric',  'method': 'random',       'interval': [-1000, 1000]},
        {'name': 'n2',       'atype': 'numeric',  'method': 'random',       'interval': [0.0, 1000.0]},
        {'name': 'nominal',  'atype': 'nominal',  'method': 'random',       'n': 1000},
        {'name': 'day',      'atype': 'day',      'method': 'random',       'interval': ['Monday', 'Tuesday', 'Thursday', 
                                                                                     'Friday', 'Saturday', 'Sunday', 'Wednesday']},
        {'name': 'weather',  'atype': 'weather',  'method': 'random',     'interval': ['Clear', 'Clouds', 'Fog', 'Unknown', 'Rain', 'Snow']},
        #{'name': 'poi',      'atype': 'poi',      'method': 'serial',    'n': 100,      'dependency': 'space'},
        {'name': 'category', 'atype': 'category', 'method': 'random',     'dependency': 'space',
         'interval': ['Residence', 'Food', 'Travel & Transport', 'Professional & Other Places', 'Shop & Service',
                      'Outdoors & Recreation', 'College & University', 'Arts & Entertainment', 'Nightlife Spot', 'Event']},
    ]

def instantiate_generators(attr_desc=default_types()):
    return list(map(lambda g: AttributeGenerator(**g), attr_desc))    

def getScale(start=100, n_ele=10):
    return [start] + (getScale(start+start, n_ele-1) if n_ele-1 else [])

def getMiddleE(X):
    return X[int((len(X) - 1)/2)]

def sample_trajectory( df, M, tid ):
    df_ = df.sample(M)
    df_.insert(0,'tid', tid+1)
    return df_

def sample_set(df_for_sampling, N, M, label, j): 
    # Number of Trajectories per class
    #n = int(N / C)
    # Creates the set of N trajectories of size M
    new_df = pd.concat( list(map(lambda i: sample_trajectory(df_for_sampling, M, i+j*N), range(N) )) )
    new_df['label'] = label
    # Orders by tid, day e time
    new_df = new_df.sort_values(['tid','day','time'])
    # Reset indexes
    new_df.reset_index(drop=True, inplace=True)
    return new_df

def cycleGenerators(L, generators):
    return list(itertools.islice(itertools.cycle(generators), L))

def random_trajectory(M, L, tid, generators):
    g = cycleGenerators(L, generators)
    df_ = pd.concat( list(map(lambda i: pd.Series(g[i].nextn(M), name='a'+str(i+1)+'_'+g[i].name), range(L))), axis=1)
    df_.insert(0,'tid', tid+1)
    return df_

def random_set(N, M, L, label, j, generators):
    # Creates the set of N trajectories of size M
    new_df = pd.concat( list(map(lambda i: random_trajectory(M, L, i+j*N, generators), range(N) )) )
    new_df['label'] = label
    # Orders by tid, day e time
    #new_df = new_df.sort_values(['tid'])
    # Reset indexes
    new_df.reset_index(drop=True, inplace=True)
    return new_df

# --------------------------------------------------------------------------------
class AttributeGenerator:
    
    def __init__(self, name='attr', atype='nominal', method='random', interval=None, n=-1, dependency=None,\
                cellSize=1, adjacents=None, precision=2):
        self.name = name
        self.atype = atype
        self.method = method
        self.dependency = dependency
        
        if atype in ['nominal', 'day', 'weather', 'poi', 'category']:
            self.generator = NominalGenerator(method, n, interval)
        elif atype in ['time', 'numeric']:
            self.generator = NumericGenerator(method, interval[0], interval[1], precision)
        elif atype in ['space']:
            self.generator = SpatialGrid2D(interval[0], interval[1], cellSize, adjacents, precision)
    
    def descType(self):
        if self.atype in ['nominal', 'day', 'weather', 'poi', 'category']:
            return 'nominal'
        elif self.atype in ['space']:
            return 'space2d'
        else:
            return self.atype
    
    def next(self):
        return self.generator.next()
    
    def nextn(self, n):
        return self.generator.nextn(n)

class NominalGenerator:
    
    @staticmethod
    def nominalInterval(n):
        def getNominalCombs(ncomb=1):
            return [''.join(comb) for comb in itertools.product((lambda x, i: [chr(ord('A')+y) for y in range(i)])('A', 26), repeat=ncomb)]

        i = 1
        ls = getNominalCombs(i)
        while len(ls) < n:
            i += 1
            ls = ls + getNominalCombs(i)

        return ls[:n]
    
    def __init__(self, method='random', n=50, interval=None):
        self.method = method
        self.n = n
        self.interval = self.nominalInterval(n) if interval is None else interval
        self.pos = -1
    
    def next(self):
        if self.method=='random':
            return random.choice(self.interval)
        elif self.method=='sequential':
            self.pos = self.pos+1 if self.pos < self.n-1 else 0
            return self.interval[self.pos]
        elif self.method=='serial':
            raise NotImplementedError('NominalGenerator.method==serial')
        
    def nextn(self, n):
        return list(map(lambda i: self.next(), range(n)))

class NumericGenerator:    
    def __init__(self, method='random', start=0, end=100, precision=2):
        self.method = method
        self.start = start
        self.end = end
        self.last = start
        self.precision = precision
        #self.pos = -1
    
    def next(self):
        if self.method=='random':
            return random.randint(self.start, self.end) if isinstance(self.start, int) else \
                    round(random.uniform(self.start, self.end), self.precision)
        elif self.method=='serial':
            raise NotImplementedError('NumericGenerator.method==serial')
            #return 0#self.interval[self.pos]
        
    def nextn(self, n):
        return list(map(lambda i: self.next(), range(n)))

class SpatialGrid2D:
    SPATIAL_ADJACENTS_2 = [
        ( -2, -2 ), ( -2, -1 ), ( -2, 0 ), ( -2, 1 ), ( -2, 2 ), 
        ( -1, -2 ), ( -1, -1 ), ( -1, 0 ), ( -1, 1 ), ( -2, 2 ),
        (  0, -2 ), (  0, -1 ),            (  0, 1 ), ( -2, 2 ), 
        (  1, -2 ), (  1, -1 ), (  1, 0 ), (  1, 1 ), ( -2, 2 ),
        (  2, -2 ), (  2, -1 ), (  2, 0 ), (  2, 1 ), ( -2, 2 )
    ]
    SPATIAL_ADJACENTS_1 = [
        ( -1, -1 ), ( -1, 0 ), ( -1, 1 ),
        (  0, -1 ),            (  0, 1 ),
        (  1, -1 ), (  1, 0 ), (  1, 1 ),
    ]
    
    def __init__(self, X=(1,5), Y=(1,5), cellSize=1, spatial_adjacents=None, precision=2, dependency=[]):
        self.X = X
        self.Y = Y
        
        #self.spatialThreshold = spatialThreshold # 0.00142
        #self.cellSize = self.spatialThreshold * cellSizeFactor #0.7071
        self.cellSize = cellSize
        self.precision = precision
        
        if not spatial_adjacents:
            self.SPATIAL_ADJACENTS = self.SPATIAL_ADJACENTS_1
        else:
            self.SPATIAL_ADJACENTS = spatial_adjacents

    def size(self):
        return int(((self.X[1] - self.X[0]) / self.cellSize) * ((self.Y[1] - self.Y[0]) / self.cellSize))
        
    def position(self, x, y):
        if x < self.X[0] or x > self.X[1] or y < self.Y[0] or y > self.Y[1]:
            return None
        
        #to include the edges:
        if x == self.X[1]:
            x = x - (self.cellSize/2)
        if y == self.Y[1]:
            y = y - (self.cellSize/2)
            
        return ( int(math.floor(x / self.cellSize)) , int(math.floor(y / self.cellSize)) )

    def adjacents(self, cell):
        return list(filter(lambda ajc: ajc[0] >= self.X[0] and ajc[0] <= (self.X[1]-self.cellSize) and 
                                       ajc[1] >= self.Y[0] and ajc[1] <= (self.Y[1]-self.cellSize), 
                           map(lambda ajc: (cell[0]+(ajc[0]*self.cellSize), cell[1]+(ajc[1]*self.cellSize)), self.SPATIAL_ADJACENTS)))
    
    def randomRoute(self, startCell, n):
        route = [startCell]
        for i in range(n-1):
            route.append( random.choice(self.adjacents(route[-1])) )
        return route
    
    def next(self):
        return round(random.uniform(self.X[0], self.X[1]+self.cellSize), self.precision), \
               round(random.uniform(self.Y[0], self.Y[1]+self.cellSize), self.precision)
    
    def nextin(self, cell): # next in cell
        return round(random.uniform(cell[0], cell[0]+self.cellSize), self.precision), \
               round(random.uniform(cell[1], cell[1]+self.cellSize), self.precision)
    
    def nextn(self, n):
        return list(map(lambda i: self.text(self.next()), range(n)))
    
    def text(self, point):
        return str(point[0]) + ' ' + str(point[1])
# --------------------------------------------------------------------------------