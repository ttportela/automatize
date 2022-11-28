# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Jul, 2022
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
from .main import importer #, display
importer(['S'], globals())
# -----------------------------------------------------------------------
def loadDataset(path, file='specific_train.csv'):
    from automatize.preprocessing import readDataset, organizeFrame
    df = readDataset(path, file=file)
    df,_,_ = organizeFrame(df)
    df.drop(['space'], axis=1, inplace=True)
    return df
# -----------------------------------------------------------------------
    
def geoHeatMap(df):
    # now let's visualize a cloropleth map of the home locations
    import folium
    from folium.plugins import HeatMap
    m = folium.Map(tiles = 'openstreetmap', zoom_start=12, control_scale=True)
    HeatMap(df[['lat', 'lon']].values).add_to(m)
    return m
# -----------------------------------------------------------------------

# Results Visualizations:
# -----------------------------------------------------------------------
def resultsBoxPlots(df, col, title=''):
    df.loc[df['method'].isin(
        ['Xiao', 'Dodge', 'Movelets', 'Zheng',  'NPOI_1', 'NPOI_2', 'NPOI_3', 'NPOI_1_2_3', 'MARC']), 'error'] = False
    
    #display(df[['accuracy', 'error', 'method']].groupby(['error', 'method']).value_counts())
    
    df.drop(df[df['accuracy'] == 0].index, inplace=True)
    df.drop(df[df['error'] == True].index, inplace=True)
#    df.drop(df[~df['method'].isin(selm)].index, inplace=True)

#    df['methodi'] = df['method'].apply(lambda x: {selm[i]:i for i in range(len(selm))}[x])
#    df = df.sort_values(['methodi', 'dataset', 'subset'])

    df['method'] = list(map(lambda m: METHODS_NAMES[m] if m in METHODS_NAMES.keys() else m, df['method']))

    df['key'] = df['dataset']+'-'+df['subset']
    df['name'] = df['method']+'-'+df['classifier']

    #sns.set(font_scale=2)
    def boxplot(df, col, xl):    
        plt.figure(figsize=(10,0.3*len(df['name'].unique())+1)) 
        p1 = sns.boxplot(data=df[['key', 'name', col]], y="name", x=col, palette="Spectral_r")
        #plt.xticks(rotation=80)
        p1.set(xlabel='', ylabel='Method', title=xl)
        if col == 'accuracy':
            p1.set(xlim=(-5, 105))
        plt.tight_layout()
        #plt.savefig(path+'/results-'+plotname+'-'+col+'.png')
        return p1.get_figure()

    return boxplot(df, col, xl=title)
# -----------------------------------------------------------------------