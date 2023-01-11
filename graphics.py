# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Jul, 2022
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
from .main import importer #, display
importer(['S', 'plt', 'sns'], globals())
from .inc.script_def import METHODS_NAMES
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
def resultsBoxPlots(df, col, title='', methods_order=None, xaxis_format=False):
#    df.loc[df['method'].isin(
#        ['Xiao', 'Dodge', 'Movelets', 'Zheng',  'NPOI_1', 'NPOI_2', 'NPOI_3', 'NPOI_1_2_3', 'MARC']), 'error'] = False
    
    #display(df[['accuracy', 'error', 'method']].groupby(['error', 'method']).value_counts())
    
#    df.drop(df[df['accuracy'] == 0].index, inplace=True)
    n = len(df)
    df.drop(df[df['error'] == True].index, inplace=True)
    print('[WARN Box Plot:] Removed results due to run errors:', n - len(df))
#    df.drop(df[~df['method'].isin(selm)].index, inplace=True)

    if not methods_order:
        methods_order = list(df['method'].unique())

    df['methodi'] = df['method'].apply(lambda x: {methods_order[i]:i for i in range(len(methods_order))}[x])
    df = df.sort_values(['methodi', 'dataset', 'subset'])

    df['method'] = list(map(lambda m: METHODS_NAMES[m] if m in METHODS_NAMES.keys() else m, df['method']))
    
    # ---
    # COLOR PALETTE:
    pre = 4
    mypal = df['method'].unique()
    mypal_idx = list(set([x[:pre] for x in mypal]))
    mypal_idx.sort()
    pale = 'husl' #"Spectral_r"
    colors = sns.color_palette(pale, len(mypal_idx))
    mypal_idx = {mypal_idx[i]:colors[i] for i in range(len(mypal_idx))}
    mypal_idx = {x: mypal_idx[x[:pre]] for x in mypal}
    from itertools import product
    clas = df['classifier'].unique()
    mypal = {k+'-'+x:v for x in clas for k,v in mypal_idx.items()}
    mypal.update(mypal_idx)
    # ---

    df['key'] = df['dataset']+'-'+df['subset']
    if len(set(df['classifier'].unique()) - set('-')) > 1:
        df['name'] = df['method'] + list(map(lambda x: '-'+x if x != '-' else '', df['classifier']))
    else:
        df['name'] = df['method']

    sns.set(font_scale=1.5)
    sns.set_style("ticks")
#    sns.set_context("poster")
    def boxplot(df, col, xl):    
        plt.figure(figsize=(10,0.3*len(df['name'].unique())+1)) 
        p1 = sns.boxplot(data=df[['key', 'name', col]], y="name", x=col, palette=mypal)
        #plt.xticks(rotation=80)
        p1.set(xlabel=xl, ylabel='Method', title='')
        #if col == 'accuracy':
        #    p1.set(xlim=(-5, 105))
            
        if xaxis_format:
            if xaxis_format[0]:
                p1.set(xlim = xaxis_format[0])
            ticks_loc = p1.get_xticks().tolist()
            xlabels = ['{:,.1f}'.format(x/xaxis_format[1]) + xaxis_format[2] for x in ticks_loc]
            #p1.set_xticks(ticks_loc)
            p1.set_xticklabels(xlabels)
            
        plt.grid()
        plt.tight_layout()
        #plt.savefig(path+'/results-'+plotname+'-'+col+'.png')
        return p1.get_figure()

    
    
    return boxplot(df, col, xl=title)
# -----------------------------------------------------------------------