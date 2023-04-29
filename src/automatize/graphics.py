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
from .inc.script_def import METHODS_NAMES, METHODS_ABRV, datasetName
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
def resultsBoxPlots(df, col, title='', methods_order=None, xaxis_format=False, plot_type='box'): # plot_type='box' | 'swarm'
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
    pre = 6
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
        if plot_type == 'swarm':
            p1 = sns.swarmplot(data=df[['key', 'name', col]], y="name", x=col, palette=mypal)
        else: # Default: boxplot
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
def lineRank(df, col, title='', methods_order=None, xaxis_format=False): # plot_type='box' | 'swarm'
    n = len(df)
    df.drop(df[df['error'] == True].index, inplace=True)
    print('[WARN Line Rank Plot:] Removed results due to run errors:', n - len(df))

    if not methods_order:
        methods_order = list(df['method'].unique())
    
    df['key'] = list(map(lambda d,s: datasetName(d,s), df['dataset'], df['subset']))
    df = df.groupby(['key', 'name', 'dataset', 'subset', 'method', 'classifier'])[col].mean().reset_index()

    df['methodi'] = df['method'].apply(lambda x: {methods_order[i]:i for i in range(len(methods_order))}[x])
    df = df.sort_values(['methodi', 'dataset', 'subset'])

    df['method'] = list(map(lambda m: METHODS_NAMES[m] if m in METHODS_NAMES.keys() else m, df['method']))

    # ---
    # COLOR PALETTE:
    mypal = list(df['key'].unique())
    mypal.sort()
    pale = 'husl' #"Spectral_r"
    colors = sns.color_palette(pale, len(mypal))
    mypal = {mypal[i]:colors[i] for i in range(len(mypal))}
    # ---
    
    if len(set(df['classifier'].unique()) - set('-')) > 1:
        df['name'] = df['method'] + list(map(lambda x: '-'+x if x != '-' else '', df['classifier']))
    else:
        df['name'] = df['method']
    
    sns.set(font_scale=1.5)
    sns.set_style("ticks")
    
    plt.figure(figsize=(0.5*len(df['name'].unique())+1,0.5*len(df['key'].unique())+1)) 
        
    #ds_wide = df.groupby(['key', 'name'])[col].mean().reset_index()
    #ds_wide = ds_wide.pivot('name', 'key', col)
    
    #ds_wide = ds_wide.sort_values([col])

#    for datakey in ds_wide['key'].unique()
    p1 = sns.lineplot(data=df[['key', 'name', col]], x='name', y=col, hue='key', palette=mypal)

    p1.set(ylabel=title, xlabel='Method', title='')
    plt.xticks(rotation=80)
    p1.legend(loc='upper left', bbox_to_anchor=(1, 0.5))

    if xaxis_format:
        if xaxis_format[0]:
            p1.set(ylim = xaxis_format[0])
        ticks_loc = p1.get_yticks().tolist()
        xlabels = ['{:,.1f}'.format(x/xaxis_format[1]) + xaxis_format[2] for x in ticks_loc]
        #p1.set_xticks(ticks_loc)
        p1.set_yticklabels(xlabels)

    plt.grid()
    plt.tight_layout()
    return p1.get_figure()

# -----------------------------------------------------------------------
def hbarPlot(df, col, title='', methods_order=None, xaxis_format=False, plot_type='bar'): # plot_type='bar' | 'line'
    n = len(df)
    df.drop(df[df['error'] == True].index, inplace=True)
    print('[WARN Line Rank Plot:] Removed results due to run errors:', n - len(df))

    if not methods_order:
        methods_order = list(df['method'].unique())
        
    df = df.groupby(['name', 'method', 'classifier', 'dataset', 'subset'])[col].mean().reset_index()

    df['methodi'] = df['method'].apply(lambda x: {methods_order[i]:i for i in range(len(methods_order))}[x])
    df = df.sort_values(['methodi', 'dataset', 'subset'])

    df['method'] = list(map(lambda m: METHODS_NAMES[m] if m in METHODS_NAMES.keys() else m, df['method']))
    
    df['key'] = list(map(lambda d,s: datasetName(d,s), df['dataset'], df['subset']))

    # ---
    # COLOR PALETTE:
#    mypal = list(df['key'].unique())
#    mypal.sort()
#    pale = 'husl' #"Spectral_r"
#    colors = sns.color_palette(pale, len(mypal))
#    mypal = {mypal[i]:colors[i] for i in range(len(mypal))}
    
    pre = 6
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
    
    if len(set(df['classifier'].unique()) - set('-')) > 1:
        df['name'] = df['method'] + list(map(lambda x: '-'+x if x != '-' else '', df['classifier']))
    else:
        df['name'] = df['method']
    
    sns.set(font_scale=1.5)
    sns.set_style("ticks")
    
    a_size = 0.03*len(df['key'].unique())+1
#    if xaxis_format and xaxis_format[0]:
#        b_size = 1*(xaxis_format[0][1]/xaxis_format[1])+1 
#    else:
        
    b_size = 100
    
    a_size = 0.03*len(df['key'].unique())+1
    plt.figure(figsize=(15,5)) 
        
    #ds_wide = ds_wide.pivot('name', 'key', col)

#    for datakey in ds_wide['key'].unique()
#    p1 = sns.catplot(ds_wide[['key', 'name', col]], y='key', x=col, hue='name', kind='bar', palette=mypal, 
#                     legend='lower center', legend_out=True)
    p1 = sns.barplot(df[['key', 'name', col]], y='key', x=col, hue='name', palette='husl')#mypal)

    p1.set(xlabel=title, ylabel='Dataset', title='')
    #plt.legend(loc='lower center', title='Method')
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0., title='Method')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title='Method')
    #plt.xticks(rotation=80)

    if xaxis_format:
        if xaxis_format[0]:
            p1.set(xlim = xaxis_format[0])
        ticks_loc = p1.get_xticks().tolist()
        xlabels = ['{:,.1f}'.format(x/xaxis_format[1]) + xaxis_format[2] for x in ticks_loc]
        p1.set_xticklabels(xlabels)

    plt.grid()
    plt.tight_layout()
    return p1.get_figure()

# -----------------------------------------------------------------------
def barPlot(df, col, title='', methods_order=None, datasets_order=None, plot_type='bar', mean_aggregation=False, 
            label_format=dict()): # plot_type='bar' | 'line'
    n = len(df)
    df.drop(df[df['error'] == True].index, inplace=True)
    print('[WARN Line Rank Plot:] Removed results due to run errors:', n - len(df))

    if not methods_order:
        methods_order = list(df['method'].unique())
        
    if not datasets_order:
        datasets_order = list(df['dataset'].unique())
        datasets_order.sort()

    df = df.groupby(['name', 'method', 'classifier', 'dataset', 'subset'])[col].mean().reset_index()
    
    df['key'] = list(map(lambda d,s: datasetName(d,s), df['dataset'], df['subset']))

    df['methodi'] = df['method'].apply(lambda x: {methods_order[i]:i for i in range(len(methods_order))}[x])
    df['dsi'] = df['key'].apply(lambda x: {datasets_order[i]:i for i in range(len(datasets_order))}[x])
    df = df.sort_values(['methodi', 'dsi'])

    df['method'] = list(map(lambda m: METHODS_ABRV[m] if m in METHODS_ABRV.keys() else m, df['method']))
    

    # ---
    format_config = {'scale':1, 'suffix':'', 'xrotation':25, 'label_pos':(0,0), 'label_rotation':70}
    format_config.update(label_format)
    # ---
    # COLOR PALETTE:
    pre = 6
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
    
    if len(set(df['classifier'].unique()) - set('-')) > 1:
        df['name'] = df['method'] + list(map(lambda x: '-'+x if x != '-' else '', df['classifier']))
    else:
        df['name'] = df['method']
    
    sns.set(font_scale=1.5)
    sns.set_style("ticks")
    
    a_size = 0.03*len(df['key'].unique())+1
#    if xaxis_format and xaxis_format[0]:
#        b_size = 1*(xaxis_format[0][1]/xaxis_format[1])+1 
#    else:
        
    b_size = 100
    
    a_size = 0.03*len(df['key'].unique())+1
    plt.figure(figsize=(14,5)) 
    plt.rcParams['font.size'] = 12
        
    #ds_wide = ds_wide.pivot('name', 'key', col)

#    for datakey in ds_wide['key'].unique()
#    p1 = sns.catplot(ds_wide[['key', 'name', col]], y='key', x=col, hue='name', kind='bar', palette=mypal, 
#                     legend='lower center', legend_out=True)
    if mean_aggregation:
        p1 = sns.barplot(df[['key', 'name', col]], x='name', y=col, palette='husl')#mypal)
    else:
        p1 = sns.barplot(df[['key', 'name', col]], x='name', y=col, hue='key', palette='husl')#mypal)
        #plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title='Dataset', fontsize=13)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=100, borderaxespad=0.)

    p1.set(ylabel=title, xlabel='Method', title='')
    #plt.legend(loc='lower center', title='Method')
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0., title='Method')
    plt.xticks(rotation=format_config['xrotation'], horizontalalignment='right')

    def format_val(x):
        if 'format_func' in format_config.keys():
            return format_config['format_func'](x)
        elif 'mask' in format_config.keys():
            return format_config['mask'].format(x/format_config['scale'])# + format_config['suffix']
        else:
            return '{:,.{}f}'.format(x/format_config['scale'], 0)# + format_config['suffix']
    def format_axis(x):
        return format_val(x) + format_config['suffix']
    
    if 'ylim' in format_config.keys():
        p1.set(ylim = format_config['ylim'])
    elif not mean_aggregation:
        ymax = df[col].max()
        ymax = ymax * 1.2
        p1.set(ylim = (0, ymax))
    ticks_loc = p1.get_yticks().tolist()
    xlabels = [format_axis(x) for x in ticks_loc]
    p1.set_yticklabels(xlabels)
        
    #position = (0,0) if 'label_pos' not in label_format.keys() else label_format['label_pos']#(0,-15)
    for container in p1.containers:
        p1.bar_label(container, labels=[format_val(x) for x in container.datavalues], 
                     rotation=format_config['label_rotation'], horizontalalignment='center', position=format_config['label_pos'])

    #plt.grid()
    plt.tight_layout()
    return p1.get_figure()
# -----------------------------------------------------------------------