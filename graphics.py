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
# -----------------------------------------------------------------------