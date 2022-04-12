# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
@author: Lucas May Petry
'''
import geohash2 as gh
import numpy as np
# from ...main import importer #, display
# importer(['S', 'gh'], locals())


base32 = ['0', '1', '2', '3', '4', '5', '6', '7',
          '8', '9', 'b', 'c', 'd', 'e', 'f', 'g',
          'h', 'j', 'k', 'm', 'n', 'p', 'q', 'r',
          's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
binary = [np.asarray(list('{0:05b}'.format(x, 'b')), dtype=int)
          for x in range(0, len(base32))]
base32toBin = dict(zip(base32, binary))


# Deprecated - for compatibility purposes
class LatLonHash:

    def __init__(self, lat, lon):
        self._lat = lat
        self._lon = lon

    def to_hash(self, precision=15):
        return gh.encode(self._lat, self._lon, precision)

    def to_binary(self, precision=15):
        hashed = self.to_hash(precision)
        return np.concatenate([base32toBin[x] for x in hashed])


def geohash(lat, lon, precision=15):
    return gh.encode(lat, lon, precision)


def bin_geohash(lat, lon, precision=15):
    hashed = geohash(lat, lon, precision)
    return np.concatenate([base32toBin[x] for x in hashed])
