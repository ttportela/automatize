# -*- coding: utf-8 -*-
'''
Automatize: Multi-Aspect Trajectory Data Mining Tool Library
The present application offers a tool, called AutoMATize, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. The AutoMATize integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
License GPL v.3 or superior

@author: Tarlis Portela
'''
import sys, os 
script_dir = os.path.dirname( __file__ )
main_dir = os.path.abspath(os.path.join( script_dir, '..' , '..'))
sys.path.append( main_dir )

from automatize.main import importer #, display
importer(['S', 'datetime','poifreq'], globals())

if len(sys.argv) < 6:
    print('Please run as:')
    print('\tPOIS.py', 'METHOD', 'SEQUENCES', 'FEATURES', 'DATASET', 'PATH TO DATASET', 'PATH TO RESULTS_DIR')
    print('Example:')
    print('\tPOIS.py', 'npoi', '"1,2,3"', '"poi,hour"', 'specific', '"./data"', '"./results"')
    exit()

METHOD = sys.argv[1]
SEQUENCES = [int(x) for x in sys.argv[2].split(',')]
FEATURES = sys.argv[3].split(',')
DATASET = sys.argv[4]
path_name = sys.argv[5]
RESULTS_DIR = sys.argv[6]

# from automatize.ensemble_models.poifreq import poifreq
time = datetime.now()
poifreq(SEQUENCES, DATASET, FEATURES, path_name, RESULTS_DIR, method=METHOD, save_all=True, doclass=False)
time_ext = (datetime.now()-time).total_seconds() * 1000

print("Done. Processing time: " + str(time_ext) + " milliseconds")
print("# ---------------------------------------------------------------------------------")