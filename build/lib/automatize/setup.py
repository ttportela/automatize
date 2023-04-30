'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
import setuptools

DEV_VERSION = "1.0b15"
VERSION = "1.0b2"
PACKAGE_NAME = 'automatize'

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name=PACKAGE_NAME,
#    version=VERSION,
    version=DEV_VERSION,
    author="Tarlis Tortelli Portela",
    author_email="tarlis@tarlis.com.br",
    description="Automatize: Multiple Aspect Trajectory Data Mining Tool Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ttportela/automatize",
    packages=setuptools.find_packages(include=[PACKAGE_NAME, PACKAGE_NAME+'.*']),
    scripts=[
#         'scripts/MAT-PrintResults.py', # Deprecated
#         'scripts/MAT-ResultsCheck.py', # Deprecated
        
        'automatize/scripts/MAT-CheckRun.py',
        'automatize/scripts/MAT-MergeDatasets.py',
        'automatize/scripts/MAT-ResultsTo.py',
        'automatize/scripts/MAT-ExportResults.py',
        'automatize/scripts/MAT-ExportMovelets.py',
        
        'automatize/scripts/MARC.py',
        'automatize/scripts/POIS.py',
        'automatize/scripts/MAT-MC.py',
        'automatize/scripts/MAT-TC.py',
        'automatize/scripts/POIS-TC.py',
#        'automatize/scripts/MAT-TEC.py', # Under Dev.
        'automatize/scripts/MAT.py',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    keywords='data mining, python, trajectory classification, trajectory analysis, movelets',
    license='GPL Version 3 or superior (see LICENSE file)',
)