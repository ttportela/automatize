# Automatize: Multiple Aspect Trajectory Data Mining Tool Library
---

\[[Publication](#)\] \[[citation.bib](citation.bib)\] \[[GitHub](https://github.com/ttportela/automatize)\] \[[PyPi](https://pypi.org/project/automatize/)\]


Welcome to Automatize Framework for Multiple Aspect Trajectory Analysis. You can use it as a web-platform or a Python library.

The present application offers a tool, called AutoMATize, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. The AutoMATize integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

### Main Modules

- [Datasets](/datasets): Datasets descriptions, statistics and files to download;
- [Methods](/methods): Methods for trajectory classification and movelet extraction;
- [Scripting](/experiments): Script generator for experimental evaluation on available methods (Linux shell);
- [Results](/results): Experiments on trajectory datasets and method rankings;
- [Analysis](/analysis): Multiple Aspect Trajectory Analysis Tool (trajectory and movelet visualization);
- [Publications](/publications): Multiple Aspect Trajectory Analysis related publications;
- [Tutorial](/tutorial): Tutorial on how to use Automatise as a Python library.


### Available Classifiers:

* **TEC**: Trajectory Ensemble Classifier [REFERENCE*]
* **MLP (Movelet)**: Multilayer-Perceptron (MLP) with movelets features. The models were implemented using the Python language, with the keras, fully-connected hidden layer of 100 units, Dropout Layer with dropout rate of 0.5, learning rate of 10−3 and softmax activation function in the Output Layer. Adam Optimization is used to avoid the categorical cross entropy loss, with 200 of batch size, and a total of 200 epochs per training. [REFERENCE*]
* **RF (Movelet)**: Random Forest (RF) with movelets features, that consists of an ensemble of 300 decision trees. The models were implemented using the Python language, with the keras. [REFERENCE*]
* **SVN (Movelet)**: Support Vector Machine (SVM) with movelets features. The models were implemented using the Python language, with the keras, linear kernel and default structure. Other structure details are default settings. [REFERENCE*]
* **POI-S**: Frequency-based method to extract features of trajectory datasets (TF-IDF approach), the method runs one dimension at a time (or more if concatenated). The models were implemented using the Python language, with the keras. [REFERENCE*]
* **MARC**: Uses word embeddings for trajectory classification. It encapsulates all trajectory dimensions: space, time and semantics, and uses them as input to a neural network classifier, and use the geoHash on the spatial dimension, combined with others. The models were implemented using the Python language, with the keras. [REFERENCE*]
* **TRF**: Random Forest for trajectory data (TRF). Find the optimal set of hyperparameters for each model, applying the grid-search technique: varying number of trees (ne), the maximum number of features to consider at every split (mf), the maximum number of levels in a tree (md), the minimum number of samples required to split a node (mss), the minimum number of samples required at each leaf node (msl), and finally, the method of selecting samples for training each tree (bs). [REFERENCE*]
* **XGBost**: Find the optimal set of hyperparameters for each model, applying the grid-search technique:  number of estimators (ne), the maximum depth of a tree (md), the learning rate (lr), the gamma (gm), the fraction of observations to be randomly samples for each tree (ss), the sub sample ratio of columns when constructing each tree (cst), the regularization parameters (l1) and (l2). [REFERENCE*]
* **BITULER**: Find the optimal set of hyperparameters for each model, applying the grid-search technique: keeps 64 as the batch size and 0.001 as the learning rate and vary the units (un) of the recurrent layer, the embedding size to each attribute (es) and the dropout (dp). [REFERENCE*]
* **TULVAE**: Find the optimal set of hyperparameters for each model, applying the grid-search technique: keeps 64 as the batch size and 0.001 as the learning rate and vary the units (un) of the recurrent layer, the embedding size to each attribute (es), the dropout (dp), and latent variable (z). [REFERENCE*]
* **DEEPEST**: DeepeST employs a Recurrent Neural Network (RNN), both LSTM and Bidirectional LSTM (BLSTM). Find the optimal set of hyperparameters for each model, applying the grid-search technique: keeps 64 as the batch size and 0.001 as the learning rate and vary the units (un) of the recurrent layer, the embedding size to each attribute (es) and the dropout (dp). [REFERENCE*]
* **MainTUL**: [REFERENCE*]

### Installation

Install directly from PyPi repository, or, download from github. Intalling with pip will also provide command line scripts (available in folder `automatize/scripts`). (python >= 3.5 required)

```bash
    pip install automatize
```

To use Automatize as a python library, find examples in this sample Jupyter Notebbok: [Automatize_Sample_Code.ipynb](./assets/examples/Automatize_Sample_Code.ipynb)

#### Available Scripts:

By installing the package the following python scripts will be installed for use in system command line tools:

* `MAT-TC.py`, and `MAT-MC.py`
* `POIS-TC.py`:
* `MARC.py`:
* `MAT-TEC.py`:

One script for running the **POI-F/POI-S** method:

* `POIS.py`:

Scripts for helping the management of result files:

* `MAT-CheckRun.py`:
* `MAT-ExportResults.py`:
* `MAT-MergeDatasets.py`:
* `MAT-ResultsTo.py`:


### Citing

If you use `automatize` please cite the following paper:

    Portela, Tarlis Tortelli; Bogorny, Vania; Bernasconi, Anna; Renso, Chiara. AutoMATise: Multiple Aspect Trajectory Data Mining Tool Library. 2022 23rd IEEE International Conference on Mobile Data Management (MDM), 2022, pp. 282-285, doi: 10.1109/MDM55031.2022.00060.

[Bibtex](citation.bib):

```bash
@inproceedings{Portela2022automatise,
    title={AutoMATise: Multiple Aspect Trajectory Data Mining Tool Library},
    author={Portela, Tarlis Tortelli and Bogorny, Vania and Bernasconi, Anna and Renso, Chiara},
    booktitle = {2022 23rd IEEE International Conference on Mobile Data Management (MDM)},
    volume={},
    number={},
    address = {Online},
    year={2022},
    pages = {282--285},
    doi={10.1109/MDM55031.2022.00060}
}
```

### Collaborate with us

Any contribution is welcome. This is an active project and if you would like to include your algorithm in `automatize`, feel free to fork the project, open an issue and contact us.

Feel free to contribute in any form, such as scientific publications referencing `automatize`, teaching material and workshop videos.

### Related packages

- [scikit-mobility](https://github.com/scikit-mobility/scikit-mobility): Human trajectory representation and visualizations in Python;
- [geopandas](https://geopandas.org/en/stable/): Library to help work with geospatial data in Python;
- [movingpandas](https://anitagraser.github.io/movingpandas/): Based on `geopandas` for movement data exploration and analysis.
- [PyMove](https://github.com/InsightLab/PyMove): a Python library for processing and visualization of trajectories and other spatial-temporal data.

### Change Log

This is a more complete and second version of the previous package [Automatise](https://github.com/ttportela/automatise). 
 
*Nov. 2022:*
 - New Classifier: **Trajectory Ensemble Classifier (TEC)** - an ensemble classifier for trajectory data based in POIS, MARC and movelets methods;
 - [POI-F](https://doi.org/10.1145/3341105.3374045) extension: **POIS** is an extension to the POI-F method capable of concatenating dimensions and sequences for trajectory classification. Available for the methods `poi`, `npoi`, and `wnpoi`.
 - New classification methods: *TULVAE, BITULER, DeepestST, XGBoost, Traj. Random Forrest*
 - Scripts for classification refactored to command line best practices, and implemented random seed parameter to all methods for reproductibility purposes.
 
 *TODO*:
 - New visualization tools: 
 - New classification methods: MainTUL*
