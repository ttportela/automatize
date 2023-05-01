### Methods for trajectory classification

1. [HIPERMovelets](/method/HIPERMovelets) ![Method-Badge](https://img.shields.io/badge/Feature_Extraction-Movelets-brightgreen.svg)

    Portela T.T., Carvalho J.T., Bogorny V. **HIPERMovelets: high-performance movelet extraction for trajectory classification**. International Journal of Geographical Information Science, 2021. DOI: https://doi.org/10.1080/13658816.2021.2018593
    
1. [SUPERMovelets](/method/SUPERMovelets) ![Method-Badge](https://img.shields.io/badge/Feature_Extraction-Movelets-brightgreen.svg)

    Portela T.T., da Silva C.L., Carvalho J.T., Bogorny V. **Fast Movelet Extraction and Dimensionality Reduction for Robust Multiple Aspect Trajectory Classification**. In: Britto A., Valdivia Delgado K. (eds) Intelligent Systems. BRACIS 2021. Lecture Notes in Computer Science, vol 13073. Springer, Cham. DOI: https://doi.org/10.1007/978-3-030-91702-9_31

1. [MARC](/method/MARC) ![Method-Badge](https://img.shields.io/badge/Classifier-orange.svg)

    May Petry L., Leite Da Silva C., Esuli A., Renso C., Bogorny V. **MARC: a robust method for multiple-aspect trajectory classification via space, time, and semantic embeddings**. International Journal of Geographical Information Science, 2020, 34:7, 1428-1450. DOI: https://doi.org/10.1080/13658816.2019.1707835

1. [POI-F](/method/POIS) ![Method-Badge](https://img.shields.io/badge/Feature_Extraction-Sequences-blue.svg)

    Vicenzi F., May Petry L., Leite da Silva C., Alvares L.O., Bogorny V. **Exploring frequency-based approaches for efficient trajectory classification**. In Proceedings of the 35th Annual ACM Symposium on Applied Computing (SAC '20), 2020. Association for Computing Machinery, New York, NY, USA, 624–631. DOI: https://doi.org/10.1145/3341105.3374045

1. [MASTERMovelets](/method/MASTERMovelets) ![Method-Badge](https://img.shields.io/badge/Feature_Extraction-Movelets-brightgreen.svg)

    Ferrero, C. A., Petry, L. M., Alvares, L. O., Silva, C. L., Zalewski, W., Bogorny, V. **MasterMovelets: Discovering Heterogeneous Movelets for Multiple Aspect Trajectory Classification**. Data Mining and Knowledge Discovery, 2020, 34(3), 652-680. DOI: https://doi.org/10.1007/s10618-020-00676-x

1. [Movelets](/method/Movelets) ![Method-Badge](https://img.shields.io/badge/Feature_Extraction-Movelets-brightgreen.svg)

    Ferrero, C. A., Alvares, L. O., Zalewski, W., Bogorny, V. **MOVELETS: Exploring Relevant Subtrajectories for Robust Trajectory Classification**. In Proceedings of the 33rd Annual ACM Symposium on Applied Computing (SAC '18). Association for Computing Machinery, New York, NY, USA, 849–856. DOI: https://doi.org/10.1145/3167132.3167225

1. [DeepeST](/method/DeepeST) ![Method-Badge](https://img.shields.io/badge/Classifier-orange.svg)

    Nicksson A. de Freitas.; Ticiana Coelho da Silva.; José Fernandes de Macêdo.; Leopoldo Melo Junior.; Matheus Cordeiro.. Using Deep Learning for Trajectory Classification. In Proceedings of the 13th International Conference on Agents and Artificial Intelligence (ICAART 2021). 2021.

1. [Xiao](/method/Xiao) ![Method-Badge](https://img.shields.io/badge/Feature_Extraction-Other-yellow.svg)

    Xiao, Z., Wang, Y., Fu, K., & Wu, F. (2017). Identifying different transportation modes from trajectory data using tree-based ensemble classifiers. ISPRS International Journal of Geo-Information, 6(2), 57. DOI: https://doi.org/10.3390/ijgi6020057

1. [Zheng](/method/Zheng) ![Method-Badge](https://img.shields.io/badge/Feature_Extraction-Other-yellow.svg)

    Zheng, Y., Chen, Y., Li, Q., Xie, X., & Ma, W. Y. (2010). Understanding transportation modes based on GPS data for web applications. ACM Transactions on the Web (TWEB), 4(1), 1-36. DOI: https://doi.org/10.1145/1658373.1658374

1. [Dodge](/method/Dodge) ![Method-Badge](https://img.shields.io/badge/Feature_Extraction-Other-yellow.svg)

    Dodge, S., Weibel, R., & Forootan, E. (2009). Revealing the physics of movement: Comparing the similarity of movement characteristics of different types of moving objects. Computers, Environment and Urban Systems, 33(6), 419-434. DOI: https://doi.org/10.1016/j.compenvurbsys.2009.07.008


____________________________________________________________________
### Setup

A. In order to run the .jar files you first need to install Java 8 (or superior). Be sure to have enough RAM memory available. 
Some of these methods are included in this library.

____________________________________________________________________
(Optional) 

B. If you opt to use the test automatization in Python (this project), you first need to install Python, and dependencies. 

See `requirements.txt` to Python dependencies. To install all dependencies you can use:

```Bash
    pip install -r ./requirements.txt
```


#### Available Scripts (for command line)

Auxiliary command tools for method results:

- `MAT-CheckRun.py`: displays completed methods and possible running errors from provided results folder;
- `MAT-MergeDatasets.py`: merge resulting files of extracted features of each class into one `train.csv` and one `test.csv` files;
- `MAT-ExportResults.py`: exports classification result files and method log files to a compaceted .zip file;
- `MAT-ExportMovelets.py`: exports movelets JSON files to a compaceted .zip file;
- `MAT-ResultsTo.py`: copy classification result files and method log files to another root folder;


Auxiliary command tools for running common classifiers:

- `MAT-MC.py`: run all movelet classifiers (MLP, RF and SVM);
- `MAT-TC.py`: run trajectory classifiers (MARC,TRF,TXGB,DEEPEST,BITULER,TULVAE);

- `MARC.py`: run MARC method;
- `POIS-TC.py`: run MLP classifier for `poi`, `npoi` or `wnpoi` output results;


Command tools for running incorporated methods:

- `MARC.py`: run MARC method;
- `POIS.py`: run POI-F and POIS method (`poi`, `npoi` or `wnpoi`);

