### MARC

\[ *Runnable is included in Automatize Library* \]

It will give you the classification results in the results.csv file (as the example). As you need the running time, you can add code in the python script to calculate the time.

You can run the classifier with the following command (in marc folder):
```Bash
python automatize/marc/multi_feature_classifier.py TRAIN_FILE TEST_FILE RESULTS_FILE DATASET_NAME EMBEDDING_SIZE MERGE_TYPE RNN_CELL
```

Example:
```Bash
python automatize/marc/multi_feature_classifier.py Datasets/Foursquare_nyc/run1/specific_train.csv Datasets/Foursquare_nyc/run1/specific_test.csv results.csv FoursquareNYC 100 concatenate lstm 
```

##### Reference:

| Title | Authors | Year | Venue | Links | Cite |
|:------|:--------|------|:------|:------|:----:|
| MARC: a robust method for multiple-aspect trajectory classification via space, time, and semantic embeddings | May Petry, L., Silva, C. L., Esuli, A., Renso, C., Bogorny, V. | 2019 | International Journal of Geographical Information Science |  [Article](https://doi.org/10.1080/13658816.2019.1707835) [Repository](https://github.com/bigdata-ufsc/petry-2020-marc) | [BibTex](https://github.com/bigdata-ufsc/research-summary/blob/master/resources/bibtex/MayPetry2019marc.bib) |