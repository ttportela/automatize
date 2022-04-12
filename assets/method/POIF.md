### POI-F 

\[ *Runnable is included in Automatise Library* \]

POI Frequency is a TF-IDF approach, the method runs one dimension at a time (or more if concatenated). You can compare it, but that is something that needs to be discussed on how to do it. This implementation I made from the original pyhton notebook, so I could run as a script. 
If you go to the original paper, you will see that are 3 approaches: poi, npoi and wnpoi. I have been using only npoi (as in the example), but feel free to try each one.

You can run the classifier with the following command:
```Bash
POIS.py 'METHOD' 'SEQUENCE_SIZE', 'FEATURE' DATASET_PREFIX TRAIN_FILE TEST_FILE RESULTS_FILE DATASET_NAME EMBEDDING_SIZE MERGE_TYPE RNN_CELL
```

Example:
```Bash
POIS.py "npoi" "1" "lat_lon" "specific" "Datasets/Foursquare_nyc/run1" "Results/NPOI_lat_lon_1-specific"
```


##### Reference:

| Title | Authors | Year | Venue | Links | Cite |
|:------|:--------|------|:------|:------|:----:|
| Exploring frequency-based approaches for efficient trajectory classification | Vicenzi, F., May Petry, L., Silva, C. L., Alvares, L. O., Bogorny, V. | 2020 | SAC '20: Proceedings of the 35th Annual ACM Symposium on Applied Computing |  [Article](https://doi.org/10.1145/3341105.3374045) [Repository](https://github.com/bigdata-ufsc/vicenzi-2020-poifreq) | [BibTex](https://github.com/bigdata-ufsc/research-summary/blob/master/resources/bibtex/Vicenzi2020poif.bib) |