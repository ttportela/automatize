### POIS / POI-F

\[ *Runnable is included in Library* \]

POI Frequency is a TF-IDF approach, the method runs one dimension at a time (or more if concatenated). You can compare it, but that is something that needs to be discussed on how to do it. This implementation I made from the original pyhton notebook, so I could run as a script. 
If you go to the original paper, you will see that are 3 approaches: poi, npoi and wnpoi. I have been using only npoi (as in the example), but feel free to try each one.

You can run the feature extractor with the following command:
```Bash
POIS.py -m "METHOD" -s "SEQUENCE_SIZE" -a "FEATURE" -d "DATASET_PREFIX" -f "RESULTS_DIR" "DATAPATH"
```

Example:
```Bash
POIS.py -m "npoi" -s "1,2" -a "poi" -d "specific" -f "Results/NPOI_poi_1-specific" "Datasets/Foursquare_nyc/run1" 
```

You can run the classifier for each sequence size with the following command:
```Bash
POIS-TC.py -m "npoi" -p "poi_1" -f "NPOI-poi_1" "Results/NPOI_poi_1-specific" 
POIS-TC.py -m "npoi" -p "poi_2" -f "NPOI-poi_2" "Results/NPOI_poi_1-specific" 
POIS-TC.py -m "npoi" -p "poi_1_2" -f "NPOI-poi_1_2" "Results/NPOI_poi_1-specific" 
```

##### Reference:

| Title | Authors | Year | Venue | Links | Cite |
|:------|:--------|------|:------|:------|:----:|
| Exploring frequency-based approaches for efficient trajectory classification | Vicenzi, F., May Petry, L., Silva, C. L., Alvares, L. O., Bogorny, V. | 2020 | SAC '20: Proceedings of the 35th Annual ACM Symposium on Applied Computing |  [Article](https://doi.org/10.1145/3341105.3374045) [Repository](https://github.com/bigdata-ufsc/vicenzi-2020-poifreq) | [BibTex](https://github.com/bigdata-ufsc/research-summary/blob/master/resources/bibtex/Vicenzi2020poif.bib) |