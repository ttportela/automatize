### Xiao

\[ [Runnable](https://github.com/ttportela/automatize/blob/main/jarfiles/Xiao.jar?raw=true) \]

You can run the feature extractor with the following command:
```bash
java  JAVA_OPTS -jar Programs/Movelets/Xiao.jar -curpath DIR_PATH -respath RESULTS_DIR_PATH -descfile DATA_DIR_PATH/DESCRIPTOR_FILE.json -nt NUMBER_OF_THREADS
```

Example:
```bash
java -Xmx300g -jar Programs/Movelets/Xiao.jar -curpath "Datasets/Foursquare_nyc/run1" -respath "Results/Foursquare_nyc/run1/Xiao" -descfile "Datasets/DESCRIPTORS/spatialMovelets.json" -nt 8
```


##### Reference:

| Title | Authors | Year | Venue | Links | Cite |
|:------|:--------|------|:------|:------|:----:|
| Identifying different transportation modes from trajectory data using tree-based ensemble classifiers | Xiao, Z., Wang, Y., Fu, K., Wu, F. | 2017 | ISPRS International Journal of Geo-Information | [Article](https://doi.org/10.3390/ijgi6020057) [Repository](https://github.com/bigdata-ufsc/MASTERMovelets) |  |