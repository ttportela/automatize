### Zheng

\[ [Runnable](https://github.com/ttportela/automatize/blob/main/jarfiles/Zheng.jar?raw=true) \]

You can run the feature extractor with the following command:
```bash
java  JAVA_OPTS -jar Programs/Movelets/Zheng.jar -curpath DIR_PATH -respath RESULTS_DIR_PATH -descfile DATA_DIR_PATH/DESCRIPTOR_FILE.json -nt NUMBER_OF_THREADS
```

Example:
```bash
java -Xmx300g -jar Programs/Movelets/Zheng.jar -curpath "Datasets/Foursquare_nyc/run1" -respath "Results/Foursquare_nyc/run1/Zheng" -descfile "Datasets/DESCRIPTORS/spatialMovelets.json" -nt 8
```

##### Reference:

| Title | Authors | Year | Venue | Links | Cite |
|:------|:--------|------|:------|:------|:----:|
| Understanding transportation modes based on GPS data for web applications | Zheng, Y., Chen, Y., Li, Q., Xie, X., Ma, W. Y. | 2010 | ACM Transactions on the Web, Volume 4, Issue 1 | [Article](https://doi.org/10.1145/1658373.1658374) [Repository](https://github.com/bigdata-ufsc/MASTERMovelets) |  |