### Movelets

\[ [Runnable](https://github.com/ttportela/automatize/blob/main/jarfiles/Movelets.jar?raw=true) \]

Movelets is the method created before MASTERMovelets, it might be better for raw trajectories (spatial and time information). The difference is that MASTERMovelets is better to deal with multiple dimensions and to choose their best combinations.

You can run the feature extractor with the following command:
```Bash
java  JAVA_OPTS -jar movelets/Movelets.jar -curpath DIR_PATH -respath RESULTS_DIR_PATH -descfile DATA_DIR_PATH/DESCRIPTOR_FILE.json -nt NUMBER_OF_THREADS -q LSP -p false -Ms -1 -ms 1 
```

Example:
```Bash
java -Xmx300g -jar Programs/movelets/Movelets.jar -curpath "Datasets/Foursquare_nyc/run1" -respath "Results/Foursquare_nyc/run1/Movelets" -descfile "Datasets/DESCRIPTORS/spatialMovelets.json" -nt 8 -q LSP -p false -Ms -3 -ms 1
```

##### Reference:

| Title | Authors | Year | Venue | Links | Cite |
|:------|:--------|------|:------|:------|:----:|
| MOVELETS: Exploring Relevant Subtrajectories for Robust Trajectory Classification | Ferrero, C. A., Alvares, L. O., Zalewski, W., Bogorny, V. | 2018 | ACM Symposium on Applied Computing | [Article](https://dl.acm.org/citation.cfm?id=3167225) [Repository](https://github.com/bigdata-ufsc/ferrero-2018-movelets) | [BibTex](https://github.com/bigdata-ufsc/research-summary/blob/master/resources/bibtex/ferrero2018movelets.bib) |