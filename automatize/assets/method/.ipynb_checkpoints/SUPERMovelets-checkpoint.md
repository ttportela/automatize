### SUPERMovelets

\[ [Runnable](https://github.com/ttportela/automatize/blob/main/jarfiles/SUPERMovelets.jar?raw=true) \]

You can run the feature extractor with the following command:
```Bash
java JAVA_OPTS -jar movelets/SUPERMovelets.jar -curpath DIR_PATH -respath RESULTS_DIR_PATH -descfile DATA_DIR_PATH/DESCRIPTOR_FILE.json -nt NUMBER_OF_THREADS -ed true -ms MIN_SUBTRAJ_SIZE -Ms MAX_SUBTRAJ_SIZE -cache true -output discrete -samples 1 -sampleSize 0.5 -medium "none" -output "discrete" -lowm "false" 
```

Example:
```Bash
java -Xmx300g -jar Programs/movelets/SUPERMovelets.jar -curpath "Datasets/Foursquare_nyc/run1" -respath "Results/Foursquare_nyc/run1/SUPERMovelets" -descfile "Datasets/DESCRIPTORS/RawTraj_spatial.json" -nt 8 -ed true -ms 1 -Ms -3 -cache true -output discrete -samples 1 -sampleSize 0.5 -medium "none" -output "discrete" -lowm "false" 
```


##### Reference:

| Title | Authors | Year | Venue | Links | Cite |
|:------|:--------|------|:------|:------|:----:|
| Fast Movelet Extraction and Dimensionality Reduction for Robust Multiple Aspect Trajectory Classification | Portela, T. T., Silva, C. L., Carvalho, J. T., Bogorny, V. | 2021 | Brazilian Conference on Intelligent Systems (BRACIS) |  [Article](https://doi.org/10.1007/978-3-030-91702-9_31) [Repository](https://github.com/bigdata-ufsc/MASTERMovelets) | [BibTex](https://github.com/bigdata-ufsc/research-summary/blob/master/resources/bibtex/Portela2021supermovelets.bib) |