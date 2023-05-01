### MASTERMovelets

\[ [Runnable](https://github.com/ttportela/automatize/blob/main/jarfiles/MASTERMovelets.jar?raw=true) \]

You can run the feature extractor with the following command:

```bash
java JAVA_OPTS -jar movelets/MASTERMovelets.jar -curpath DIR_PATH -respath RESULTS_DIR_PATH -descfile DATA_DIR_PATH/DESCRIPTOR_FILE.json -nt NUMBER_OF_THREADS -ed true -ms MIN_SUBTRAJ_SIZE -Ms MAX_SUBTRAJ_SIZE -cache true -output discrete -samples 1 -sampleSize 0.5 -medium "none" -output "discrete" -lowm "false" 
```

Example:
```bash
java -Xmx300g -jar Programs/movelets/MASTERMovelets.jar -curpath "Datasets/Foursquare_nyc/run1" -respath "Results/Foursquare_nyc/run1/MASTERMovelets" -descfile "Datasets/DESCRIPTORS/RawTraj_spatial.json" -nt 8 -ed true -ms 1 -Ms -3 -cache true -output discrete -samples 1 -sampleSize 0.5 -medium "none" -output "discrete" -lowm "false" 
```

##### Reference:

| Title | Authors | Year | Venue | Links | Cite |
|:------|:--------|------|:------|:------|:----:|
| MasterMovelets: Discovering Heterogeneous Movelets for Multiple Aspect Trajectory Classification | Ferrero, C. A., Petry, L. M., Alvares, L. O., Silva, C. L., Zalewski, W., Bogorny, V. | 2020 | Data Mining and Knowledge Discovery | [Article](https://link.springer.com/article/10.1007/s10618-020-00676-x) [Repository](https://github.com/bigdata-ufsc/MASTERMovelets) | [BibTex](https://github.com/bigdata-ufsc/research-summary/blob/master/resources/bibtex/Ferrero2020mastermovelets.bib) |