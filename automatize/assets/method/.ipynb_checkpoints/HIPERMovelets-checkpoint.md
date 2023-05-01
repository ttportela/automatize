### HiPerMovelets

\[ [Runnable](https://github.com/ttportela/automatize/blob/main/jarfiles/HIPERMovelets.jar?raw=true) \]


**HIPERMovelets: high-performance movelet extraction for trajectory classification** is a new method for movelets discovery, developed as an optimization for MASTERMovelets.

You can run the feature extractor with the following command:
 
```Bash
    java JAVA_OPTS -jar HIPERMovelets.jar -curpath DIR_PATH -respath RESULTS_DIR_PATH 
    -descfile DATA_DIR_PATH/DESCRIPTOR_FILE.json -nt NUMBER_OF_THREADS -ed true 
    -ms MIN_SUBTRAJ_SIZE -Ms MAX_SUBTRAJ_SIZE -cache true -output discrete 
    -samples 1 -sampleSize 0.5 -medium "none" -output "discrete" -lowm "false"
    -version hiper|hiper-pivots
```

**HiPerMovelets** command example to run:
```Bash
    java -Xmx300g -jar programs/HIPERMovelets.jar 
    -curpath "datasets/multiple_trajectories/Foursquare_nyc/run1" 
    -respath "results/Foursquare_nyc/run1/HIPERMovelets" 
    -descfile "datasets/multiple_trajectories/descriptors/FoursquareNYC_specific_hp.json" 
    -nt 8 -version hiper 
```

**HiPerMovelets-Pivots** command example to run:
```Bash
    java -Xmx300g -jar programs/HIPERMovelets.jar 
    -curpath "datasets/multiple_trajectories/Foursquare_nyc/run1" 
    -respath "results/Foursquare_nyc/run1/HIPERMovelets-Pivots" 
    -descfile "datasets/multiple_trajectories/descriptors/FoursquareNYC_specific_hp.json" 
    -nt 8 -version hiper-pivots
```

#### 1.1 Versions

This implementation has three options of optimizations.

- *HIPERMovelets*: the greedy search method (`-version hiper`).
- *HIPERMovelets-Log*: the greedy search method, plus, limits the movelets size to the ln size of the trajectory (`-version hiper -Ms -3`).
- *HIPERMovelets-Pivots*: limits the movelets search space to the points that are neighbour of well qualified movelets of size one (`-version hiper-pivots`).
- *HIPERMovelets-Pivots-Log*: plus, limits the movelets size to the ln size of the trajectory (`-version hiper-pivots -Ms -3`).

##### Reference:

| Title | Authors | Year | Venue | Links | Cite |
|:------|:--------|------|:------|:------|:----:|
| HiPerMovelets: high-performance movelet extraction for trajectory classification | Portela, T. T.; Carvalho, J. T.; Bogorny, V. | 2022 | International Journal of Geographical Information Science | [Article](https://doi.org/10.1080/13658816.2021.2018593) [Repository](https://github.com/bigdata-ufsc/HiPerMovelets) | [BibTex](https://github.com/bigdata-ufsc/research-summary/blob/master/resources/bibtex/Portela2020hipermovelets.bib) |