### Dodge

\[ [Runnable](https://github.com/ttportela/automatize/blob/main/jarfiles/Dodge.jar?raw=true) \]

You can run the feature extractor with the following command:
```bash
java  JAVA_OPTS -jar Programs/Movelets/Dodge.jar -curpath DIR_PATH -respath RESULTS_DIR_PATH -descfile DATA_DIR_PATH/DESCRIPTOR_FILE.json -nt NUMBER_OF_THREADS
```

Example:
```bash
java -Xmx300g -jar Programs/Movelets/Dodge.jar -curpath "Datasets/Foursquare/run1" -respath "Results/Foursquare/run1/Dodge" -descfile "Datasets/DESCRIPTORS/spatialMovelets.json" -nt 8
```


##### Reference:

| Title | Authors | Year | Venue | Links | Cite |
|:------|:--------|------|:------|:------|:----:|
| Revealing the physics of movement: Comparing the similarity of movement characteristics of different types of moving object | Dodge, S., Weibel, R., Forootan, E. | 2009 | Computers, Environment and Urban Systems | [Article](http://dx.doi.org/10.1016/j.compenvurbsys.2009.07.008) [Repository](https://github.com/bigdata-ufsc/MASTERMovelets) |  |