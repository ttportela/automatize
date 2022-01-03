3. MASTERMovelets (Movelets > MASTERMovelets.jar)
https://github.com/bigdata-ufsc/MASTERMovelets

You can run the feature extractor with the following command:
java JAVA_OPTS -jar movelets/MASTERMovelets.jar -curpath DIR_PATH -respath RESULTS_DIR_PATH -descfile DATA_DIR_PATH/DESCRIPTOR_FILE.json -nt NUMBER_OF_THREADS -ed true -ms MIN_SUBTRAJ_SIZE -Ms MAX_SUBTRAJ_SIZE -cache true -output discrete -samples 1 -sampleSize 0.5 -medium "none" -output "discrete" -lowm "false" 

Example:
java -Xmx300g -jar Programs/movelets/MASTERMovelets.jar -curpath "Datasets/Foursquare_nyc/run1" -respath "Results/Foursquare_nyc/run1/MASTERMovelets" -descfile "Datasets/DESCRIPTORS/RawTraj_spatial.json" -nt 8 -ed true -ms 1 -Ms -3 -cache true -output discrete -samples 1 -sampleSize 0.5 -medium "none" -output "discrete" -lowm "false" 