import pandas as pd
import glob2 as glob
import sys, os 
sys.path.insert(0, os.path.abspath(os.path.join('.')))

# from automatize.main import importer
# importer(['S'], globals())

if len(sys.argv) < 1:
    print('Please run as:')
    print('\tMergeClasses.py', 'PATH TO FOLDER')
    print('Example:')
    print('\tMergeClasses.py', '"./results/MASTERMovelets"')
    exit()

results_path = sys.argv[1]

# --------------------------------------------------------------------------------------
def mergeDatasets(dir_path, file='train.csv'):
    files = [i for i in glob.glob(os.path.join(dir_path, '*', '**', file))]

    print("Loading files - " + file)
    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f).drop('class', axis=1) for f in files[:len(files)-1]], axis=1)
    combined_csv = pd.concat([combined_csv, pd.read_csv(files[len(files)-1])], axis=1)
    #export to csv
    print("Writing "+file+" file")
    combined_csv.to_csv(os.path.join(dir_path, file), index=False)
    
    print("Done.")

mergeDatasets(results_path, 'train.csv')
mergeDatasets(results_path, 'test.csv')