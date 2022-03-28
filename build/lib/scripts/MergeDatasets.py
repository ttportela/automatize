import sys, os 
sys.path.insert(0, os.path.abspath(os.path.join('.')))

# from automatize.main import importer
# importer(['S', 'mergeDatasets'], globals())
from automatize.run import mergeDatasets

if len(sys.argv) < 1:
    print('Please run as:')
    print('\tMergeDatasets.py', 'PATH TO FOLDER')
    print('Example:')
    print('\tensemble-cls.py', '"./results/HiPerMovelets"')
    exit()

results_path = sys.argv[1]


mergeDatasets(results_path, 'train.csv')
mergeDatasets(results_path, 'test.csv')