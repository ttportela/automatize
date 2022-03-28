import sys, os 
sys.path.insert(0, os.path.abspath(os.path.join('.')))
import glob2 as glob

# from automatize.main import importer
# importer(['S', 'mergeDatasets'], globals())
from automatize.run import moveResults

if len(sys.argv) < 1:
    print('Please run as:')
    print('\tMergeDatasets.py', 'PATH TO FOLDER')
    print('Example:')
    print('\tensemble-cls.py', '"./results/HiPerMovelets"')
    exit()

results_path = sys.argv[1]

dir_from = os.path.dirname(glob.glob(os.path.join(results_path, '**', 'train.csv'))[0])
# print(os.path.dirname(glob.glob(os.path.join(results_path, '**', 'train.csv'))[0]))

moveResults(dir_from, results_path)
