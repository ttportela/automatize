import pandas as pd
import sys, os 
sys.path.insert(0, os.path.abspath(os.path.join('.')))
import glob2 as glob
from automatize.results import containErrors, containWarnings

# from automatize.main import importer
# importer(['S', 'glob', 'containErrors', 'containWarnings'], globals())

if len(sys.argv) < 1:
    print('Please run as:')
    print('\tResultsTo.py', 'PATH TO RESULTS')
    print('Example:')
    print('\tResultsTo.py', '"./results"')
    exit()

results_path = sys.argv[1]

path = os.path.join(results_path, '**', '*.txt' )

# filelist = []
filesList = {}

# 1: Build up list of files:
print("Looking for result files in " + path)
for files in glob.glob(path):
    fileName, fileExtension = os.path.splitext(files)
#     filelist.append(os.path.dirname(fileName)) # parent
    filesList[os.path.dirname(fileName)] = files # filename with extension

for dirName, fileName in filesList.items(): 
    path = os.path.join(dirName, '*', '**', 'train.csv')
    ctc = 0
    for files in glob.glob(path):
        ctc += 1
    print('Classes: ', ctc, ' - Errors: ', containErrors(fileName), ' - Warnings: ', containWarnings(fileName), '-', dirName.replace(results_path, '...'))

print("Done.")