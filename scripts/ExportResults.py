import pandas as pd
import sys, os 
sys.path.insert(0, os.path.abspath(os.path.join('.')))
import glob2 as glob
import shutil
import tarfile

# from automatize.main import importer
# importer(['S', 'glob', 'shutil'], globals())

if len(sys.argv) < 2:
    print('Please run as:')
    print('\tExportResults.py', 'PATH TO RESULTS')
    print('Example:')
    print('\tExportResults.py', '"./results"')
    exit()

results_path = sys.argv[1]
to_file    = os.path.join(results_path, os.path.basename(os.path.normpath(results_path))+'.tgz')

def getFiles(path):
    filesList = []
    print("Looking for result files in " + path)
    for files in glob.glob(path):
        fileName, fileExtension = os.path.splitext(files)
#         filelist.append(fileName) #filename without extension
        filesList.append(files) #filename with extension

    return filesList

filelist = ['*.txt', 'classification_times.csv', '*_history_Step5.csv', '*_history.csv', '*_results.csv']
filesList = []

for file in filelist:
    path = os.path.join(results_path, '**', file)
    filesList = filesList + getFiles(path)

filesList = list(set(filesList))

with tarfile.open(to_file, "w:gz") as tar:
    for source in filesList:
        target = source.replace(results_path, '')
        print('Add:', target)
        tar.add(source, arcname=target)

print("Done.")
