# # from datetime import datetime
import sys, os 
sys.path.insert(0, os.path.abspath(os.path.join('.')))

# from automatize.main import importer
# importer(['S', 'ClassifierEnsemble2'], globals())
from automatize.ensemble.tec import ClassifierEnsemble2

if len(sys.argv) < 4:
    print('Please run as:')
    print('\tEnsemble2-cls.py', 'PATH TO DATASET', 'PATH TO RESULTS_DIR', 'ENSEMBLES', 'DATASET')
    print('Example:')
    print('\tEnsemble2-cls.py', '"./data"', '"./results"', '"{\'movelets\': \'./movelets-res\', \'pois\': \'./poifreq-res\'}"', 'specific')
    exit()

data_path = sys.argv[1]
results_path = sys.argv[2]
ensembles = eval(sys.argv[3])
dataset = sys.argv[4]

modelfolder='model'
if len(sys.argv) >= 4:
    modelfolder = sys.argv[5]
    
# time = datetime.now()
time_ext = ClassifierEnsemble2(data_path, results_path, ensembles, dataset, save_results=True, modelfolder=modelfolder)
# time_ext = (datetime.now()-time).total_seconds() * 1000

print("Done. Processing time: " + str(time_ext) + " milliseconds")
print("# ---------------------------------------------------------------------------------")