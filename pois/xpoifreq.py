# Deprecated script (renamed)

import sys, os 
script_dir = os.path.dirname( __file__ )
main_dir = os.path.abspath(os.path.join( script_dir, '..' , '..'))
sys.path.append( main_dir )

importer(['S', 'datetime','poifreq'], globals())

if len(sys.argv) < 6:
    print('Please run as:')
    print('\tpoifreq.py', 'METHOD', 'SEQUENCES', 'FEATURES', 'DATASET', 'PATH TO DATASET', 'PATH TO RESULTS_DIR')
    print('Example:')
    print('\tpoifreq.py', 'npoi', '"1,2,3"', '"poi,hour"', 'specific', '"./data"', '"./results"')
    exit()

METHOD = sys.argv[1]
SEQUENCES = [int(x) for x in sys.argv[2].split(',')]
FEATURES = sys.argv[3].split(',')
DATASET = sys.argv[4]
path_name = sys.argv[5]
RESULTS_DIR = sys.argv[6]

# from automatize.ensemble_models.poifreq import poifreq
time = datetime.now()
poifreq(SEQUENCES, DATASET, FEATURES, path_name, RESULTS_DIR, method=METHOD, save_all=True, doclass=True)
time_ext = (datetime.now()-time).total_seconds() * 1000

# name = ('_'.join(FEATURES))+'_'+('_'.join([str(n) for n in SEQUENCES]))+'_'+DATASET
# OUTPUT_FILE = METHOD #'classification-results'
# time = datetime.now()
# from automatize.ensemble_models.poifreq_model import model_poifreq
# os.system('automatize/poifreq/classification-p.py "npoi" "'+name+'" "'+RESULTS_DIR+'" "'+OUTPUT_FILE+'"')
# time_cls = (datetime.now()-time).total_seconds() * 1000

# f=open(os.path.join(RESULTS_DIR, OUTPUT_FILE+'.txt'), "a+")
# f.write("Processing time: %d milliseconds\r\n" % (time_ext))
# f.write("Classification time: %d milliseconds\r\n" % (time_cls))
# f.write("Total time: %d milliseconds\r\n" % (time_ext+time_cls))
# f.close()

print("Done. Processing time: " + str(time_ext) + " milliseconds")
print("# ---------------------------------------------------------------------------------")