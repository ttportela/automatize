import sys, os 
sys.path.insert(0, os.path.abspath(os.path.join('.')))
from automatize.results import check_run

if len(sys.argv) < 1:
    print('Please run as:')
    print('\tResultsTo.py', 'PATH TO RESULTS')
    print('Example:')
    print('\tResultsTo.py', '"./results"')
    exit()

res_path = sys.argv[1]

check_run(res_path)
print("Done.")
