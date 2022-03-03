'''
Created on Aug, 2020

@author: Tarlis Portela
'''
#TODO: RF não pode fazer aquele reshape?
#TODO: MARC Model
from ..main import importer #, display
importer(['S'], globals())

# --------------------------------------------------------------------------------
# ANALYSIS By Ensemble Learning Models

# def Ensemble(data_path, results_path, postfix, methods=['movelets','poifreq'], \
#              modelfolder='model', save_results=True, print_only=False, py_name='python3', \
#              descriptor='', sequences=[1,2,3], features=['poi'], dataset='specific', num_runs=1,\
#              movelets_line=None):
#     import os
    
#     ensembles = dict()
#     for method in methods:
#         if method is 'poi' or method is 'npoi' or method is 'wnpoi':
#             from automatize.run import POIFREQ
# #             sequences = [2, 3]
# #             features  = ['sequence']
# #             results_npoi = os.path.join(results_path, prefix, 'npoi')
#             prefix = ''
#             core_name = POIFREQ(data_path, results_path, prefix, dataset, sequences, features, \
#                                 print_only=print_only, doclass=False)
#             ensembles['npoi'] = core_name
            
#         elif method == 'marc':
#             ensembles['marc'] = data_path
            
#         elif method == 'rf':
#             ensembles['rf'] = data_path
            
#         else: # the method is 'movelets':
#             if movelets_line is None:
#                 from automatize.run import Movelets
#                 mname = method.upper()+'L-'+dataset
#                 prefix = ''
#                 Movelets(data_path, results_path, prefix, mname, descriptor, Ms=-3, \
#                          extra='-T 0.9 -BU 0.1 -version '+method, \
#                          print_only=print_only, jar_name='HIPERMovelets2', n_threads=4, java_opts='-Xmx60G')
#                 ensembles['movelets'] = os.path.join(results_path, prefix, mname)
#             else:
#                 ensembles['movelets'] = movelets_line
                     
#     if print_only:
#         if num_runs == 1:
#             CMD = py_name + " automatize/Ensemble-cls.py "
#             CMD = CMD + "\""+data_path+"\" "
#             CMD = CMD + "\""+os.path.join(results_path, postfix)+"\" "
#             CMD = CMD + "\""+str(ensembles)+"\" "
#             CMD = CMD + "\""+dataset+"\" "
#             print(CMD)
#             print('')
#         else:
#             for i in range(1, num_runs+1):
#                 print('# Classifier Ensemble run-'+str(i))
#                 CMD = py_name + " automatize/Ensemble-cls.py "
#                 CMD = CMD + "\""+data_path+"\" "
#                 CMD = CMD + "\""+os.path.join(results_path, postfix)+"\" "
#                 CMD = CMD + "\""+str(ensembles)+"\" "
#                 CMD = CMD + "\""+dataset+"\" "
#                 CMD = CMD + "\""+modelfolder+'-'+str(i)+"\" "
#                 print(CMD)
#                 print('')
#     else:
#         return ClassifierEnsemble(data_path, results_path, ensembles, dataset, save_results, modelfolder)

def ClassifierEnsemble(data_path, results_path, ensembles, dataset='specific', save_results=True, modelfolder='model_ensemble'):
#     from ..main import importer
    importer(['S', 'datetime', 'tf', 'TEC.report', 'KerasClassifier', 'readDataset'], globals())
    
#     import os
#     import pandas as pd
#     import numpy as np
#     from datetime import datetime
#     import tensorflow
    tf.keras.backend.clear_session()
    
    print('[Ensemble]: Loading base data description.')
    if dataset == '':
        TRAIN_FILE = os.path.join(data_path, 'train.csv')
        TEST_FILE  = os.path.join(data_path, 'test.csv')
    else:
        TRAIN_FILE = os.path.join(data_path, dataset+'_train.csv')
        TEST_FILE  = os.path.join(data_path, dataset+'_test.csv')
        
#     print(TRAIN_FILE)
    df_test = readDataset(TEST_FILE, missing='-999') #pd.read_csv(TEST_FILE)
    y_test = df_test.drop_duplicates(subset=['tid', 'label'],
                                       inplace=False)['label'].values

    keys = list(pd.unique(y_test))
    keys.sort()
    y_labels = [keys.index(x) for x in y_test]

#     from keras.wrappers.scikit_learn import KerasClassifier
    
    time = datetime.now()
    # create the sub models
    models = dict()
    estimators = []
    print('[Ensemble]: '+', '.join(ensembles.keys()))
    for method, folder in ensembles.items():
        ctime = datetime.now()
        y_pred = []
        model = []
        if method == 'movelets_mlp':
#             from ..main import importer
            importer(['TEC.MLP'], globals())
#             from automatize.ensemble_models.movelets import model_movelets_mlp
#             from numpy import argmax
            model, x_test, y_test_pred, y_test_true = model_movelets_mlp(folder)
#             y_test_pred_dec = argmax( y_test_pred_dec , axis = 1)
#             y_pred = [keys.index(x) for x in y_test_pred_dec]
#             return y_labels, x_test, y_test_pred, y_test_true
#             from automatize.Methods import f1
            models[method] = KerasClassifier(model)
            model = y_test_pred
#             model = model.predict(x_test)
#             y_pred = [np.argmax(f) for f in y_test_pred_dec] 

        if method == 'movelets_nn' or method == 'movelets':
#             from ..main import importer
            importer(['TEC.NN'], globals())
#             from automatize.ensemble_models.movelets import model_movelets_nn
            model, x_test = model_movelets_nn(folder)
#             from automatize.Methods import f1
            models[method] = KerasClassifier(model)
            model = model.predict(x_test)

        if method == 'marc':
#             from ..main import importer
            importer(['TEC.MARC'], globals())
#             from automatize.ensemble_models.marc2 import model_marc
            model, x_test = model_marc(folder, results_path, dataset)
            models[method] = KerasClassifier(model)
            model = model.predict(x_test)
            
        if method == 'poi' or method == 'npoi' or method == 'wnpoi':
#             from ..main import importer
            importer(['TEC.POIS'], globals())
#             from automatize.ensemble_models.poifreq import model_poifreq
            model, x_test = model_poifreq(folder)
            models[method] = KerasClassifier(model)
            model = model.predict(x_test)
            
        if method == 'rf':
#             from ..main import importer
            importer(['TEC.RF'], globals())
#             from automatize.ensemble_models.randomforrest import model_rf
            model, x_test = model_rf(folder, dataset)
            models[method] = model
            model = model.predict(x_test)
            
        if method == 'rfhp':
#             from ..main import importer
            importer(['TEC.RFHP'], globals())
#             from automatize.ensemble_models.randomforresthp import model_rfhp
            model, x_test = model_rfhp(folder, dataset)
            models[method] = model
            model = model.predict(x_test)
        
#         print(method, 'ESTIMATORS:', model)
        y_pred = [np.argmax(f) for f in model]  
        ctime = (datetime.now()-ctime).total_seconds() * 1000
        
#         return model, y_pred
#         print(method, 'PRED:', y_pred)
        estimators.append(model) 
        ensembles[method] = get_line(y_labels, y_pred, model) + [ctime]
        print(method+': ', ensembles[method])
        print("---------------------------------------------------------------------------------")
    
#     print(estimators)
    final_pred = estimators[0]
    for i in range(1, len(estimators)):
        final_pred = final_pred + estimators[i]
    y_pred = [np.argmax(f) for f in final_pred]
    # ----------------------------------------------------------------------------------
    time = (datetime.now()-time).total_seconds() * 1000
    
    print('[Ensemble]: Final results.')
    print(ensembles)
    line=get_line(y_labels, y_pred, final_pred) + [time]
    ensembles['EnsembleClassifier'] = line
    print('[Ensemble]:', line)
    
    # Correct Class names:
    y_labels = [keys[f] for f in y_labels]
    y_pred = [keys[f] for f in y_pred]
    ensembles = [[key] + ensembles[key] for key in ensembles.keys()]
    # ---------------------------------------------------------------------------------
    if (save_results) :
        if not os.path.exists(os.path.join(results_path, modelfolder)):
            os.makedirs(os.path.join(results_path, modelfolder))
#         from sklearn.metrics import classification_report
#         from automatize.Methods import classification_report_csv
        report = classification_report(y_labels, y_pred)
        classification_report_csv(report, os.path.join(results_path, modelfolder, "model_approachEnsemble_report.csv"),"Ensemble") 
        pd.DataFrame(ensembles, columns=['classifier', 'accuracy', 'f1_score', 'precision', 'recall', 'accTop5', 'time']).to_csv(os.path.join(results_path, modelfolder, "model_approachEnsemble_history.csv")) 

    # ---------------------------------------------------------------------------------
    print("Done. " + str(time) + " milliseconds")
    print("---------------------------------------------------------------------------------")
    return time

def ClassifierEnsemble2(data_path, results_path, ensembles, dataset='specific', save_results=True, modelfolder='model_ensemble'):
#     from ..main import importer
    importer(['S', 'datetime', 'tf', 'KerasClassifier', 'A2'], globals())
    # V2.0 - Concatenate data from POI-S and Movelets (pois, movelets)
#     import os
#     import pandas as pd
#     import numpy as np
#     from datetime import datetime
#     import tensorflow
    tf.keras.backend.clear_session()

#     from keras.wrappers.scikit_learn import KerasClassifier
    
    time = datetime.now()
    # Concatenate:
    dir_path = ensembles['pois']
    print("Loading train and test data from... " + dir_path)
    x_train_1 = pd.read_csv(dir_path+'-x_train.csv', header=None)
    y_train = pd.read_csv(dir_path+'-y_train.csv')

    x_test_1 = pd.read_csv(dir_path+'-x_test.csv', header=None)
    y_test = pd.read_csv(dir_path+'-y_test.csv')
    
    dir_path = ensembles['movelets']
    print("Loading train and test data from... " + dir_path)
    dataset_train = pd.read_csv(os.path.join(dir_path, "train.csv"))
    dataset_test  = pd.read_csv(os.path.join(dir_path, "test.csv"))

    nattr = len(dataset_train.iloc[1,:])
    print("Number of attributes: " + str(nattr))

    # Separating attribute data (X) than class attribute (y)
    x_train_2 = dataset_train.iloc[:, 0:(nattr-1)].values
#     y_train_2 = dataset_train.iloc[:, (nattr-1)].values
    x_test_2 = dataset_test.iloc[:, 0:(nattr-1)].values
#     y_test_2 = dataset_test.iloc[:, (nattr-1)].values

    # Replace distance 0 for presence 1
    # and distance 2 to non presence 0
    x_train_2[x_train_2 == 0] = 1
    x_train_2[x_train_2 == 2] = 0
    x_test_2[x_test_2 == 0] = 1
    x_test_2[x_test_2 == 2] = 0
    
    x_train = pd.concat([x_train_1, pd.DataFrame(x_train_2)], axis=1).values
    x_test = pd.concat([x_test_1, pd.DataFrame(x_test_2)], axis=1).values
    print("Done.")
    
    # ---------------------------------------------------------------------------
    # Neural Network - Definitions:
    par_droupout = 0.5
    par_batch_size = 200
    par_epochs = 80
    par_lr = 0.00095
    
    # Building the neural network-
    print("Building neural network")
    lst_par_epochs = [80,50,50,30,20]
    lst_par_lr = [0.00095,0.00075,0.00055,0.00025,0.00015]
    
#     from automatize.Methods import Approach2
    Approach2(x_train, y_train, x_test, y_test, par_batch_size, lst_par_epochs, lst_par_lr, par_droupout, save_results, results_path, modelfolder)
    # ----------------------------------------------------------------------------------
    time = (datetime.now()-time).total_seconds() * 1000
    # ---------------------------------------------------------------------------------
#     if (save_results) :
#         if not os.path.exists(os.path.join(results_path, modelfolder)):
#             os.makedirs(os.path.join(results_path, modelfolder))
#         from sklearn.metrics import classification_report
#         from automatize.Methods import classification_report_csv
#         report = classification_report(y_labels, y_pred)
#         classification_report_csv(report, os.path.join(results_path, modelfolder, "model_approachEnsemble_report.csv"),"Ensemble") 
#         pd.DataFrame(line).to_csv(os.path.join(results_path, modelfolder, "model_approachEnsemble_history.csv")) 

    # ---------------------------------------------------------------------------------
    print("Done. " + str(time) + " milliseconds")
    print("---------------------------------------------------------------------------------")
    return time

# def ClassifierEnsemble(data_path, results_path, ensembles, dataset='specific', save_results=True, modelfolder='model_ensemble'):
    
#     import os
#     import pandas as pd
#     import numpy as np
#     from datetime import datetime
#     import tensorflow
#     tensorflow.keras.backend.clear_session()
    
#     print('[Ensemble]: Loading base data description.')
#     if dataset == '':
#         TRAIN_FILE = os.path.join(data_path, 'train.csv')
#         TEST_FILE  = os.path.join(data_path, 'test.csv')
#     else:
#         TRAIN_FILE = os.path.join(data_path, dataset+'_train.csv')
#         TEST_FILE  = os.path.join(data_path, dataset+'_test.csv')
        
#     from automatize.ensemble_models.marc2 import loadTrajectories    
#     (keys, vocab_size,
#      labels,
#      num_classes,
#      max_length,
#      x_train, x_test,
#      y_train, y_test) = loadTrajectories(train_file=TRAIN_FILE,
#                                          test_file=TEST_FILE,
#                                          tid_col='tid',
#                                          label_col='label')

#     keys = list(pd.unique(labels))
# #     y_labels = [keys.index(x) for x in labels]
#     y_labels = [np.argmax(f) for f in y_test] 

#     from keras.wrappers.scikit_learn import KerasClassifier
    
#     time = datetime.now()
#     # create the sub models
#     models = dict()
#     estimators = []
#     print('[Ensemble]: '+', '.join(ensembles.keys()))
#     for method, folder in ensembles.items():
#         y_pred = []
#         model = []
#         if method == 'movelets':
#             from automatize.ensemble_models.movelets import model_movelets
#             model, x_test = model_movelets(folder)
#             from automatize.Methods import f1
#             models[method] = KerasClassifier(model)
#             model = model.predict(x_test)

#         if method is 'marc':
#             from automatize.ensemble_models.marc2 import model_marc
#             model, x_test = model_marc(folder, results_path, dataset)
#             models[method] = KerasClassifier(model)
#             model = model.predict(x_test)
            
#         if method is 'poi' or method is 'npoi' or method is 'wnpoi':
#             from automatize.ensemble_models.poifreq import model_poifreq
#             model, x_test = model_poifreq(folder)
#             models[method] = KerasClassifier(model)
#             model = model.predict(x_test)
            
#         if method is 'rf':
#             from automatize.ensemble_models.randomforrest import model_rf
#             model, x_test = model_rf(folder, dataset)
#             models[method] = model
#             model = model.predict(x_test)
            
#         if method is 'rfhp':
#             from automatize.ensemble_models.randomforresthp import model_rfhp
#             model, x_test = model_rfhp(folder, dataset)
#             models[method] = model
#             model = model.predict(x_test)
        
# #         print(method, 'ESTIMATORS:', model)
#         y_pred = [np.argmax(f) for f in model]  
# #         return model, y_pred
# #         print(method, 'PRED:', y_pred)
#         estimators.append(model) 
#         ensembles[method] = get_line(y_labels, y_pred)
#         print(method+': ', ensembles[method])
#         print("---------------------------------------------------------------------------------")
    
# #     print(estimators)
#     final_pred = estimators[0]
#     for i in range(1, len(estimators)):
#         final_pred = final_pred + estimators[i]
#     y_pred = [np.argmax(f) for f in final_pred]
    
#     print('[Ensemble]: Final results.')
#     print(ensembles)
#     line=get_line(y_labels, y_pred)
#     print('[Ensemble]:', line)
#     # ---------------------------------------------------------------------------------
#     if (save_results) :
#         if not os.path.exists(os.path.join(results_path, modelfolder)):
#             os.makedirs(os.path.join(results_path, modelfolder))
#         from sklearn.metrics import classification_report
#         from automatize.Methods import classification_report_csv
#         report = classification_report(y_labels, y_pred)
#         classification_report_csv(report, os.path.join(results_path, modelfolder, "model_approachEnsemble_report.csv"),"Ensemble") 
#         pd.DataFrame(line).to_csv(os.path.join(results_path, modelfolder, "model_approachEnsemble_history.csv")) 

#     # ----------------------------------------------------------------------------------
#     time = (datetime.now()-time).total_seconds() * 1000
#     # ---------------------------------------------------------------------------------
#     print("Done. " + str(time) + " milliseconds")
#     print("---------------------------------------------------------------------------------")
#     return time
# # ---------------------------------------------------------------------------------

# def ClassifierEnsemble(data_path, results_path, ensembles, dataset='specific', save_results=True, modelfolder='model_ensemble'):
    
#     import os
#     import pandas as pd
#     import numpy as np
#     from scipy import stats
#     from datetime import datetime
#     import tensorflow
#     tensorflow.keras.backend.clear_session()
    
#     if dataset == '':
#         TRAIN_FILE = os.path.join(data_path, 'train.csv')
#         TEST_FILE  = os.path.join(data_path, 'test.csv')
#     else:
#         TRAIN_FILE = os.path.join(data_path, dataset+'_train.csv')
#         TEST_FILE  = os.path.join(data_path, dataset+'_test.csv')

# #     X_train, y_train, X_test, y_test = loadData(list(ensembles.values())[0]) # temp
# #     print(y_train)
        
#     from automatize.ensemble_models.marc2 import loadTrajectories    
#     (keys, vocab_size,
#      labels,
#      num_classes,
#      max_length,
#      x_train, x_test,
#      y_train, y_test) = loadTrajectories(train_file=TRAIN_FILE,
#                                          test_file=TEST_FILE,
#                                          tid_col='tid',
#                                          label_col='label')

#     keys = list(pd.unique(labels))
#     y_labels = [keys.index(x) for x in labels]
# #     labels   = list(set(y_train))
# #     y_train  = [labels.index(x) for x in y_train]
# #     y_test   = [labels.index(x) for x in y_test]
    
# #     return X_train, y_train, X_test, y_test
    
#     time = datetime.now()
#     # create the sub models
#     estimators = []
#     print('[Ensemble]: '+', '.join(ensembles.keys()))
#     for method, folder in ensembles.items():
# #         print(method+': ', folder)
#         y_pred = []
#         model = []
#         if method == 'movelets':
# #             from automatize.analysis import loadData
#             from automatize.ensemble_models.movelets import model_movelets
# #             x_train_m, y_train_m, x_test_m, y_test_m = loadData(folder) # temp
# #             labels   = list(set(y_train))
# #             y_train_m  = [labels.index(x) for x in y_train_m]
# #             y_test_m   = [labels.index(x) for x in y_test_m]
#             model, x_test = model_movelets(folder)
#             model = model.predict(x_test)
# #             estimators.append((method, model))
# #             estimators.append(model)
# #             print(method+': ', get_line(y_test, model))

#         if method is 'marc':
#             from automatize.ensemble_models.marc2 import model_marc
#             model, x_test = model_marc(folder, results_path, dataset)
#             model = model.predict(x_test)
# #             estimators.append((method, model))
# #             estimators.append(model)
# #             print(method+': ', get_line(y_test, model))
            
#         if method is 'npoi':
#             from automatize.ensemble_models.poifreq import model_poifreq
#             model, x_test = model_poifreq(folder)
#             model = model.predict(x_test)
# #             estimators.append((method, model))
# #             estimators.append(model)
# #             print(method+': ', get_line(y_test, model))
            
#         if method is 'rf':
#             from automatize.ensemble_models.randomforrest import model_rf
#             model, x_test = model_rf(folder, dataset)
#             model = model.predict_proba(x_test)
# #             estimators.append((method, model))
# #             estimators.append(model)
# #             print(method+': ', get_line(y_test, model))
        
# #         print(method, model)
#         y_pred = [np.argmax(f) for f in model]
#         estimators.append(y_pred) 
# #         print('idx_pred', y_pred)
# #         print(y_labels, y_pred)
#         ensembles[method] = get_line(y_labels, y_pred)
#         print(method+': ', ensembles[method])
#         print("---------------------------------------------------------------------------------")
        
#     # create the ensemble model
# #     from sklearn import model_selection
#     from sklearn.ensemble import VotingClassifier
# #     ensemble = VotingClassifier(estimators)
# #     results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
# #     results = ensemble.predict(X_test)
#     mlds = [(key, model) for key, model in ensembles]
#     eclf = VotingClassifier(estimators=mlds)
    
# #     print(estimators)
#     final_pred = stats.mode(estimators).mode[0]
# #     print(final_pred)
#     y_pred = [np.argmax(f) for f in final_pred]
# #     for i in range(0,len(X_test)):
# #         final_pred = np.append(final_pred, mode([pred1[i], pred2[i], pred3[i]]))
    
#     print('[Ensemble] Final results:')
#     print(ensembles)
#     line=get_line(y_labels, y_pred)
#     print('[Ensemble]:', line)
#-------------------------------------------------------------------------------------------------------

# def ApproachEnsemble(dir_path, dir_path2, save_results=True, method2='poifreq', modelfolder='model'):
        
#     import os
#     import pandas as pd
#     import numpy as np
#     from datetime import datetime
#     import tensorflow
#     from sklearn.metrics import classification_report
#     from automatize.Methods import classification_report_csv
#     from automatize.analysis import loadData
#     from automatize.ensemble_models.movelets import model_movelets
        
#     X_train, y_train, X_test, y_test = loadData(dir_path)
        
#     labels   = list(set(y_train))
#     y_train  = [labels.index(x) for x in y_train]
#     y_test   = [labels.index(x) for x in y_test]
# #     y_train2 = [labels.index(x) for x in y_train2]
# #     y_test2  = [labels.index(x) for x in y_test2]
    
#     print("Building Ensemble models")
#     time = datetime.now()
    
#     tensorflow.keras.backend.clear_session()
#     pred1 = model_movelets(X_train, y_train, X_test, y_test).predict(X_test)

#     tensorflow.keras.backend.clear_session()
#     if method2 is 'marc':
#         from automatize.ensemble_models.marc import model_marc
#         pred2 = model_marc(dir_path2).predict(X_test)
#     if method2 is 'poifreq':
#         from automatize.ensemble_models.poifreq import model_poifreq
#         pred2 = model_poifreq(dir_path2).predict(X_test)
#     if method2 is 'rf':
#         from automatize.ensemble_models.randomforrest import model_rf
#         pred2 = model_rf(dir_path2).predict_proba(X_test)

#     y_pred1 = [np.argmax(f) for f in pred1]
#     y_pred2 = [np.argmax(f) for f in pred2]

#     final_pred = (pred1*0.5+pred2*0.5)
#     y_pred = [np.argmax(f) for f in final_pred]
    
#     print('Models results:')
#     print(get_line(y_test, y_pred1))
#     print(get_line(y_test, y_pred2))
    
#     print('Ensembled results:')
#     line=get_line(y_test, y_pred)
#     print(line)
    
#     # ---------------------------------------------------------------------------------
#     if (save_results) :
#         if not os.path.exists(os.path.join(dir_path, modelfolder)):
#             os.makedirs(os.path.join(dir_path, modelfolder))
#         report = classification_report(y_test, y_pred) #classifier.predict(X_test) )
#         classification_report_csv(report, os.path.join(dir_path, modelfolder, "model_approachEnsemble_report.csv"),"Ensemble") 
#         pd.DataFrame(line).to_csv(os.path.join(dir_path, modelfolder, "model_approachEnsemble_history.csv")) 
    
#     # ----------------------------------------------------------------------------------
#     time = (datetime.now()-time).total_seconds() * 1000
#     # ---------------------------------------------------------------------------------
#     print("Done. " + str(time) + " milliseconds")
#     print("---------------------------------------------------------------------------------")
#     return time
# # --------------------------------------------------------------------------------
# importer(['precision_score', 'recall_score', 'f1_score', 'accuracy_score'], locals())
importer(['metrics'], globals())

# Statistics:
def get_line(y_true, y_pred, y_test_pred=None):
    acc = accuracy(y_true, y_pred)
    f1  = f1_macro(y_true, y_pred)
    prec= precision_macro(y_true, y_pred)
    rec = recall_macro(y_true, y_pred)
    if y_test_pred is not None:
        accTop5 = calculateAccTop5(y_test_pred, y_pred, 5)
    else:
        accTop5 = 0
    line=[acc, f1, prec, rec, accTop5]
    return line

def precision_macro(y_true, y_pred):
#     from ..main import importer
#     importer(['precision_score'], locals())
#     from sklearn.metrics import precision_score
    return precision_score(y_true, y_pred, average='macro')
def recall_macro(y_true, y_pred):
#     from ..main import importer
#     importer(['recall_score'], locals())
#     from sklearn.metrics import recall_score
    return recall_score(y_true, y_pred, average='macro')
def f1_macro(y_true, y_pred):
#     from ..main import importer
#     importer(['f1_score'], locals())
#     from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average='macro')
def accuracy(y_true, y_pred):
#     from ..main import importer
#     importer(['accuracy_score'], locals())
#     from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred, normalize=True)
# --------------------------------------------------------------------------------
def calculateAccTop5(y_test_pred, y_true, K):
#     from ..main import importer
#     importer(['np'], locals())
#     import numpy as np
    
    K = K if len(y_true) > K else len(y_true)
    
#     y_test_pred = classifier.predict_proba(X_test)
    order=np.argsort(y_test_pred, axis=1)
#     n=classifier.classes_[order[:, -K:]]
    n=order[:, -K:]
    soma = 0;
    for i in range(0,len(y_true)) :
        if ( y_true[i] in n[i,:] ) :
            soma = soma + 1
    accTopK = soma / len(y_true)
    
    return accTopK