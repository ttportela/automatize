# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
import random as rn
from metrics import MetricsLogger
from metrics import compute_acc_acc5_f1_prec_rec

import os, sys
script_dir = os.path.dirname( __file__ )
main_dir = os.path.abspath(os.path.join( script_dir, '..' , '..'))
sys.path.append( main_dir )
# sys.path.insert(0, os.path.abspath(os.path.join('.')))
from main import importer #, display
importer(['S', 'datetime', 'sys', 'encoding', 'POIS'], globals())
###############################################################################

if len(sys.argv) < 4:
    print('Please run as:')
    print('\tPOIS-Classifier.py', 'METHOD', 'POIS_PREFIX', 'PATH TO RESULTS_DIR', 'RESULTS_FOLDER')
    print('Example:')
    print('\tPOIS-Classifier.py', 'npoi', 'poi_1_2_3_specific', '"./results"', '"./results"')
    exit()

METHOD = sys.argv[1]
DATASET = sys.argv[2]
path_name = sys.argv[3]
RESULTS_FILE = os.path.join(path_name, sys.argv[4], 'poifreq_results')
METRICS_FILE = os.path.join(path_name, 'metrics', METHOD+'_'+DATASET+'.csv')


if not os.path.exists(os.path.join(path_name, 'metrics')):
    os.makedirs(os.path.join(path_name, 'metrics'))
    
if os.path.exists(RESULTS_FILE):
    print("POI-S ", RESULTS_FILE, "Done.")
    quit()
          
print("POI-S Starting: building neural network for", METHOD, DATASET)
os.makedirs(os.path.join(path_name, sys.argv[4]))
time = datetime.now()

metrics = MetricsLogger().load(METRICS_FILE)

np.random.seed(2019)

# from tensorflow import set_random_seed
# set_random_seed(2017)

rn.seed(1254)

###############################################################################
#   LOAD DATA
###############################################################################

x_train = pd.read_csv(os.path.join(path_name, METHOD+'_'+DATASET+'-x_train.csv'), header=None)
# x_train = x_train[x_train.columns[:-1]]
y_train = pd.read_csv(os.path.join(path_name, METHOD+'_'+DATASET+'-y_train.csv'))

x_test = pd.read_csv(os.path.join(path_name, METHOD+'_'+DATASET+'-x_test.csv'), header=None)
# x_test = x_test[x_test.columns[:-1]]
y_test = pd.read_csv(os.path.join(path_name, METHOD+'_'+DATASET+'-y_test.csv'))

num_features = len(list(x_train))
num_classes = len(y_train['label'].unique())


one_hot_y = OneHotEncoder()
one_hot_y.fit(y_train.loc[:, ['label']])

y_train = one_hot_y.transform(y_train.loc[:, ['label']]).toarray()
y_test = one_hot_y.transform(y_test.loc[:, ['label']]).toarray()

x_train = preprocessing.scale(x_train)
x_test = preprocessing.scale(x_test)

###############################################################################
#   MOVELETS CLASSIFIER
###############################################################################

keep_prob = 0.5

HIDDEN_UNITS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 30
BASELINE_METRIC = 'acc'
BASELINE_VALUE = 0.5
BATCH_SIZE = 64
EPOCHS = 250

print('===================================', METHOD,
      '===================================')

print('keep_prob =', keep_prob)
print('HIDDEN_UNITS =', HIDDEN_UNITS)
print('LEARNING_RATE =', LEARNING_RATE)
print('EARLY_STOPPING_PATIENCE =', EARLY_STOPPING_PATIENCE)
print('BASELINE_METRIC =', BASELINE_METRIC)
print('BASELINE_VALUE =', BASELINE_VALUE)
print('BATCH_SIZE =', BATCH_SIZE)
print('EPOCHS =', EPOCHS, '\n')


class EpochLogger(EarlyStopping):

    def __init__(self, metric='val_acc', baseline=0):
        super(EpochLogger, self).__init__(monitor='val_acc',
                                          mode='max',
                                          patience=EARLY_STOPPING_PATIENCE)
        self._metric = metric
        self._baseline = baseline
        self._baseline_met = False

    def on_epoch_begin(self, epoch, logs={}):
        print("===== Training Epoch %d =====" % (epoch + 1))

        if self._baseline_met:
            super(EpochLogger, self).on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs={}):
        pred_y_train = np.array(self.model.predict(x_train))
        (train_acc,
         train_acc5,
         train_f1_macro,
         train_prec_macro,
         train_rec_macro) = compute_acc_acc5_f1_prec_rec(y_train,
                                                         pred_y_train,
                                                         print_metrics=True,
                                                         print_pfx='TRAIN')

        pred_y_test = np.array(self.model.predict(x_test))
        (test_acc,
         test_acc5,
         test_f1_macro,
         test_prec_macro,
         test_rec_macro) = compute_acc_acc5_f1_prec_rec(y_test, pred_y_test,
                                                        print_metrics=True,
                                                        print_pfx='TEST')
        metrics.log(METHOD, int(epoch + 1), DATASET,
                    logs['loss'], train_acc, train_acc5,
                    train_f1_macro, train_prec_macro, train_rec_macro,
                    logs['val_loss'], test_acc, test_acc5,
                    test_f1_macro, test_prec_macro, test_rec_macro)
        metrics.save(METRICS_FILE)

        if self._baseline_met:
            super(EpochLogger, self).on_epoch_end(epoch, logs)

        if not self._baseline_met \
           and logs[self._metric] >= self._baseline:
            self._baseline_met = True

    def on_train_begin(self, logs=None):
        super(EpochLogger, self).on_train_begin(logs)

    def on_train_end(self, logs=None):
        if self._baseline_met:
            super(EpochLogger, self).on_train_end(logs)

model = Sequential()
hist = History()
model.add(Dense(units=HIDDEN_UNITS,
                input_dim=(num_features),
                kernel_initializer='uniform',
                kernel_regularizer=l2(0.02)))
model.add(Dropout(keep_prob))
model.add(Dense(units=num_classes,
                kernel_initializer='uniform',
                activation='softmax'))

# opt = Adam(lr=LEARNING_RATE)
opt = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['acc', 'top_k_categorical_accuracy'])

model.fit(x=x_train,
          y=y_train,
          validation_data=(x_test, y_test),
          batch_size=BATCH_SIZE,
          shuffle=True,
          epochs=EPOCHS,
          verbose=0,
          callbacks=[EpochLogger(metric=BASELINE_METRIC,
                                 baseline=BASELINE_VALUE), hist])
print(METRICS_FILE)
time_ext = (datetime.now()-time).total_seconds() * 1000
df = pd.read_csv(METRICS_FILE)
f = open(RESULTS_FILE+'.txt', 'a+')
print('------------------------------------------------------------------------------------------------', file=f)
print(f"Method: {METHOD} | Dataset: {DATASET}", file=f)
print(f"Acc: {np.array(df['test_acc'])[-EARLY_STOPPING_PATIENCE]} \
      | Acc_top_5: {np.array(df['test_acc_top5'])[-EARLY_STOPPING_PATIENCE]} \
      | F1_Macro: {np.array(df['test_f1_macro'])[-EARLY_STOPPING_PATIENCE]}",
      file=f)
print(f"Classification Time: {time_ext} milliseconds", file=f)
print('------------------------------------------------------------------------------------------------', file=f)
f.close()

# print(f"Processing time: {time_ext} milliseconds. Done.")
# print('------------------------------------------------------------------------------------------------')


