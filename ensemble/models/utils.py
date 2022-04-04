# -*- coding: utf-8 -*-
'''
Automatize: Multi-Aspect Trajectory Data Mining Tool Library
The present application offers a tool, called AutoMATize, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. The AutoMATize integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
License GPL v.3 or superior

@author: Tarlis Portela
@author: Lucas May Petry (adapted)
'''

###############################################################################
#   GOHASH
###############################################################################
from ...main import importer
importer(['np','gh'], globals())

base32 = ['0', '1', '2', '3', '4', '5', '6', '7',
          '8', '9', 'b', 'c', 'd', 'e', 'f', 'g',
          'h', 'j', 'k', 'm', 'n', 'p', 'q', 'r',
          's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
binary = [np.asarray(list('{0:05b}'.format(x, 'b')), dtype=int)
          for x in range(0, len(base32))]
base32toBin = dict(zip(base32, binary))


# Deprecated - for compatibility purposes
class LatLonHash:

    def __init__(self, lat, lon):
        self._lat = lat
        self._lon = lon

    def to_hash(self, precision=15):
        return gh.encode(self._lat, self._lon, precision)

    def to_binary(self, precision=15):
        hashed = self.to_hash(precision)
        return np.concatenate([base32toBin[x] for x in hashed])


def geohash(lat, lon, precision=15):
    return gh.encode(lat, lon, precision)


def bin_geohash(lat, lon, precision=15):
    hashed = geohash(lat, lon, precision)
    return np.concatenate([base32toBin[x] for x in hashed])


###############################################################################
#   Logger
###############################################################################
# from ..main import importer
importer(['sys','datetime'], globals())
# import sys
# from datetime import datetime
class Logger(object):

    LOG_LINE = None
    INFO        = '[    INFO    ]'
    WARNING     = '[  WARNING   ]'
    ERROR       = '[   ERROR    ]'
    CONFIG      = '[   CONFIG   ]'
    RUNNING     = '[  RUNNING   ]'
    QUESTION    = '[  QUESTION  ]'

    def log(self, type, message):
        if Logger.LOG_LINE:
            sys.stdout.write("\n")
            sys.stdout.flush()
            Logger.LOG_LINE = None

        sys.stdout.write(str(type) + " " + self.cur_date_time() + " :: " + message + "\n")
        sys.stdout.flush()

    def log_dyn(self, type, message):
        line = str(type) + " " + self.cur_date_time() + " :: " + message
        sys.stdout.write("\r\x1b[K" + line.__str__())
        sys.stdout.flush()
        Logger.LOG_LINE = line

    def get_answer(self, message):
        return input(Logger.QUESTION + " " + self.cur_date_time() + " :: " + message)

    def cur_date_time(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# EARLY_STOPPING_PATIENCE = 30
    
# from keras.callbacks import EarlyStopping
# class EpochLogger(EarlyStopping):

#     def __init__(self, metric='val_acc', baseline=0):
#         super(EpochLogger, self).__init__(monitor='val_acc',
#                                           mode='max',
#                                           patience=EARLY_STOPPING_PATIENCE)
#         self._metric = metric
#         self._baseline = baseline
#         self._baseline_met = False

#     def on_epoch_begin(self, epoch, logs={}):
#         print("===== Training Epoch %d =====" % (epoch + 1))

#         if self._baseline_met:
#             super(EpochLogger, self).on_epoch_begin(epoch, logs)

#     def on_epoch_end(self, epoch, logs={}):
#         pred_y_train = np.array(self.model.predict(cls_x_train))
#         (train_acc,
#          train_acc5,
#          train_f1_macro,
#          train_prec_macro,
#          train_rec_macro) = compute_acc_acc5_f1_prec_rec(cls_y_train,
#                                                          pred_y_train,
#                                                          print_metrics=True,
#                                                          print_pfx='TRAIN')

#         pred_y_test = np.array(self.model.predict(cls_x_test))
#         (test_acc,
#          test_acc5,
#          test_f1_macro,
#          test_prec_macro,
#          test_rec_macro) = compute_acc_acc5_f1_prec_rec(cls_y_test,
#                                                         pred_y_test,
#                                                         print_metrics=True,
#                                                         print_pfx='TEST')
#         metrics.log(METHOD, int(epoch + 1), dataset,
#                     logs['loss'], train_acc, train_acc5,
#                     train_f1_macro, train_prec_macro, train_rec_macro,
#                     logs['val_loss'], test_acc, test_acc5,
#                     test_f1_macro, test_prec_macro, test_rec_macro)
#         metrics.save(METRICS_FILE)

#         if self._baseline_met:
#             super(EpochLogger, self).on_epoch_end(epoch, logs)

#         if not self._baseline_met \
#            and logs[self._metric] >= self._baseline:
#             self._baseline_met = True

#     def on_train_begin(self, logs=None):
#         super(EpochLogger, self).on_train_begin(logs)

#     def on_train_end(self, logs=None):
#         if self._baseline_met:
#             super(EpochLogger, self).on_train_end(logs)


###############################################################################
#   Metrics
###############################################################################
# from ..main import importer
importer(['S', 'metrics'], globals())
# from sklearn.metrics import f1_score
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# import numpy as np
# import pandas as pd
# import os
# from datetime import datetime


def _process_pred(y_pred):
    argmax = np.argmax(y_pred, axis=1)
    y_pred = np.zeros(y_pred.shape)

    for row, col in enumerate(argmax):
        y_pred[row][col] = 1

    return y_pred


def precision_macro(y_true, y_pred):
    proc_y_pred = _process_pred(y_pred)
    return precision_score(y_true, proc_y_pred, average='macro')


def recall_macro(y_true, y_pred):
    proc_y_pred = _process_pred(y_pred)
    return recall_score(y_true, proc_y_pred, average='macro')


def f1_macro(y_true, y_pred):
    proc_y_pred = _process_pred(y_pred)
    return f1_score(y_true, proc_y_pred, average='macro')


def accuracy(y_true, y_pred):
    proc_y_pred = _process_pred(y_pred)
    return accuracy_score(y_true, proc_y_pred, normalize=True)


def accuracy_top_k(y_true, y_pred, K=5):
    order = np.argsort(y_pred, axis=1)
    correct = 0

    for i, sample in enumerate(np.argmax(y_true, axis=1)):
        if sample in order[i, -K:]:
            correct += 1

    return correct / len(y_true)


def compute_acc_acc5_f1_prec_rec(y_true, y_pred, print_metrics=True,
                                 print_pfx=''):
    acc = accuracy(y_true, y_pred)
    acc_top5 = accuracy_top_k(y_true, y_pred, K=5)
    _f1_macro = f1_macro(y_true, y_pred)
    _prec_macro = precision_macro(y_true, y_pred)
    _rec_macro = recall_macro(y_true, y_pred)

    if print_metrics:
        pfx = '' if print_pfx == '' else print_pfx + '\t\t'
        print(pfx + 'acc: %.6f\tacc_top5: %.6f\tf1_macro: %.6f\tprec_macro: %.6f\trec_macro: %.6f'
              % (acc, acc_top5, _f1_macro, _prec_macro, _rec_macro))

    return acc, acc_top5, _f1_macro, _prec_macro, _rec_macro


class MetricsLogger:

    def __init__(self):
        self._df = pd.DataFrame({'method': [],
                                 'epoch': [],
                                 'dataset': [],
                                 'timestamp': [],
                                 'train_loss': [],
                                 'train_acc': [],
                                 'train_acc_top5': [],
                                 'train_f1_macro': [],
                                 'train_prec_macro': [],
                                 'train_rec_macro': [],
                                 'train_acc_up': [],
                                 'test_loss': [],
                                 'test_acc': [],
                                 'test_acc_top5': [],
                                 'test_f1_macro': [],
                                 'test_prec_macro': [],
                                 'test_rec_macro': [],
                                 'test_acc_up': []})

    def log(self, method, epoch, dataset, train_loss, train_acc,
            train_acc_top5, train_f1_macro, train_prec_macro, train_rec_macro,
            test_loss, test_acc, test_acc_top5, test_f1_macro,
            test_prec_macro, test_rec_macro):
        timestamp = datetime.now()

        if len(self._df) > 0:
            train_max_acc = self._df['train_acc'].max()
            test_max_acc = self._df['test_acc'].max()
        else:
            train_max_acc = 0
            test_max_acc = 0

        self._df = self._df.append({'method': method,
                                    'epoch': epoch,
                                    'dataset': dataset,
                                    'timestamp': timestamp,
                                    'train_loss': train_loss,
                                    'train_acc': train_acc,
                                    'train_acc_top5': train_acc_top5,
                                    'train_f1_macro': train_f1_macro,
                                    'train_prec_macro': train_prec_macro,
                                    'train_rec_macro': train_rec_macro,
                                    'train_acc_up': 1 if train_acc > train_max_acc else 0,
                                    'test_loss': test_loss,
                                    'test_acc': test_acc,
                                    'test_acc_top5': test_acc_top5,
                                    'test_f1_macro': test_f1_macro,
                                    'test_prec_macro': test_prec_macro,
                                    'test_rec_macro': test_rec_macro,
                                    'test_acc_up': 1 if test_acc > test_max_acc else 0},
                                   ignore_index=True)

    def save(self, file):
        self._df.to_csv(file, index=False)

    def load(self, file):
        if os.path.isfile(file):
            self._df = pd.read_csv(file)
        else:
            print("WARNING: File '" + file + "' not found!")

        return self
