import sys

if len(sys.argv) < 6:
    print('Please run as:')
    print('\tpython bituler.py', 'TRAIN_FILE', 'TEST_FILE', 'EMBEDDINGS_FILE',
          'RESULTS_FILE', 'DATASET_NAME')
    exit()

METHOD = 'TULVAE'
TRAIN_FILE = sys.argv[1]
TEST_FILE = sys.argv[2]
EMBEDDINGS_FILE = sys.argv[3]
METRICS_FILE = sys.argv[4]
DATASET = sys.argv[5]

print('====================================', 'PARAMS',
      '====================================')
print('TRAIN_FILE =', TRAIN_FILE)
print('TEST_FILE =', TEST_FILE)
print('EMBEDDINGS_FILE =', EMBEDDINGS_FILE)
print('METRICS_FILE =', METRICS_FILE)
print('DATASET =', DATASET, '\n')

from metrics import MetricsLogger
metrics = MetricsLogger().load(METRICS_FILE)


###############################################################################
#   LOAD DATA
###############################################################################
import pandas as pd
from sklearn.preprocessing import LabelEncoder

tid_col = 'tid'
label_col = 'label'
poi_col = 'poi'

df_train = pd.read_csv(TRAIN_FILE)
df_test = pd.read_csv(TEST_FILE)

poi_encoder = LabelEncoder().fit(df_train['poi'].copy().append(df_test['poi']))
#label_encoder = LabelEncoder().fit(df_train[label_col])

df_train[poi_col] = poi_encoder.transform(df_train[poi_col])
df_test[poi_col] = poi_encoder.transform(df_test[poi_col])

#df_train[label_col] = label_encoder.transform(df_train[label_col])
#df_test[label_col] = label_encoder.transform(df_test[label_col])

max_traj_length = df_train.copy().append(df_test)[[tid_col, poi_col]]\
                          .groupby([tid_col]).agg(['count']).max().values[0]
num_classes = len(df_train[label_col].unique())


###############################################################################
#   BUILD POI EMBEDDINGS
###############################################################################
from gensim.models import Word2Vec
import numpy as np
import os

train_traj = []
train_label = []
test_traj = []
test_label = []
EMBEDDING_SIZE = 250
WINDOW_SIZE = 5

print('==================================', 'EMBEDDINGS',
      '==================================')
print('EMBEDDING_SIZE =', EMBEDDING_SIZE)
print('WINDOW_SIZE =', WINDOW_SIZE, '\n')

for tid in df_train[tid_col].unique():
    traj = df_train.loc[df_train[tid_col] == tid, :]
    train_traj.append(list(np.asarray(traj[poi_col].values, dtype=str)))
    train_label.append(traj[label_col].values[0])

for tid in df_test[tid_col].unique():
    traj = df_test.loc[df_test[tid_col] == tid, :]
    test_traj.append(list(np.asarray(traj[poi_col].values, dtype=str)))
    test_label.append(traj[label_col].values[0])

w2v_data = np.concatenate([train_traj, test_traj])

if os.path.isfile(EMBEDDINGS_FILE):
    print("Embeddings loaded from file '" + EMBEDDINGS_FILE + "'")
    w2v = Word2Vec.load(EMBEDDINGS_FILE)
else:
    print("Training embeddings")
    w2v = Word2Vec(w2v_data,
                   size=EMBEDDING_SIZE,
                   min_count=0,
                   window=WINDOW_SIZE)
    w2v.save(EMBEDDINGS_FILE)
    print("Embeddings saved to file '" + EMBEDDINGS_FILE + "'")

vocab_size = len(w2v.wv.vocab)
embedding_mx = np.zeros(shape=(vocab_size, EMBEDDING_SIZE))

for idx in range(0, vocab_size):
    poi = w2v.wv.index2word[idx]
    embedding = w2v.wv[poi]
    embedding_mx[int(poi)] = embedding

print('Embedding matrix shape:', embedding_mx.shape)


###############################################################################
#   CLASSIFICATION DATA
###############################################################################
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from keras.preprocessing.sequence import pad_sequences

one_hot_y = OneHotEncoder().fit(pd.DataFrame({label_col: train_label}))

x_train = pad_sequences(train_traj, max_traj_length, padding='pre')
x_test = pad_sequences(test_traj, max_traj_length, padding='pre')
y_train = one_hot_y.transform(pd.DataFrame({label_col: train_label})).toarray()
y_test = one_hot_y.transform(pd.DataFrame({label_col: test_label})).toarray()


###############################################################################
#   TULVAE
###############################################################################
# Code based on https://github.com/AI-World/IJCAI-TULVAE
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Concatenate, Embedding, RepeatVector
from keras.layers import Bidirectional, LSTM
from keras.layers.core import Dense, Lambda
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import objectives
from metrics import compute_acc_acc5_f1_prec_rec


hidden_cells = 512
cls_hidden_units = 300
latent_dim = 100
output_size = num_classes
keep_prob = 0.5

LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 30
BASELINE_METRIC = 'classifier_acc'
BASELINE_VALUE = 0.5
BATCH_SIZE = 64
EPOCHS = 1000

z_mean = None
z_log_sigma = None


print('====================================', 'TULVAE',
      '====================================')
print('hidden_cells =', hidden_cells)
print('cls_hidden_units =', cls_hidden_units)
print('output_size =', output_size)
print('keep_prob =', keep_prob)
print('LEARNING_RATE =', LEARNING_RATE)
print('EARLY_STOPPING_PATIENCE =', EARLY_STOPPING_PATIENCE)
print('BASELINE_METRIC =', BASELINE_METRIC)
print('BASELINE_VALUE =', BASELINE_VALUE)
print('BATCH_SIZE =', BATCH_SIZE)
print('EPOCHS =', EPOCHS, '\n')


class EpochLogger(EarlyStopping):

    def __init__(self, metric='val_acc', baseline=0):
        super(EpochLogger, self).__init__(monitor='val_classifier_acc',
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
        b_size = 128
        pred_y_train = np.zeros(y_train.shape)

        for i in range(0, len(y_train) // b_size):
            beg = i * b_size
            end = min(len(y_train), beg + b_size)
            pred_y_train[beg:end] = model.predict([x_train[beg:end],
                                                   y_train[beg:end]],
                                                  batch_size=b_size)[0]

        (train_acc,
         train_acc5,
         train_f1_macro,
         train_prec_macro,
         train_rec_macro) = compute_acc_acc5_f1_prec_rec(y_train,
                                                         pred_y_train,
                                                         print_metrics=True,
                                                         print_pfx='TRAIN')

        pred_y_test = np.zeros(y_test.shape)

        for i in range(0, len(y_test) // b_size):
            beg = i * b_size
            end = min(len(y_test), beg + b_size)
            pred_y_test[beg:end] = model.predict([x_test[beg:end],
                                                  y_test[beg:end]],
                                                 batch_size=b_size)[0]

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


def encoder(x, x_label):
    global z_mean, z_log_sigma
    e = Embedding(vocab_size,
                  EMBEDDING_SIZE,
                  weights=[embedding_mx],
                  input_length=max_traj_length,
                  trainable=False)(x)
    h = LSTM(hidden_cells, dropout=keep_prob)(e)
    c = Concatenate(axis=1)([h, x_label])

    z_mean = Dense(latent_dim)(c)
    z_log_sigma = Dense(latent_dim)(c)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev=1.)
        return z_mean + K.exp(z_log_sigma) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
    return Dense(hidden_cells, activation='softplus')(z)


def decoder(z, name):
    z = RepeatVector(max_traj_length)(z)
    decoder_h = LSTM(hidden_cells,
                     return_sequences=True)(z)
    decoder_mean = LSTM(EMBEDDING_SIZE, return_sequences=True)
    decoded = decoder_mean(decoder_h)
    return Dense(vocab_size,
                 name=name)(decoded)


def classifier(x):
    e = Embedding(vocab_size,
                  EMBEDDING_SIZE,
                  weights=[embedding_mx],
                  input_length=max_traj_length,
                  trainable=False)(x)
    h = Bidirectional(LSTM(cls_hidden_units, dropout=keep_prob))(e)
    return Dense(output_size,
                 activation='softmax',
                 name='classifier')(h)


def vae_loss(x, x_decoded_mean):
    expanded_x = K.one_hot(K.cast(K.squeeze(x, axis=1), 'int32'), vocab_size)
    xent_loss = objectives.mse(expanded_x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
    loss = xent_loss + kl_loss
    return loss

x = Input(shape=(max_traj_length,))
x_label = Input(shape=(output_size,))

# Output classifier
c = classifier(x)

# Output labeled input
z = encoder(x, x_label)
t = decoder(z, 'decoder_l')

# Output unlabeled input
oh = Lambda(lambda x: K.one_hot(K.argmax(x), output_size))(c)
z2 = encoder(x, oh)
t2 = decoder(z2, 'decoder_u')

model = Model(inputs=[x, x_label], outputs=[c, t, t2])

opt = Adam(lr=LEARNING_RATE)
model.compile(optimizer=opt,
              loss=['categorical_crossentropy', vae_loss, vae_loss],
              loss_weights=[1.0, 1.0, 0.5],
              metrics={'classifier': ['acc', 'top_k_categorical_accuracy']})

model.fit(x=[x_train, y_train],
          y=[y_train, np.expand_dims(x_train, 1), np.expand_dims(x_train, 1)],
          validation_data=([x_test, y_test],
                           [y_test, np.expand_dims(x_test, 1), np.expand_dims(x_test, 1)]),
          batch_size=BATCH_SIZE,
          shuffle=True,
          epochs=EPOCHS,
          verbose=0,
          callbacks=[EpochLogger(metric=BASELINE_METRIC,
                                 baseline=BASELINE_VALUE)])  # To avoid too early stopping
