# Under Construction:
def model_marc(dir_path, dataset=''):
    from keras.models import Model
    from keras.layers import Dense, LSTM, GRU, Dropout
    from keras.initializers import he_uniform
    from keras.regularizers import l1
    from keras.optimizers import Adam
    from keras.layers import Input, Add, Average, Concatenate, Embedding
    
    EMBEDDER_SIZE = 100
    MERGE_TYPE    = 'concatenate' # 'add', 'average'
    RNN_CELL      = 'lstm' # 'gru'
    
    CLASS_DROPOUT = 0.5
    CLASS_HIDDEN_UNITS = 100
    CLASS_LRATE = 0.001
    CLASS_BATCH_SIZE = 64
    CLASS_EPOCHS = 1000
    BASELINE_METRIC = 'acc'
    BASELINE_VALUE = 0.5
    
    keys, vocab_size, num_classes, max_length, x_train, y_train, x_test, y_test = loadTrajectories(dir_path, dataset)
    
    print("[MARC:] Building classifier")
    from keras.callbacks import EarlyStopping
    class EpochLogger(EarlyStopping):

        def __init__(self, metric='val_acc', baseline=0):
            
            EARLY_STOPPING_PATIENCE = 30
            
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
            pred_y_train = np.array(self.model.predict(cls_x_train))
            (train_acc,
             train_acc5,
             train_f1_macro,
             train_prec_macro,
             train_rec_macro) = compute_acc_acc5_f1_prec_rec(cls_y_train,
                                                             pred_y_train,
                                                             print_metrics=True,
                                                             print_pfx='TRAIN')

            pred_y_test = np.array(self.model.predict(cls_x_test))
            (test_acc,
             test_acc5,
             test_f1_macro,
             test_prec_macro,
             test_rec_macro) = compute_acc_acc5_f1_prec_rec(cls_y_test,
                                                            pred_y_test,
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
    
    inputs = []
    embeddings = []

    for idx, key in enumerate(keys):
        if key == 'lat_lon':
            i = Input(shape=(max_length, vocab_size[key]),
                      name='input_' + key)
            e = Dense(units=EMBEDDER_SIZE,
                      kernel_initializer=he_uniform(seed=1),
                      name='emb_' + key)(i)
        else:
            i = Input(shape=(max_length,),
                      name='input_' + key)
            e = Embedding(vocab_size[key],
                          EMBEDDER_SIZE,
                          input_length=max_length,
                          name='emb_' + key)(i)
        inputs.append(i)
        embeddings.append(e)

    if len(embeddings) == 1:
        hidden_input = embeddings[0]
    elif MERGE_TYPE == 'add':
        hidden_input = Add()(embeddings)
    elif MERGE_TYPE == 'average':
        hidden_input = Average()(embeddings)
    else:
        hidden_input = Concatenate(axis=2)(embeddings)

    hidden_dropout = Dropout(CLASS_DROPOUT)(hidden_input)

    if RNN_CELL == 'lstm':
        rnn_cell = LSTM(units=CLASS_HIDDEN_UNITS,
                        recurrent_regularizer=l1(0.02))(hidden_dropout)
    else:
        rnn_cell = GRU(units=CLASS_HIDDEN_UNITS,
                       recurrent_regularizer=l1(0.02))(hidden_dropout)

    rnn_dropout = Dropout(CLASS_DROPOUT)(rnn_cell)

    softmax = Dense(units=num_classes,
                    kernel_initializer=he_uniform(),
                    activation='softmax')(rnn_dropout)

    classifier = Model(inputs=inputs, outputs=softmax)
    opt = Adam(lr=CLASS_LRATE)

    classifier.compile(optimizer=opt,
                       loss='categorical_crossentropy',
                       metrics=['acc'])

    classifier.fit(x=x_train,
                   y=y_train,
                   validation_data=(x_test, y_test),
                   batch_size=CLASS_BATCH_SIZE,
                   shuffle=True,
                   epochs=CLASS_EPOCHS,
                   verbose=0,
                   callbacks=[EpochLogger(metric=BASELINE_METRIC,
                                      baseline=BASELINE_VALUE)])
    
    print("[MARC:] OK")
#     return classifier.predict(x_test)
    return classifier, x_test
# --------------------------------------------------------------------------------

# Load data:
def loadTrajectories(dir_path, dataset='', tid_col='tid',
                     label_col='label', geo_precision=8, drop=[]):
    import os
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from marc.core.utils.geohash import bin_geohash
    
    print("Loading data from file(s) " + dir_path + "... ")
    if dataset is '':
        df_train = pd.read_csv(os.path.join(dir_path, 'train.csv'))
        df_test = pd.read_csv(os.path.join(dir_path, 'test.csv'))
    else:
        df_train = pd.read_csv(os.path.join(dir_path, dataset+'_train.csv'))
        df_test = pd.read_csv(os.path.join(dir_path, dataset+'_test.csv'))
    
#     df_train = pd.read_csv(os.path.join(dir_path, 'train.csv'))
#     df_test = pd.read_csv(os.path.join(dir_path, 'test.csv'))
    df = df_train.copy().append(df_test)
    tids_train = df_train[tid_col].unique()

    keys = list(df.keys())
    vocab_size = {}
    keys.remove(tid_col)

    for col in drop:
        if col in keys:
            keys.remove(col)
            print("Column '" + col + "' dropped " +
                       "from input file!")
        else:
            print("Column '" + col + "' cannot be " +
                       "dropped because it was not found!")

    num_classes = len(set(df[label_col]))
    count_attr = 0
    lat_lon = False

    if 'lat' in keys and 'lon' in keys:
        keys.remove('lat')
        keys.remove('lon')
        lat_lon = True
        count_attr += geo_precision * 5
        print("Attribute Lat/Lon: " +
                   str(geo_precision * 5) + "-bits value")

    for attr in keys:
        if attr != label_col:
            df[attr] = LabelEncoder().fit_transform(df[attr])
            vocab_size[attr] = max(df[attr]) + 1

            values = len(set(df[attr]))
            count_attr += values
            print("Attribute '" + attr + "': " +
                       str(values) + " unique values")

    print("Total of attribute/value pairs: " +
               str(count_attr))
    keys.remove(label_col)

    x = [[] for key in keys]
    y = []
    idx_train = []
    idx_test = []
    max_length = 0
    trajs = len(set(df[tid_col]))

    if lat_lon:
        x.append([])

    for idx, tid in enumerate(set(df[tid_col])):
        traj = df.loc[df[tid_col].isin([tid])]
        features = np.transpose(traj.loc[:, keys].values)

        for i in range(0, len(features)):
            x[i].append(features[i])

        if lat_lon:
            loc_list = []
            for i in range(0, len(traj)):
                lat = traj['lat'].values[i]
                lon = traj['lon'].values[i]
                loc_list.append(bin_geohash(lat, lon, geo_precision))
            x[-1].append(loc_list)

        label = traj[label_col].iloc[0]
        y.append(label)

        if tid in tids_train:
            idx_train.append(idx)
        else:
            idx_test.append(idx)

        if traj.shape[0] > max_length:
            max_length = traj.shape[0]

    if lat_lon:
        keys.append('lat_lon')
        vocab_size['lat_lon'] = geo_precision * 5

    one_hot_y = OneHotEncoder().fit(df.loc[:, [label_col]])

    x = [np.asarray(f) for f in x]
    y = one_hot_y.transform(pd.DataFrame(y)).toarray()
    print("Loading data from files ... DONE!")
    
    x_train = np.asarray([f[idx_train] for f in x])
    y_train = y[idx_train]
    x_test = np.asarray([f[idx_test] for f in x])
    y_test = y[idx_test]

    print('Trajectories:  ' + str(trajs))
    print('Labels:        ' + str(len(keys)))
    print('Train size:    ' + str(len(x_train[0]) / trajs))
    print('Test size:     ' + str(len(x_test[0]) / trajs))
    print('x_train shape: ' + str(x_train.shape))
    print('y_train shape: ' + str(y_train.shape))
    print('x_test shape:  ' + str(x_test.shape))
    print('y_test shape:  ' + str(y_test.shape))
    
    from keras.preprocessing.sequence import pad_sequences
    x_train = np.asarray([pad_sequences(f, max_length, padding='pre') for f in x_train])
    x_test  = np.asarray([pad_sequences(f, max_length, padding='pre') for f in x_test])

    return (keys, vocab_size, num_classes, max_length,
            x_train, y_train,
            x_test, y_test)
