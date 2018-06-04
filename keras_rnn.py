from time import time
import os
import csv
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score, make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pdb
from tqdm import tqdm
from util import read_data
import sys
import pickle
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.pipeline import Pipeline
import tensorflow as tf


data_home = 'data'
base_data_filename = os.path.join(data_home, 'training_set_rel3.tsv')

vsmdata_home = 'vsmdata'

glove_home = os.path.join(vsmdata_home, 'glove.6B')

def glove2dict(src_filename):
    """GloVe Reader.

    Parameters
    ----------
    src_filename : str
        Full path to the GloVe file to be processed.

    Returns
    -------
    dict
        Mapping words to their GloVe vectors.

    """
    data = {}
    with open(src_filename,  encoding='utf8') as f:
        while True:
            try:
                line = next(f)
                line = line.strip().split()
                data[line[0]] = np.array(line[1: ], dtype=np.float)
            except StopIteration:
                break
            except UnicodeDecodeError:
                pass
    print('Read in GloVe Dict...')
    return data

glove_size = 300
glove_lookup = glove2dict(os.path.join(glove_home, 'glove.6B.'+str(glove_size)+'d.txt'))

def create_glove_representations(all_essays):
    to_return = {}
    for x in range(1, len(all_essays)+1):
        essays = all_essays[x]
        glove_essays = np.zeros((len(essays), glove_size))
        for i in tqdm(range(len(essays))):
            essay = essays[i][1]
            glove = [glove_lookup[w] for w in essay.split(' ') if w in glove_lookup]
            glove = [sum(i) / len(i) for i in zip(*glove)]
            if len(glove) != glove_size:
                glove_essays[i] = np.zeros(glove_size)
            else:
                glove_essays[i] = glove
        to_return[x] = glove_essays
    print('Created GloVe representations...')
    return to_return

def encode_data_for_keras(avg_scores):

    # Convert y
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(avg_scores)
    encoded_Y = encoder.transform(avg_scores)
    # convert integers to dummy variables (i.e. one hot encoded)

    dummy_y = np_utils.to_categorical(encoded_Y)
    return (dummy_y, len(dummy_y[0]))


def cohen_kappa(y_true, y_pred):
    y_t_clean = [y_true[i].index(1) for i in y_true]
    y_p_clean = [y_pred[j].index(1) for k in y_pred]
    return cohen_kappa_score(y_t_clean, y_p_clean)

def test_model(num_scores):
    model = Sequential()
    model.add(Dense(100, input_dim=glove_size, activation='relu'))
    model.add(Dense(rnn_model.num_scores, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', cohen_kappa])
    return model

def MLP(num_scores):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=glove_size))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_scores, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    return model



num_scores = 0

def main():
    seed = 7
    np.random.seed(seed)
    all_essays, all_avg_scores = read_data()
    if 'reload_data' in sys.argv or not os.path.isfile(os.path.join(data_home, 'glove_vectors.pickle')):
        print('Reloading data...')
        print('Loaded all data...')

        all_glove_data = create_glove_representations(all_essays)
        with open(os.path.join(data_home, 'glove_vectors.pickle'), 'wb') as handle:
            pickle.dump(all_glove_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Wrote data to pickle file.')
    else:

        with open(os.path.join(data_home, 'glove_vectors.pickle'), 'rb') as handle:
            all_glove_data = pickle.load(handle)
            print('Loaded vectors from Pickle file... ')

    for i in range(1, len(all_glove_data)+1):
        glove_data = all_glove_data[i]
        avg_scores = [round (x, 0) for x in all_avg_scores[i]]

        # Limit data just for testing

        #X_train, X_test, y_train, y_test = train_test_split(keras_data, keras_labels, train_size=0.9)
        # Split data into test and train
        print('Creating Estimator...')
        Y, keras_length = encode_data_for_keras(avg_scores)
        X_train, X_test, y_train, y_test = train_test_split(glove_data, Y, train_size=0.9)
        # c_y_train, keras_length = encode_data_for_keras(y_train)
        # c_y_test, test_len = encode_data_for_keras(y_test)
        param_matrix = {'activation': ['relu'], }


        rnn = MLP(keras_length)
        rnn.fit(X_train, y_train, epochs=50, batch_size=20)

        scores = rnn.evaluate(X_test, y_test, verbose=0)
        for i in range(len(rnn.metrics_names)):
            print("%s: %.2f%%" % (rnn.metrics_names[i], scores[i]*100))
        # estimator = KerasClassifier(build_fn=rnn_model, epochs=200, batch_size=20)


        # estimator.fit()


        # Compile model
        # print('Compiling Model...')
        # kfold = KFold(n_splits=1, shuffle=True, random_state=seed)
        # results = cross_val_score(estimator, glove_data, keras_labels, cv=kfold)
        # print("Baseline: %.2f%% (%.2f%%)" % (results.mean(), results.std()))
        #
        # with open(os.path.join(data_home, 'results.pickle'), 'wb') as handle:
        #     pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        break


if __name__ == "__main__":
    main()
