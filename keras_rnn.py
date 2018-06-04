from time import time
import os
import csv
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pdb
from tqdm import tqdm
from util import read_data
import sys
import pickle
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
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

setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)

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

glove_size = 50
glove_lookup = glove2dict(os.path.join(glove_home, 'glove.6B.'+str(glove_size)+'d.txt'))

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

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


def train_models(X_train, y_train, verbose=True):
    if verbose: print('Training model')




    if verbose: print('Training complete\n')

def encode_data_for_keras(avg_scores):

    # Convert y
    # encode class values as integers
    avg_scores = [round(x, 0) for x in avg_scores]
    encoder = LabelEncoder()
    encoder.fit(avg_scores)
    encoded_Y = encoder.transform(avg_scores)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)

    return dummy_y


def rnn_model():
    model = Sequential()
    model.add(Dense(8, input_dim=glove_size, activation='relu'))
    model.add(Dense(num_scores, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
        avg_scores = all_avg_scores[i]

        keras_labels = encode_data_for_keras(avg_scores);
        num_scores = len(keras_labels[0])
        #X_train, X_test, y_train, y_test = train_test_split(keras_data, keras_labels, train_size=0.9)
        # Split data into test and train
        print('Creating Estimator...')

        estimator = KerasClassifier(build_fn=rnn_model, epochs=200, batch_size=5)
        # Compile model
        print('Compiling Model...')
        kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
        results = cross_val_score(estimator, glove_data, keras_labels, cv=kfold)
        print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


if __name__ == "__main__":
    main()
