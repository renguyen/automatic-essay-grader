from time import time
import os
import csv
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pdb
from tqdm import tqdm
from gensim.sklearn_api import d2vmodel
from util import read_data
import sys
import pickle

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

def train_models(X_train, y_train, verbose=True):

    if verbose: print('Training model')
    param_dist = {'activation' :['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam']}
    n_iter_search = 8
    random_search = RandomizedSearchCV(MLPRegressor(hidden_layer_sizes=(100, ), max_iter = 1000), param_distributions=param_dist,
                                  n_iter=n_iter_search, scoring=scoring_loss)
    start = time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
        " parameter settings." % ((time() - start), n_iter_search))
    print(random_search.cv_results_)
    if verbose: print('Training complete\n')

    # Build best model from results
    return {
        'model': random_search.best_estimator_,
        'result': random_search.cv_results_,
    }


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


def score_model(estimator, X, y):
    lab_enc = preprocessing.LabelEncoder()
    y_lab = lab_enc.fit_transform(y)
    predictions = estimator.predict(X)
    predictions = lab_enc.fit_transform(predictions)
    print(cohen_kappa_score(y_lab, predictions))
    return cohen_kappa_score(y_lab, predictions)

def scoring_loss(estimator, X, y):
    predictions = estimator.predict(X)
    loss = sum([abs(predictions[i]-y[i]) for i in range(len(y))])/len(y)
    print(estimator.get_params())
    print(loss)
    return -loss

def main():

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
        print(glove_data[:10])
        print(avg_scores[:10])
        # Split data into test and train
        X_train, X_test, y_train, y_test = train_test_split(glove_data, avg_scores, train_size=0.9)

        train_result = train_models(X_train, y_train, verbose=True)
        model = train_result['model']

        predictions = model.predict(X_test)

        lab_enc = preprocessing.LabelEncoder()
        y_lab = lab_enc.fit_transform(y_test)
        predictions = model.predict(X_test)
        predictions = lab_enc.fit_transform(predictions)
        print('Scoring on essay set: '+str(i))
        print('Cohens Kappa Score: '+str(cohen_kappa_score(y_lab, predictions)))

if __name__ == "__main__":
    main()
