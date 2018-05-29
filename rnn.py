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

data_home = 'data'
base_data_filename = os.path.join(data_home, 'training_set_rel3.tsv')

essays = []
avg_scores = []

X_train = None
X_test = None
y_train = None
y_test = None


def read_data():
    '''
    Read the data from a .tsv file into two lists: one for essays, one for scores.
    Each essay is represented by a tuple where the first element is a string concatenation
    of its essay id and essay set and the second is the text itself.
    '''
    with open(base_data_filename, 'rt') as f:
        for line in csv.reader(f, delimiter='\t'):
            essay_id = line[0]
            essay_set = line[1]

            # Take the average of the graders' scores
            if line[1] is '8' and line[5] is not '':    # some essays in set 8 have a third score
                score = (int(line[3]) + int(line[4]) + int(line[5])) / 3.0
            else:
                score = (int(line[3]) + int(line[4])) / 2.0

            essays.append((essay_id + ',' + essay_set, line[2]))
            avg_scores.append(score)

###########################################################################
#                                                                         #
#                             FEATURIZERS                                 #
#                                                                         #
###########################################################################

def word_count_featurizer(feature_counter, essay):
    '''
    Adds word count as a feature.
    '''
    word_count = len(essay.split(' '))
    feature_counter['word_count'] = word_count

def featurize_datasets(
      essays_set,
      featurizers=[word_count_featurizer],
      vectorizer=None):
    # Create feature counters for each essay.
    essay_features = []
    for essay in essays_set:
        essay_id, essay_text = essay
        feature_counter = {}
        for featurizer in featurizers:
            featurizer(feature_counter, essay_text)
        essay_features.append(feature_counter)

    essay_features_matrix = []
    # If we haven't been given a Vectorizer, create one and fit it to all the feature counters.
    if vectorizer == None:
        vectorizer = DictVectorizer(sparse=True)

    essay_features_matrix = vectorizer.fit_transform(essay_features).toarray()
    return essay_features_matrix, vectorizer


def score_model(estimator, X, y):
    lab_enc = preprocessing.LabelEncoder()
    y_lab = lab_enc.fit_transform(y)
    predictions = estimator.predict(X)
    predictions = lab_enc.fit_transform(predictions)
    print(cohen_kappa_score(y_lab, predictions))
    return cohen_kappa_score(y_lab, predictions)

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean Cohen Kappa score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def train_models(
        featurizers,
        verbose=True):
    if verbose: print('Featurizing')
    train_X, vectorizer = featurize_datasets(essays_set=X_train, featurizers=featurizers)

    if verbose: print('Training model')
    param_dist = {'activation' :['identity', 'logistic', 'tanh', 'relu']}
    n_iter_search = 4
    random_search = RandomizedSearchCV(MLPRegressor(hidden_layer_sizes=(100, )), param_distributions=param_dist,
                                  n_iter=n_iter_search, scoring=score_model)
    start = time()
    random_search.fit(train_X, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
        " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)
    if verbose: print('Training complete\n')

    # Build best model from results

    return {
        'featurizers': featurizers,
        'vectorizer': vectorizer,
        'model': random_search.best_estimator_,
        'result': random_search.cv_results_,
    }

def predict(featurizers, vectorizer, model):
  test_X, _ = featurize_datasets(essays_set=X_test, featurizers=featurizers, vectorizer=vectorizer)
  return model.predict(test_X)


def main():
    read_data()

    global X_train
    global X_test
    global y_train
    global y_test

    # Split data into test and train
    X_train, X_test, y_train, y_test = train_test_split(essays, avg_scores, train_size=0.7)

    featurizers = [word_count_featurizer]

    train_result = train_models(featurizers, verbose=True)
    test_X, _ = featurize_datasets(essays_set=X_test, featurizers=featurizers, vectorizer=train_result['vectorizer'])

    predictions = predict(featurizers, train_result['vectorizer'], train_result['model'])
    # print(predictions)

    lab_enc = preprocessing.LabelEncoder()
    y_test = lab_enc.fit_transform(y_test)
    predictions = lab_enc.fit_transform(predictions)# specify parameters and distributions to sample from
    # specify parameters and distributions to sample from
    print(cohen_kappa_score(y_test, predictions))

if __name__ == "__main__":
    main()
