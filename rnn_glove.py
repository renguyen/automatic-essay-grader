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
from gensim.sklearn_api import d2vmodel

data_home = 'data'
base_data_filename = os.path.join(data_home, 'training_set_rel3.tsv')

essays = []
avg_scores = []

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
    return data

glove_size = 300
glove_lookup = glove2dict(os.path.join(glove_home, 'glove.6B.'+str(glove_size)+'d.txt'))


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
    return essays, avg_scores

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

def train_models(X_train, y_train, verbose=True):
	# print(X_train[:10])
	# print(y_train)

	if verbose: print('Training model')
	param_dist = {'activation' :['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam']}
	n_iter_search = 8
	random_search = RandomizedSearchCV(MLPRegressor(hidden_layer_sizes=(100, )), param_distributions=param_dist,
                                  n_iter=n_iter_search, scoring=score_model)
	start = time()
	random_search.fit(X_train, y_train)
	print("RandomizedSearchCV took %.2f seconds for %d candidates"
        " parameter settings." % ((time() - start), n_iter_search))
	report(random_search.cv_results_)
	if verbose: print('Training complete\n')

    # Build best model from results
	return {
        'model': random_search.best_estimator_,
        'result': random_search.cv_results_,
    }


def create_glove_representations(essays):
	glove_essays = np.zeros((len(essays), glove_size))
	for i in range(len(essays)):
		essay = essays[i][1]
		glove = [glove_lookup[w] for w in essay.split(' ') if w in glove_lookup]
		glove = [sum(i) / len(i) for i in zip(*glove)]
		if len(glove) is not glove_size:
			glove_essays[i] = np.zeros(glove_size)
		else:
			glove_essays[i] = glove
	return glove_essays

def score_model(estimator, X, y):
    lab_enc = preprocessing.LabelEncoder()
    y_lab = lab_enc.fit_transform(y)
    predictions = estimator.predict(X)
    predictions = lab_enc.fit_transform(predictions)
    print(cohen_kappa_score(y_lab, predictions))
    return cohen_kappa_score(y_lab, predictions)

def main():
    essays, avg_scores = read_data()

    glove_data = create_glove_representations(essays)

    # Split data into test and train
    X_train, X_test, y_train, y_test = train_test_split(glove_data, avg_scores, train_size=0.9)

    train_result = train_models(X_train, y_train, verbose=True)

    predictions = model.predict(X_test)

    print(cohen_kappa_score(y_test, predictions))

if __name__ == "__main__":
    main()
