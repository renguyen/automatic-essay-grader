# This is the baseline for our project. It uses a linear regression model
# with word count and character count as features.
from collections import Counter, defaultdict
import datetime
import numpy as np 
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import BayesianRidge, LinearRegression, SGDRegressor
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pdb
from tqdm import tqdm

from util import *
from featurizers import *


def featurize_datasets(
      essays_set,
      featurizers=[word_count_featurizer],
      vectorizer=None,
      scaler=None,
      train=True):
  # Create feature counters for each essay.
  essay_features = []
  for e in tqdm(range(len(essays_set))):
    essay_id, essay_text = essays_set[e]
    feature_counter = defaultdict(float)
    for featurizer in featurizers:
      featurizer(feature_counter, essay_text)
    essay_features.append(feature_counter)

  essay_features_matrix = []
  # If we haven't been given a Vectorizer or Scaler, create one and fit it to all the feature counters.
  if vectorizer == None:
    vectorizer = DictVectorizer(sparse=True)
  if scaler == None:
    scaler = preprocessing.StandardScaler()

  if train:
    essay_features_matrix = vectorizer.fit_transform(essay_features).toarray()
    # scaler.fit(essay_features_matrix)
    
  else:
    essay_features_matrix = vectorizer.transform(essay_features).toarray()
    # essay_features_matrix = scaler.transform(essay_features_matrix)

  return essay_features_matrix, vectorizer, scaler

###########################################################################

def train_models(
        train_essays,
        train_scores,
        featurizers,
        model_factory=lambda: LinearRegression(),
        # model_factory=lambda: SGDRegressor(),
        verbose=True):
  if verbose: 
    print('Featurizing')
    featurizer_start = datetime.datetime.now()
  train_X, vectorizer, scaler = featurize_datasets(essays_set=train_essays, featurizers=featurizers, vectorizer=None, scaler=None, train=True)

  if verbose:
    featurizer_end = datetime.datetime.now()
    print('Featurizing took %d seconds \n' % (featurizer_end - featurizer_start).seconds)
  
  if verbose: print('Training model')
  model = model_factory()
  model.fit(train_X, train_scores)
  if verbose: print('Training complete\n')
  
  return {
    'featurizers': featurizers,
    'vectorizer': vectorizer,
    'model': model,
    'scaler': scaler
  }       

def predict(test_set, featurizers, vectorizer, scaler, model):
  print('Predicting')

  test_X, _, __ = featurize_datasets(essays_set=test_set, featurizers=featurizers, vectorizer=vectorizer, scaler=scaler, train=False)
  return model.predict(test_X)

###########################################################################

def main():
  all_essays, all_scores = read_data()
  essays = all_essays[1]  # only essays from set 1
  scores = all_scores[1]

  # Split data into test and train
  X_train, X_test, y_train, y_test = train_test_split(essays, scores, train_size=0.9)

  # Sanity check
  # X_train = [X_train[0]]
  # X_test = [X_train[0]]
  # y_train = [y_train[0]]
  # y_test = [y_train[0]]

  X_train = X_train[:300]
  X_test = X_test[:50]
  y_train = y_train[:300]
  y_test = y_test[:50]

  featurizers = [ 
                  # word_count_featurizer,
                  # avg_word_len_featurizer,
                  # sentence_count_featurizer,
                  # spell_checker_featurizer,
                  # punctuation_count_featurizer,
                  # stopword_count_featurizer,
                  # min_max_word_len_featurizer,
                  ngram_featurizer
                ]

  train_result = train_models(train_essays=X_train, train_scores=y_train, featurizers=featurizers, verbose=True)
  predictions = predict(test_set=X_test, featurizers=featurizers, vectorizer=train_result['vectorizer'], scaler=train_result['scaler'], model=train_result['model'])

  print('loss: %f' % mean_squared_error(y_test, predictions))

  print('true | predicted')
  for i, prediction in enumerate(predictions):
    print('%f | %f' % (y_test[i], prediction))

  lab_enc = preprocessing.LabelEncoder()
  y_test = lab_enc.fit_transform(y_test)
  predictions = lab_enc.fit_transform(predictions)

  print(cohen_kappa_score(y_test, predictions))  


if __name__ == "__main__":
  main() 
