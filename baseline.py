# This is the baseline for our project. It uses a linear regression model
# with word count and character count as features.
from collections import Counter, defaultdict
import numpy as np 
import nltk
from sklearn.feature_extraction import DictVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import string
import pdb
import re

from util import *
from featurizers import *

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

###########################################################################

def train_models(
        train_essays,
        train_scores,
        featurizers,
        model_factory=lambda: OneVsRestClassifier(LinearSVC(random_state=0)),
        verbose=True):
  if verbose: print('Featurizing')
  train_X, vectorizer = featurize_datasets(essays_set=train_essays, featurizers=featurizers)
  
  if verbose: print('Training model')
  model = model_factory()
  model.fit(train_X, train_scores)
  if verbose: print('Training complete\n')
  
  return {
    'featurizers': featurizers,
    'vectorizer': vectorizer,
    'model': model,
  }       

def predict(test_set, featurizers, vectorizer, model):
  print('')
  test_X, _ = featurize_datasets(essays_set=test_set, featurizers=featurizers, vectorizer=vectorizer)
  return model.predict(test_X)

###########################################################################

def main():
  all_essays, all_scores = read_data()
  metrics = []

  for essay_set in all_essays.keys():
  # for essay_set in [1]:
    print('\n\n' + '='*20 + ' Processing set {} '.format(essay_set) + '='*20 + '\n')
    essays = all_essays[essay_set]
    scores = all_scores[essay_set]

    # Split data into test and train
    X_train, X_test, y_train, y_test = train_test_split(essays, scores, train_size=0.9)

    featurizers = [ 
                    word_count_featurizer,
                    avg_word_len_featurizer,
                    sentence_count_featurizer
                  ]

    train_result = train_models(train_essays=X_train, 
                                train_scores=y_train, 
                                featurizers=featurizers, 
                                verbose=True)
    predictions = predict(test_set=X_test, 
                          featurizers=featurizers, 
                          vectorizer=train_result['vectorizer'], 
                          model=train_result['model'])

    # print('true | predicted')
    # for i, prediction in enumerate(predictions):
    #   print('%d | %d' % (y_test[i], prediction))

    accuracy = get_accuracy(y_test, predictions)
    cohen_kappa = cohen_kappa_score(y_test, predictions)
    print('Accuracy for set %d: %f' % (essay_set, accuracy))
    print('Cohen Kappa score for set %d: %f' % (essay_set, cohen_kappa))  
    metrics.append((accuracy, cohen_kappa))

  print_metrics(metrics) 


if __name__ == "__main__":
  main() 
