# This is the baseline for our project. It uses a linear regression model
# with word count and character count as features.
from collections import Counter, defaultdict
import datetime
import numpy as np 
from sklearn.feature_extraction import DictVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pdb
from tqdm import tqdm

from featurizers import *
from util import *

BASELINE_NUM_RUNS = 5 

def featurize_datasets(
      essays,
      essay_set,
      featurizers=[word_count_featurizer],
      vectorizer=None,
      scaler=None,
      train=True):
  # Create feature counters for each essay.
  essay_features = []
  for e in tqdm(range(len(essays))):
    essay_id, essay_text = essays[e]
    feature_counter = defaultdict(float)
    for featurizer in featurizers:
      featurizer(feature_counter=feature_counter, essay=essay_text, essay_set=essay_set)
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
        essay_set,
        model_factory=lambda: OneVsRestClassifier(LinearSVC(random_state=0)),
        verbose=True):
  if verbose: 
    print('Featurizing')
    start = datetime.datetime.now()
  train_X, vectorizer, scaler = featurize_datasets(essays=train_essays, 
                                                   featurizers=featurizers, 
                                                   vectorizer=None, 
                                                   scaler=None, 
                                                   train=True,
                                                   essay_set=essay_set)

  if verbose:
    end = datetime.datetime.now()
    print('Featurizing took %d seconds \n' % (end - start).seconds)
  
  if verbose: 
    print('Training model')
    start = datetime.datetime.now()
  model = model_factory()
  model.fit(train_X, train_scores)
  if verbose: 
    end = datetime.datetime.now()
    print('Training complete')
    print('Training took %d seconds \n' % (end - start).seconds)
  
  return {
    'featurizers': featurizers,
    'vectorizer': vectorizer,
    'model': model,
    'scaler': scaler
  }       

def predict(test_set, featurizers, vectorizer, scaler, model, essay_set):
  print('Predicting')

  test_X, _, __ = featurize_datasets(essays=test_set, 
                                     featurizers=featurizers, 
                                     vectorizer=vectorizer, 
                                     scaler=scaler, 
                                     train=False,
                                     essay_set=essay_set)
  return model.predict(test_X)

###########################################################################

def main():
  all_essays, all_scores = read_data()
  metrics = [
              ([], []),
              ([], []),
              ([], []),
              ([], []),
              ([], []),
              ([], []),
              ([], []),
              ([], []),
            ]

  for essay_set in all_essays.keys():
    print('\n\n' + '='*20 + ' Processing set {} '.format(essay_set) + '='*20 + '\n')
    essays = all_essays[essay_set]
    scores = all_scores[essay_set]

    # Split data into test and train
    X_train, X_test, y_train, y_test = train_test_split(essays, scores, train_size=0.9)

    featurizers = [ 
                    word_count_featurizer,
                    avg_word_len_featurizer,
                    sentence_count_featurizer,
                  ]

    train_result = train_models(train_essays=X_train, 
                                train_scores=y_train, 
                                featurizers=featurizers, 
                                verbose=True, 
                                essay_set=essay_set)
    predictions = predict(test_set=X_test, 
                          featurizers=featurizers, 
                          vectorizer=train_result['vectorizer'], 
                          scaler=train_result['scaler'], 
                          model=train_result['model'],
                          essay_set=essay_set)

    # print('true | predicted')
    # for i, prediction in enumerate(predictions):
    #   print('%d | %d' % (y_test[i], prediction))

    accuracy = get_accuracy(y_test, predictions)
    cohen_kappa = cohen_kappa_score(y_test, predictions)
    print('Accuracy for set %d: %f' % (essay_set, accuracy))
    print('Cohen Kappa score for set %d: %f' % (essay_set, cohen_kappa))  
    metrics_index = essay_set - 1 
    metrics[metrics_index[0]].append(accuracy)
    metrics[metrics_index[1]].append(cohen_kappa)

  print_metrics_with_accuracy(metrics)


if __name__ == "__main__":
  main() 
