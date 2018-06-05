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
        model_factory=lambda: LinearRegression(),
        # model_factory=lambda: SGDRegressor(),
        verbose=True):
  if verbose: 
    print('Featurizing')
    featurizer_start = datetime.datetime.now()
  train_X, vectorizer, scaler = featurize_datasets(essays=train_essays, 
                                                   featurizers=featurizers, 
                                                   vectorizer=None, 
                                                   scaler=None, 
                                                   train=True,
                                                   essay_set=essay_set)

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

def print_metrics(metrics):
  print('\n\n{0:10s} {1:17s} {2:15s}'.format('set', 'mse', 'cohen'))
  for set_id, metric in enumerate(metrics):
    mse, cohen_kappa = metric
    print('{0:2d} {1:15f} {2:15f}'.format(set_id+1, mse, cohen_kappa))

###########################################################################

def main():
  metrics = []
  all_essays, all_scores = read_data()

  # for essay_set in all_essays.keys():
  for essay_set in [1]:
    print('\n\n' + '='*20 + ' Processing set {} '.format(essay_set) + '='*20 + '\n')
    essays = all_essays[essay_set]
    scores = all_scores[essay_set]

    # Split data into test and train
    X_train, X_test, y_train, y_test = train_test_split(essays, scores, train_size=0.9)

    featurizers = [ 
                    word_count_featurizer,
                    avg_word_len_featurizer,
                    sentence_count_featurizer,
                    #spell_checker_featurizer,
                    punctuation_count_featurizer,
                    stopword_count_featurizer,
                    min_max_word_len_featurizer,
                    #ngram_featurizer,
                    #pos_ngram_featurizer
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

    mse = mean_squared_error(y_test, predictions)
    print('MSE for set %d: %f' % (essay_set, mse))

    #ROUND PREDICTIONS: 
    for i in range(len(predictions)):
      predictions[i] = round(predictions[i])


    print('true | predicted')
    for i, prediction in enumerate(predictions):
      print('%f | %f' % (y_test[i], prediction))

    #WEIGHTS:
    #print('Feature coefficients for set %d:' % essay_set)
    #print('Word count:  %f' % train_result['model'].coef_[0])
    #print('Average word length:  %f' % train_result['model'].coef_[1])
    #print('Sentence count:  %f' % train_result['model'].coef_[2])
    #print('Spell checker:  %f' % train_result['model'].coef_[3])
    #print('Punctuation count:  %f' % train_result['model'].coef_[3])
    #print('Stop word count:  %f' % train_result['model'].coef_[4])
    #print('Min Max word length:  %f' % train_result['model'].coef_[5])
    #print('Ngrams:  %f' % train_result['model'].coef_[7])


    lab_enc = preprocessing.LabelEncoder()
    y_test = lab_enc.fit_transform(y_test)
    predictions = lab_enc.fit_transform(predictions)

    cohen_kappa = cohen_kappa_score(y_test, predictions)
    print('Cohen Kappa score for set %d: %f' % (essay_set, cohen_kappa))  

    metrics.append((mse, cohen_kappa))

  print_metrics(metrics)



if __name__ == "__main__":
  main() 
