# This is the baseline for our project. It uses a linear regression model
# with word count and character count as features.

import os
import csv
from collections import Counter, defaultdict
import numpy as np 
import nltk
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import BayesianRidge, LinearRegression 
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import string
import pdb

data_home = 'data'
base_data_filename = os.path.join(data_home, 'training_copy.tsv')

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

def char_count_featurizer(feature_counter, essay):
  '''
  Adds character count as a feature.
  '''
  feature_counter['character_count'] = len(essay)

def avg_word_len_featurizer(feature_counter, essay):
  '''
  Adds the average length of the words as a feature.
  '''
  essay_without_puncutation = essay.translate(None, string.punctuation)
  words = essay_without_puncutation.split()
  lengths = 0.0
  for word in words:
    lengths += len(word)
  feature_counter['avg_word_len'] = lengths / len(words)


def sentence_count_featurizer(feature_counter, essay):
  '''
  Adds sentence count as a feature.
  '''
  try:
    feature_counter['sentence_count'] = len(nltk.sent_tokenize(essay))
  except UnicodeDecodeError, e:
    pdb.set_trace()

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
        featurizers,
        model_factory=lambda: LinearRegression(),
        verbose=True):
  if verbose: print('Featurizing')
  train_X, vectorizer = featurize_datasets(essays_set=X_train, featurizers=featurizers)
  
  if verbose: print('Training model')
  model = model_factory()
  model.fit(train_X, y_train)
  if verbose: print('Training complete\n')
  
  return {
    'featurizers': featurizers,
    'vectorizer': vectorizer,
    'model': model,
  }       

def predict(featurizers, vectorizer, model):
  test_X, _ = featurize_datasets(essays_set=X_test, featurizers=featurizers, vectorizer=vectorizer)
  return model.predict(test_X)

###########################################################################

def main():
  read_data()

  global X_train
  global X_test
  global y_train
  global y_test

  # Split data into test and train
  X_train, X_test, y_train, y_test = train_test_split(essays, avg_scores, train_size=0.7)

  featurizers = [ 
                  word_count_featurizer,
                  char_count_featurizer,
                  avg_word_len_featurizer,
                  sentence_count_featurizer
                ]

  train_result = train_models(featurizers, verbose=True)
  predictions = predict(featurizers, train_result['vectorizer'], train_result['model'])
  # print(predictions)

  lab_enc = preprocessing.LabelEncoder()
  y_test = lab_enc.fit_transform(y_test)
  predictions = lab_enc.fit_transform(predictions)

  print(cohen_kappa_score(y_test, predictions))  


if __name__ == "__main__":
  main() 