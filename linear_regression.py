# This is the baseline for our project. It uses a linear regression model
# with word count and character count as features.
from collections import Counter, defaultdict
import datetime
import numpy as np 
from nltk import pos_tag
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import BayesianRidge, LinearRegression 
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import string
import pdb
import re
import sys
from tqdm import tqdm
import enchant
from enchant.checker import SpellChecker

from util import *

stopWords = set(stopwords.words('english'))

# Removes the UnicodeDecodeError when using NLTK sent_tokenize
reload(sys)
sys.setdefaultencoding('utf-8')

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
  essay_without_punctuation = essay.translate(None, string.punctuation)
  words = essay_without_punctuation.split()
  lengths = 0.0
  for word in words:
    lengths += len(word)
  feature_counter['avg_word_len'] = lengths / len(words)

def sentence_count_featurizer(feature_counter, essay):
  '''
  Adds sentence count as a feature.
  '''
  # try:
  #  feature_counter['sentence_count'] = len(sent_tokenize(essay))
  # except UnicodeDecodeError, e:
  #  pdb.set_trace()
  sentences = essay.count(('?<!\.)\.(?!\.)')) + essay.count("?") + essay.count("!")
  feature_counter['sentence_count'] = sentences

def spell_checker_featurizer(feature_counter, essay):
  chkr = SpellChecker("en_UK","en_US")
  chkr.set_text(essay)
  counter = 0
  for error in chkr: 
    counter += 1
  #print(counter)
  feature_counter['spell_checker'] = counter

def punctuation_count_featurizer(feature_counter, essay):
  '''
  Adds different punctuation counts as features.
  '''
  feature_counter['question_mark_count'] = essay.count("?")
  feature_counter['exclamation_mark_count'] = essay.count("!")

def stopword_count_featurizer(feature_counter, essay):
  '''
  Adds number of stopgwords (as defined by NLTK corpus) as features.
  '''
  feature_counter['stopword_count'] = 0
  essay_without_punctuation = essay.translate(None, string.punctuation)
  for word in essay_without_punctuation.split():
    if word in stopWords:
      feature_counter['stopword_count'] += 1

def min_max_word_len_featurizer(feature_counter, essay):
  '''
  Adds the minimum and maximum word lengths in the essay, ignoring stopwords
  and censored pronouns.
  '''
  essay_without_punctuation = essay.translate(None, string.punctuation)
  min_len = float('inf')
  max_len = float('-inf')
  for word in essay_without_punctuation.split():
    min_len = min(min_len, len(word))
    max_len = max(max_len, len(word))

  feature_counter['min_word_len'] = min_len
  feature_counter['max_word_len'] = max_len

def ngram_featurizer(feature_counter, essay, ngrams=4, plain=True, pos=True):
  '''
  Adds ngrams as a feature. plain=True will add normal word ngrams as a feature
  and pos=True will also add POS ngrams as a feature.
  '''
  essay_without_punctuation = essay.translate(None, '!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~')
  pos_tags = pos_tag(essay_without_punctuation.split())
  pos_tags = [('<S>', '<S>')] + pos_tags + [('</S>', '</S>')]
  for i in range(len(pos_tags) - (ngrams - 1)):
    norm_key = pos_tags[i][0] + ' '
    pos_key = pos_tags[i][1] + ' '
    for j in range(1, ngrams):
      if plain:
        norm_key += pos_tags[i+j][0] + ' '
      if pos:
        if pos_tags[i+j][0].startswith('@'):  # Personally identifying nouns were removed
          pos_key += 'NN '
        else:
          pos_key += pos_tags[i+j][1] + ' '

    if plain:
      feature_counter[norm_key.strip()] += 1
    if pos:
      feature_counter[pos_key.strip()] += 1

def high_vocab_count_featurizer(feature_counter, essay):
  '''
  Adds number of "high vocabulary words" as a feature. 
  TODO: determine high vocab words.
  '''
  pass

def essay_prompt_similarity_featurizer(feature_counter, essay):
  '''
  Adds score for how similar an essay is to the score.
  '''
  pass



def featurize_datasets(
      essays_set,
      featurizers=[word_count_featurizer],
      vectorizer=None,
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
  # If we haven't been given a Vectorizer, create one and fit it to all the feature counters.
  if vectorizer == None:
    vectorizer = DictVectorizer(sparse=True)

  if train:
    essay_features_matrix = vectorizer.fit_transform(essay_features).toarray()
  else:
    essay_features_matrix = vectorizer.transform(essay_features).toarray()

  return essay_features_matrix, vectorizer

###########################################################################

def train_models(
        train_essays,
        train_scores,
        featurizers,
        model_factory=lambda: LinearRegression(),
        verbose=True):
  if verbose: 
    print('Featurizing')
    featurizer_start = datetime.datetime.now()
  train_X, vectorizer = featurize_datasets(essays_set=train_essays, featurizers=featurizers, vectorizer=None, train=True)

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
  }       

def predict(test_set, featurizers, vectorizer, model):
  print('Predicting')

  test_X, _ = featurize_datasets(essays_set=test_set, featurizers=featurizers, vectorizer=vectorizer, train=False)
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

  # X_train = X_train[:300]
  # X_test = X_test[:50]
  # y_train = y_train[:300]
  # y_test = y_test[:50]

  featurizers = [ 
                  # word_count_featurizer,
                  # char_count_featurizer,
                  # avg_word_len_featurizer,
                  # sentence_count_featurizer,
                  # spell_checker_featurizer,
                  # punctuation_count_featurizer,
                  # stopword_count_featurizer,
                  # min_max_word_len_featurizer,
                  ngram_featurizer,
                ]

  train_result = train_models(train_essays=X_train, train_scores=y_train, featurizers=featurizers, verbose=True)
  predictions = predict(test_set=X_test, featurizers=featurizers, vectorizer=train_result['vectorizer'], model=train_result['model'])

  print('loss: %f' % mean_squared_error(y_test, predictions))

  # print('true | predicted')
  # for i, prediction in enumerate(predictions):
  #   print('%f | %f' % (y_test[i], prediction))

  lab_enc = preprocessing.LabelEncoder()
  y_test = lab_enc.fit_transform(y_test)
  predictions = lab_enc.fit_transform(predictions)

  print(cohen_kappa_score(y_test, predictions))  


if __name__ == "__main__":
  main() 
