# Contains various feature extractors to be used across different models.

from nltk import pos_tag
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from enchant.checker import SpellChecker
import sys
import string
import pdb
import csv
import os

# Removes the UnicodeDecodeError when using NLTK sent_tokenize
reload(sys)
sys.setdefaultencoding('utf-8')

stopWords = set(stopwords.words('english'))

vocab_words = []
with open('vocab.txt', mode='rt') as f:
  for line in csv.reader(f, delimiter='\t'):
    vocab_words.append(line[0])


def word_count_featurizer(feature_counter, essay):
  '''
  Adds word count as a feature.
  '''
  word_count = len(essay.split(' '))
  feature_counter['word_count'] = word_count

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

def ngram_featurizer(feature_counter, essay, ngrams=2, plain=True, pos=True):
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
  essay_without_punctuation = essay.translate(None, string.punctuation)
  for word in essay_without_punctuation.split():
    if word in vocab_words:
      feature_counter[word] += 1
      #print('%s | %d' % (word, feature_counter[word]))

def essay_prompt_similarity_featurizer(feature_counter, essay):
  '''
  Adds score for how similar an essay is to the score.
  '''
  pass