# Contains various feature extractors to be used across different models.

import nltk
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from enchant.checker import SpellChecker
import sys
import string
import pdb
import csv
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Removes the UnicodeDecodeError when using NLTK sent_tokenize
reload(sys)
sys.setdefaultencoding('utf-8')

stopWords = set(stopwords.words('english'))

# Read list of high vocab words
vocab_words = []
with open('vocab.txt', mode='rt') as f:
  for line in csv.reader(f, delimiter='\t'):
    vocab_words.append(line[0])

# Read essay prompts for each set
essay_prompts = []
with open('prompts.txt', mode='rt') as f: 
  essay_prompts = f.readlines()

###########################################################################
#                               FEATURIZERS                               #
###########################################################################

def word_count_featurizer(feature_counter, essay, essay_set=None):
  '''
  Adds word count as a feature.
  '''
  word_count = len(essay.split(' '))
  feature_counter['word_count'] = word_count

def avg_word_len_featurizer(feature_counter, essay, essay_set=None):
  '''
  Adds the average length of the words as a feature.
  '''
  essay_without_punctuation = essay.translate(None, string.punctuation)
  words = essay_without_punctuation.split()
  lengths = 0.0
  for word in words:
    lengths += len(word)
  feature_counter['avg_word_len'] = lengths / len(words)

def sentence_count_featurizer(feature_counter, essay, essay_set=None):
  '''
  Adds sentence count as a feature.
  '''
  try:
    sentences = sent_tokenize(essay)
    feature_counter['sentence_count'] = len(sentences)
    min_len = float('inf')
    max_len = float('-inf')
    for sentence in sentences:
      sentence_len = len(sentence.split())
      feature_counter['sentence_len_%d' % sentence_len] += 1
      min_len = min(min_len, sentence_len)
      max_len = max(max_len, sentence_len)
    feature_counter['sentence_len_range'] = max_len - min_len
  except UnicodeDecodeError, e:
    print('error parsing sentences for essay:\n %s' % essay)

def avg_sentence_length_featurizer(feature_counter, essay, essay_set=None):
  '''
  Adds sentence count as a feature.
  '''
  try:
    sentences = sent_tokenize(essay)
    sentence_lengths = 0.0
    for sentence in sentences:
      sentence_lengths += len(sentence.split())
    feature_counter['avg_sentence_length'] = sentence_lengths / len(sentences)
  except UnicodeDecodeError, e:
    print('error parsing sentences for essay:\n %s' % essay)

def spell_checker_featurizer(feature_counter, essay, essay_set=None):
  chkr = SpellChecker("en_UK","en_US")
  chkr.set_text(essay)
  counter = 0
  for error in chkr: 
    counter += 1
  #print(counter)
  feature_counter['spell_checker'] = counter

def punctuation_count_featurizer(feature_counter, essay, essay_set=None):
  '''
  Adds different punctuation counts as features.
  '''
  feature_counter['question_mark_count'] = essay.count("?")
  feature_counter['exclamation_mark_count'] = essay.count("!")

def stopword_count_featurizer(feature_counter, essay, essay_set=None):
  '''
  Adds number of stopgwords (as defined by NLTK corpus) as features.
  '''
  feature_counter['stopword_count'] = 0
  essay_without_punctuation = essay.translate(None, string.punctuation)
  for word in essay_without_punctuation.split():
    if word in stopWords:
      feature_counter['stopword_count'] += 1

def min_max_word_len_featurizer(feature_counter, essay, essay_set=None):
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

def ngram_featurizer(feature_counter, essay, ngrams=2, essay_set=None):
  '''
  Adds ngrams as a feature. 
  '''
  essay = '<S> ' + essay + ' </S>'
  words = essay.split()
  for i in range(len(words) - (ngrams - 1)):
    key = words[i] + ' '
    for j in range(1, ngrams):
      key += words[i+j] + ' '

    feature_counter[key.strip()] += 1

def pos_ngram_featurizer(feature_counter, essay, ngrams=2, essay_set=None):
  '''
  Adds part of speech ngrams as a feature. 
  '''
  essay_without_punctuation = essay.translate(None, '!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~')
  pos_tags = pos_tag(essay_without_punctuation.split())
  pos_tags = [('<S>', '<S>')] + pos_tags + [('</S>', '</S>')]
  for i in range(len(pos_tags) - (ngrams - 1)):
    key = pos_tags[i][1] + ' '
    for j in range(1, ngrams):
      if pos_tags[i+j][0].startswith('@'):  # Personally identifying nouns were removed
        key += 'NN '
      else:
        key += pos_tags[i+j][1] + ' '

    feature_counter[key.strip()] += 1

def high_vocab_count_featurizer(feature_counter, essay, essay_set=None):
  '''
  Adds number of "high vocabulary words" as a feature. 
  TODO: determine high vocab words.
  '''
  essay_without_punctuation = essay.translate(None, string.punctuation)
  for word in essay_without_punctuation.split():
    if word in vocab_words:
      feature_counter[word] += 1
      #print('%s | %d' % (word, feature_counter[word]))

###########################################################################
# ESSAY PROMPT SIMILARITY FEATURIZER AND HELPER FUNCTIONS
# from https://stackoverflow.com/questions/8897593/similarity-between-two-text-documents

def stem_tokens(tokens):
  return [stemmer.stem(item) for item in tokens]

def normalize(text):
  '''remove punctuation, lowercase, stem'''
  return stem_tokens(word_tokenize(text.lower().translate(remove_punctuation_map)))

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def essay_prompt_similarity_featurizer(feature_counter, essay, essay_set=None):
  '''
  Adds score for how similar an essay is to the score.
  '''
  essay_prompt = essay_prompts[essay_set-1]
  tfidf = vectorizer.fit_transform([essay_prompt, essay])
  cosine_sim = ((tfidf * tfidf.T).A)[0,1]
  feature_counter['essay_prompt_similarity'] = cosine_sim

###########################################################################

def unique_word_count_featurizer(feature_counter, essay, essay_set=None):
  '''
  Adds number of unique words in the essay.
  '''
  essay_without_punctuation = essay.translate(None, string.punctuation)
  words = essay_without_punctuation.split()
  feature_counter['unique_word_count'] = len(set(words))

def quotes_count_featurizer(feature_counter, essay, essay_set=None):
  '''
  Adds number of times the writer quotes someone.
  '''
  feature_counter['quotes_count'] = essay.count('"') / 2

def bag_of_words_featurizer(feature_counter, essay, essay_set=None): 

  '''
  Adds the different words and how many times they appear as a feature.
  '''
  words = word_tokenize(essay.lower().translate(remove_punctuation_map))
  for word in words:
    feature_counter[word] += 1

def bag_of_pos_featurizer(feature_counter, essay, essay_set=None):
  '''
  Adds the different POS and how many times they appear as a feature.
  '''
  essay_without_punctuation = essay.translate(None, '!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~')
  pos_tags = pos_tag(essay_without_punctuation.split())
  for pos_tag in pos_tags:
    if pos_tag[0].startswith('@'):  # Personally identifying nouns were removed
      feature_counter['NN'] += 1
    else:
      feature_counter[pos_tag[1]] += 1

