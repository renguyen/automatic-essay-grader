# Common functions shared across all files.
import os
import csv
import pdb

def read_data():
  '''
  Read the data from a .tsv file into two dictionaries: one for essays, one for scores.
  Each essay is represented by a tuple where the first element is a string concatenation
  of its essay id and essay set and the second is the text itself. The keys in the dictionaries
  refer to the essay set.
  '''
  data_home = 'data'
  base_data_filename = os.path.join(data_home, 'training_set_rel3.tsv')

  with open(base_data_filename, mode='rt') as f:
    essays = { 
                1: [],
                2: [],
                3: [],
                4: [],
                5: [],
                6: [],
                7: [],
                8: [],
             }
    scores = { 
                1: [],
                2: [],
                3: [],
                4: [],
                5: [],
                6: [],
                7: [],
                8: [],
              }

    for line in csv.reader(f, delimiter='\t'):
      essay_id = int(line[0])
      essay_set = int(line[1])
      score = int(line[6])

      essays[essay_set].append((essay_id, line[2]))
      scores[essay_set].append(score)

    return essays, scores

def get_accuracy(test, predictions):
  num_right = 0.0
  for i, correct in enumerate(test):
    if correct == predictions[i]:
      num_right += 1

  return num_right / len(test)

def print_metrics(metrics):
  print('\n\n{0:9s} {1:15s} {2:15s}'.format('set', 'accuracy', 'cohen'))
  for set_id, metric in enumerate(metrics):
    accuracy, cohen_kappa = metric
    print('{0:2d} {1:15f} {2:15f}'.format(set_id+1, accuracy, cohen_kappa))