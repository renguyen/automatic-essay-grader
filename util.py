# Common functions shared across all files.
import os
import csv

def read_data():
  '''
  Read the data from a .tsv file into two lists: one for essays, one for scores.
  Each essay is represented by a tuple where the first element is a string concatenation
  of its essay id and essay set and the second is the text itself.
  '''
  data_home = 'data'
  base_data_filename = os.path.join(data_home, 'training_set_rel3.tsv')

  with open(base_data_filename, mode='rt') as f:
    essays = []
    avg_scores = []

    for line in csv.reader(f, delimiter='\t'):
      essay_id = line[0]
      essay_set = line[1]
      if essay_set is '1':

        # Take the average of the graders' scores
        if line[1] is '8' and line[5] is not '':    # some essays in set 8 have a third score
          score = (int(line[3]) + int(line[4]) + int(line[5])) / 3.0
        else:
          score = (int(line[3]) + int(line[4])) / 2.0

        essays.append((essay_id + ',' + essay_set, line[2]))
        avg_scores.append(score)

    return essays, avg_scores