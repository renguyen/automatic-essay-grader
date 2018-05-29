import os
import csv
from collections import Counter
from sklearn.linear_model import LogisticRegression
import pdb

data_home = 'data'
base_data_filename = os.path.join(data_home, 'training_set_rel3.tsv')

# Data will be stored as a dictionary where:
# key(string)   = '{essay_id},{essay_set}'
# value(dict)   = { 'text': the ascii text of a student's response,
#                   'score': the average grader score }
data = {}

with open(base_data_filename, 'rt') as f:
  for line in csv.reader(f, delimiter='\t'):
    key = line[0] + ',' + line[1]
    score = 0
    if line[1] is '8' and line[5] is not '':    # some essays in set 8 have a third score
      score = (int(line[3]) + int(line[4]) + int(line[5])) / 3.0
    else:
      score = (int(line[3]) + int(line[4])) / 2.0
    value = { 'text': line[2], 'score': score }
    data[key] = value

print(data)