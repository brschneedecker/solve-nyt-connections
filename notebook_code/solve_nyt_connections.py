# -*- coding: utf-8 -*-
"""
# Solve New York Times Connections

In this notebook we'll explore using dense word embeddings to cluster the words in [Connections](https://www.nytimes.com/games/connections) puzzles to see how well it is able to solve the puzzle.

# Imports and Setup
"""

### This cell might take 3 min to run ###
! echo "Installing Magnitude.... (please wait, can take a while)"
! (curl https://raw.githubusercontent.com/plasticityai/magnitude/master/install-colab.sh | /bin/bash 1>/dev/null 2>/dev/null)
! echo "Done installing Magnitude."

!pip install spacy
!python -m spacy download en_core_web_sm
!python -m spacy download en

!pip uninstall -y annoy
!pip install annoy

!pip install -U lz4==1.0.0
!pip install -U xxhash==1.0.1
!pip install -U fasteners==0.14.1

import spacy

# Check your python version
!python --version

spacy.load('en_core_web_sm')
import collections
collections.Sequence = collections.abc.Sequence
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Iterable = collections.abc.Iterable
collections.MutableSet = collections.abc.MutableSet
collections.Callable = collections.abc.Callable

from pymagnitude import * # if you encounter an error for this line, try re-running it - I know it's silly but it might work

# Commented out IPython magic to ensure Python compatibility.
# # This might take a while....
# %%capture
# !wget http://magnitude.plasticity.ai/word2vec/light/GoogleNews-vectors-negative300.magnitude
# !wget http://magnitude.plasticity.ai/glove/medium/glove.6B.50d.magnitude
# !wget http://magnitude.plasticity.ai/glove/medium/glove.6B.100d.magnitude
# !wget http://magnitude.plasticity.ai/glove/medium/glove.6B.200d.magnitude
# !wget http://magnitude.plasticity.ai/glove/medium/glove.6B.300d.magnitude
# !wget http://magnitude.plasticity.ai/glove/medium/glove.840B.300d.magnitude

!pip install k-means-constrained

from itertools import combinations
from prettytable import PrettyTable
from sklearn.cluster import KMeans
import random
import pandas as pd
import numpy as np
import scipy.stats as stats

from k_means_constrained import KMeansConstrained
from prettytable import PrettyTable
import pickle

"""# Load Data"""

# Run this cell to mount your drive (you will be prompted to sign in)
from google.colab import drive
drive.mount('/content/drive')

indata = '/content/drive/MyDrive/nyt-connections/past-puzzles-2023-10-08.pkl'
with open(indata, 'rb') as pkl_file:
  past_puzzles = pickle.load(pkl_file)

# Print a few obs to verify data was loaded correctly
for key in list(past_puzzles.keys())[:5]:
  print(past_puzzles[key])

"""# Solve Puzzles

## Define Connections Puzzle Object Type
"""

class Connections():

  def __init__(self, puzzle_date, puzzle_data, vector_model):
    self.puzzle_date = puzzle_date
    self.puzzle_answers = puzzle_data
    self.vector_model = vector_model
    self.vectors = Magnitude(vector_model)
    self.puzzle_words = self.get_puzzle_words(self.puzzle_answers)
    self.puzzle_predictions = set()

  def get_puzzle_words(self, puzzle_answers):
    ''' Get Connections puzzle words from puzzle answer groupings'''
    random.seed(123)
    puzzle_words = [word for group in self.puzzle_answers for word in group]
    random.shuffle(puzzle_words)
    return puzzle_words

  def word_to_vector(self, word):
    ''' Convert a word to a vector based on the chosen vector space model '''
    if word in self.vectors:
      vector = self.vectors.query(word)
    else:
      vector = np.random.rand(self.vectors.dim)
    return vector

  def build_vector_list(self):
    ''' Build a list of vectors representing puzzle words '''
    return [self.word_to_vector(word) for word in self.puzzle_words]

  def solve(self):
    ''' Solve the puzzle! '''

    X = self.build_vector_list()

    # Instantiate and fit model
    clf = KMeansConstrained(
        n_clusters=4,
        size_min=4,
        size_max=4,
        random_state=0
    )
    clf.fit_predict(X)

    # Group words into their labeled categories
    classifier_groups = dict()

    for word, label in zip(self.puzzle_words, clf.labels_):
      if label not in classifier_groups:
        classifier_groups[label] = []
      classifier_groups[label].append(word)

    for label in classifier_groups.keys():
      self.puzzle_predictions.add(frozenset(classifier_groups[label]))

  def evaluate(self):
    ''' Check if the solver got any groups correct '''
    self.correct_predictions = self.puzzle_answers & self.puzzle_predictions
    self.wrong_predictions = self.puzzle_predictions - self.correct_predictions

  def display_results(self):
    ''' Print results to the console '''
    x = PrettyTable()

    print(f"Guesses for model {self.vector_model}")
    x = PrettyTable()
    x.add_rows(self.puzzle_predictions)
    print(x)

    print("")
    print("")

    print(f"Correct guesses for model {self.vector_model}")
    if self.correct_predictions:
      x = PrettyTable()
      x.add_rows(self.correct_predictions)
      print(x)
    else:
      print("None")

    print("")
    print("")

"""## Run Puzzle Solver"""

def run_puzzle_solver(puzzle_date, puzzle_data, vector_model, display_results=True):
  puzzle = Connections(puzzle_date, puzzle_data, vector_model)
  puzzle.solve()
  puzzle.evaluate()

  if display_results:
    puzzle.display_results()

  return puzzle

vector_model = 'GoogleNews-vectors-negative300.magnitude'
total_correct = 0

for puzzle_date in past_puzzles:
  puzzle = run_puzzle_solver(puzzle_date, past_puzzles[puzzle_date], vector_model, display_results=False)
  total_correct += len(puzzle.correct_predictions)

print(f"Total correct guesses: {total_correct}")
print(f"Total puzzles attempted: {len(past_puzzles.keys())}")
print(f"Average correct guesses per puzzle: {total_correct/len(past_puzzles.keys())}")

# TODO - Try more vector models, record and plot results to compare performance

# vector_models = ['GoogleNews-vectors-negative300.magnitude',
#                  'glove.6B.50d.magnitude',
#                  'glove.6B.100d.magnitude',
#                  'glove.6B.200d.magnitude',
#                  'glove.6B.300d.magnitude',
#                  'glove.840B.300d.magnitude',]


puzzles = dict()

"""# End of Notebook"""