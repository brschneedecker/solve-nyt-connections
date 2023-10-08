# -*- coding: utf-8 -*-
"""
# Scrape New York Times Connections

This notebook builds a file of past New York Times Connections puzzles.

Prior puzzles are pulled from https://connectionsanswers.com/nyt-connections-september-20-2023-answers/

# Imports
"""

from lxml import html
from datetime import date, timedelta

import requests
import time
import pickle
import os

"""# Get Puzzle Data

## Define Functions
"""

def daterange(start_date, end_date):
  ''' Yield dates in a date range one at a time '''
  for n in range(int((end_date - start_date).days)):
    yield start_date + timedelta(n)

def get_month_name(a_date):
  ''' Get name of month from a date '''
  return a_date.strftime("%B").lower()

def build_puzzle_link(a_date):
  ''' Build the link to the puzzle answers based on the input date '''
  return f"https://connectionsanswers.com/nyt-connections-{get_month_name(a_date)}-{a_date.day}-{a_date.year}-answers/"

def fetch_puzzle_data(link):
  '''
  Get puzzle data from the specified link

  Returns a tuple with two items:
    - puzzle_data: A set of sets, where each inner set is a word grouping from
                   the puzzle answer -or- 'None' if not puzzle data could be found.

    - result_msg: A string with a description of the result of the attempt
                  to get the puzzle data.
  '''
  w = requests.get(link)

  if w.status_code == 404:
    puzzle_data = None
    result_msg = "Invalid link"
  else:
    # Get page content
    dom_tree = html.fromstring(w.content)

    # Parse page content for puzzle solution words
    puzzle_words = [x.strip() for x in dom_tree.xpath('//li/text()') if x.strip() != '']

    if len(puzzle_words) != 16:
      # This covers the case where a page is formatted weirdly such
      # that the scraping methods doesn't work as intended.
      puzzle_data = None
      result_msg = f"Invalid number of words: ({len(puzzle_words)})"
    else:
      # Assumes words are scraped in puzzle solution grouping order
      puzzle_data = {
          frozenset(puzzle_words[0:4]),
          frozenset(puzzle_words[4:8]),
          frozenset(puzzle_words[8:12]),
          frozenset(puzzle_words[12:16]),
      }
      result_msg = "Fetch successful"

  return puzzle_data, result_msg

def fetch_all_puzzle_data(start_date, end_date):
  '''
  Pulls all puzzles from a specified date range

  Returns a tuple with two items:
    - past_puzzles: A dictionary where keys are the date of the puzzle and
                    the values are the puzzle date for that date
    - missing_puzzles: A list of dates where no puzzle data was found
  '''

  past_puzzles = dict()
  missing_puzzles = []

  for a_date in daterange(start_date, end_date):

    # Build link
    link = build_puzzle_link(a_date)
    print(f"Fetching data from {link}")

    # Fetch data
    puzzle_dict_data, result_msg = fetch_puzzle_data(link)
    print(f"   {result_msg}")

    # Build string for date
    puzzle_dict_key = a_date.strftime("%Y-%m-%d")

    if puzzle_dict_data is not None:
      # Add puzzle data to dictionary
      past_puzzles[puzzle_dict_key] = puzzle_dict_data
    else:
      # Add date to missing list if no puzzle was found
      missing_puzzles.append(puzzle_dict_key)

    # Sleep after each iteration so as to not overwhelm the website with requests
    time.sleep(3)
    print("")
    print("")

  return past_puzzles, missing_puzzles

"""## Get Puzzle Data"""

# Set this to True if doing development
dev_mode = True

# Use small date range if testing
if dev_mode:
  start_date = date(2023, 9, 1)
  end_date = date(2023, 9, 10)
# Otherwise use full date range
else:
  start_date = date(2023, 6, 12)
  end_date = date.today()

past_puzzles, missing_puzzles = fetch_all_puzzle_data(start_date, end_date)

# How many puzzles were not found?
print(f'Missing puzzles: {len(missing_puzzles)}')

for puzzle_date in missing_puzzles:
  print(puzzle_date)

"""# Save Puzzle Data

This section assumes you are running this notebook in Google Colab with your Google Drive account connected.

If you are running this notebook locally or in another cloud environment, update paths and drive mounting according to your chosen environment.
"""

# Run this cell to mount your drive (you will be prompted to sign in)
from google.colab import drive
drive.mount('/content/drive')

# Location where the data will be saved
outpath = '/content/drive/MyDrive/nyt-connections'

# create the directory if it doesn't already exist
if not os.path.exists(outpath):
    os.makedirs(outpath)

# Name of the output file (no extenstion)
outfilename = f'past-puzzles-{date.today().strftime("%Y-%m-%d")}'

# Pickle dictionary using protocol 0.
with open(f'{outpath}/{outfilename}.pkl', 'wb') as output:
  pickle.dump(past_puzzles, output)