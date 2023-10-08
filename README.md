# solve-nyt-connections

Attempt to solve New York Times Connections puzzles with NLP techniques.

The New York Times has a puzzle called [Connections](https://www.nytimes.com/games/connections) that has players group a 4x4 grid of words into 4 categories of 4 words each.

What constitutes a "group" varies, and is sometimes a pun, relies on some other context that may not be obvious looking at the literal words, or more rarely is a property of the word's sounds or letters (e.g. homophones, palindromes).

This puzzle presents an interesting challenge for categorizing words into groups via scoring words on similarity based on a vector space model given the variety of word senses and attributes applied by the puzzle designer when grouping the words.

For the initial approach, we will use word embeddings of each word in the puzzle and use a clustering technique to make 4 groups of 4 words each. The clustering algorithm used must account for fixed size clusters. The choice of word embedding will also impact the ability to solve the puzzle.