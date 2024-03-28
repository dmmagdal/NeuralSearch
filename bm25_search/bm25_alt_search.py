# bm25_alt_search.py
# Using the code from the medium article to implement a BM25 search
# engine.
# Python 3.7
# Windows/MacOS/Linux


import math
from collections import Counter


# NOTE:
# In general, the preprocessing for documents/datasets for BM25 goes
# like this:
# 1) Tokenize text into individual words.
# 2) Remove special characters
# 3) Remove stopwords
# 4) Normalize text 
#   -> Process text into a standard/canonical form
#   -> Map near identical words together
#   -> Remove/ignore out-of-vocabulary (OOV) words
# 5) Stem words to their roots


class BM25:
    def __init__(self, corpus):
        self.corpus_size = len(corpus)
        self.avg_doc_length = sum(map(len, corpus)) / self.corpus_size
        self.corpus = corpus
        self.tf = []
        self.df = {}
        self.idf = {}
        self.k1 = 1.5
        self.b = 0.75
        self.initialize()

    def initialize(self):
        for document in self.corpus:
            frequencies = dict(Counter(document))
            self.tf.append(frequencies)

            for term in frequencies:
                if term in self.df:
                    self.df[term] += 1
                else:
                    self.df[term] = 1

        for term, freq in self.df.items():
            self.idf[term] = math.log(self.corpus_size - freq + 0.5) / (freq + 0.5)

    def compute_score(self, query, index):
        score = 0
        doc_length = len(self.corpus[index])

        for term in query:
            if term not in self.tf[index]:
                continue

            tf = self.tf[index][term]
            numerator = self.idf[term] * tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            score += numerator / denominator

        return score

    def search(self, query):
        scores = []

        for index in range(self.corpus_size):
            score = self.compute_score(query, index)
            scores.append((index, score))

        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return sorted_scores


# Example usage
corpus = [
    ["I", "love", "programming"],
    ["Python", "is", "my", "favorite", "programming", "language"],
    ["I", "enjoy", "writing", "code", "in", "Python"],
    ["Java", "is", "another", "popular", "programming", "language"],
    ["I", "find", "programming", "fascinating"],
]

bm25 = BM25(corpus)

query = "Python programming"
query_terms = query.lower().split()
results = bm25.search(query_terms)

print("Search results for '{}':".format(query))
for index, score in results:
    print("Document {}: Score = {}".format(index, score))
# Search results for 'Python programming':
# Document 0: Score = 0.10988214311874374
# Document 4: Score = 0.09901467841469214
# Document 1: Score = 0.08266363060309163
# Document 3: Score = 0.08266363060309163
# Document 2: Score = 0