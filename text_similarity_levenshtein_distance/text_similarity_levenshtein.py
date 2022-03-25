# text_similarity_levenshtein.py
# Build a plagiarism detection pipeline with by analyzing the
# similarity between texts using Levenshtein distance.
# Source: https://towardsdatascience.com/text-similarity-w-
# levenshtein-distance-in-python-2f7478986e75
# Windows/MacOS/Linux
# Python 3.7


import json
import string
import wikipedia
import numpy as np
import Levenshtein as lev
from functools import lru_cache
from nltk.corpus import stopwords


# Calculate the levenshtein distance between two input strings a and b.
# The code for this implementation uses dynamic programming but can be
# implemented through a brute force and iterative solution (be aware 
# that the brute force solution would not be optimal in terms of time
# complexity).
# @param: a (string), the first string you want to compare.
# @param: b (string), the second string you want to compare.
# @return: returns the distance between string a and b.
def lev_dist(a, b):
	@lru_cache(None) # For memorization.
	def min_dist(s1, s2):
		if s1 == len(a) or s2 == len(b):
			return len(a) - s1 + len(b) - s2

		# No change required.
		if a[s1] == b[s2]:
			return min_dist(s1 + 1, s2 + 1)

		return 1 + min(
			min_dist(s1, s2 + 1), 	# Insert character
			min_dist(s1 + 1, s2), 	# Delete character
			min_dist(s1 + 1, s2 + 1), 	# Replace character
		)


	return min_dist(0, 0)


def main():
	# Text similarity can be broken down into two components. semantic
	# similarity and lexical similarity. Given a pair of text, the 
	# semantic similarity of the pair refers to how close the documents
	# are in meaning, whereas lexical similarity is a measure of
	# overlap in vocabulary. If both documents in the pairs have the
	# same vocabularies, then they would have a lexical similarity of 1
	# vs 0 if there was no overlap. This example will focus on lexical
	# similarity.

	# Levenshtein Distance
	# Invented in 1965 by soviet mathematician Vladimir Levenshtein.
	# Levenshtein distance is very impactful because it does not
	# require two strings to be of equal length for them to be
	# compared. Intuitively speaking, Levenshtein distance is quite
	# easy to understand. "Informally, the Levenshtein distance between
	# two words is the minimum number of single characters edits
	# insertions, deletions, or substitutions) required to change one
	# word into the other". The larger output distance implies that 
	# more changes were necesary to make the two words equal to each
	# other and lower output distance implies that fewer changes were
	# necessary. Thus, a large value for Levenshtein distance implies
	# that two documents are not similar and a small value implies that
	# the two are similar. 

	# Problem statement
	# Similar to software like Turnitin, we want to build a pipeline 
	# which identifies if an input article is plagiarized.

	# Solution Architecture
	# To solve this problem, a few things will be required. Firstly, we
	# need to get the information passed on by the user of the
	# pipeline. For this we not only require the article that they want
	# to check the plagiarism against but also a keyword tag which
	# corresponds to the theme of the article. For the simplicity of
	# the example, we'll use the initial text written here with the tag
	# being "Levenshtein Distance". Second, we need a large corpus of
	# documents to compare the user input text with. We can leverage
	# the Wikipedia API to get access to Wikipedia articles associated
	# with the tage of the user input data. We can then clean the user
	# input document for redundancies like stopwords and punctuation to
	# better optimize the calculation of Levenshtein distance. We pass
	# this cleaned document through each document in the corpus under
	# the same tag as the user input document and identify if there is
	# any document which is very similar to the one the user submitted.
	# user input -> refernce corpus (wikipedia) -> cleaning -> distance
	# -> plagiarism check.

	# Fetch data
	user_article = '''
Identifying similarity between text is a common problem in NLP and is used by many companies world wide. The most common application of text similarity comes from the form of identifying plagiarized text. Educational facilities ranging from elementary school, high school, college and universities all around the world use services like Turnitin to ensure the work submitted by students is original and their own. Other applications of text similarity is commonly used by companies which have a similar structure to Stack Overflow or Stack Exchange. They want to be able to identify and flag duplicated questions so the user posting the question can be referenced to the original post with the solution. This increases the number of unique questions being asked on their platform. 
Text similarity can be broken down into two components, semantic similarity and lexical similarity. Given a pair of text, the semantic similarity of the pair refers to how close the documents are in meaning. Whereas, lexical similarity is a measure of overlap in vocabulary. If both documents in the pairs have the same vocabularies, then they would have a lexical similarity of 1 and vice versa of 0 if there was no overlap in vocabularies [2].
Achieving true semantic similarity is a very difficult and unsolved task in both NLP and Mathematics. It's a heavily researched area and a lot of the solutions proposed does involve a certain degree of lexical similarity in them. For the focuses of this article, I will not dive much deeper into semantic similarity, but focus a lot more on lexical similarity.
Levenshtein Distance
There are many ways to identify the lexical similarities between a pair of text, the one which we'll be covering today in this article is Levenshtein distance. An algorithm invented in 1965 by Vladimir Levenshtein, a Soviet mathematician [1].
Intuition
Levenshtein distance is very impactful because it does not require two strings to be of equal length for them to be compared. Intuitively speaking, Levenshtein distance is quite easy to understand.
Informally, the Levenshtein distance between two words is the minimum number of single-character edits (insertions, deletions or substitutions) required to change one word into the other. [1]
- https://en.wikipedia.org/wiki/Levenshtein_distance
Essentially implying that the output distance between the two is the cumulative sum of the single-character edits. The larger the output distance is implies that more changes were necessary to make the two words equal each other, and the lower the output distance is implies that fewer changes were necessary. For example, given a pair of words dream and dream the resulting Levenshtein distance would be 0 because the two words are the same. However, if the words were dream and steam the Levenshtein distance would be 2 as you would need to make 2 edits to change dr to st .
Thus a large value for Levenshtein distance implies that the two documents were not similar, and a small value for the distance implies that the two documents were similar.
'''

	# pass the text under the tag of "Levenshtein Distance". Fetch the
	# wikipedia page for levenshtein distance through the wikipedia
	# module.
	tags = ["Levenshtein Distance"]
	tag_content = fetch_wiki_data(tags)

	# Clean data
	# Clean the data associated to the user submitted and wikipedia
	# article. Do some simple data preprocessing on the text by
	# lowering the text, removing stopwords and punctuations.
	user_article = clean_text(user_article)
	for tag, content in tag_content.items():
		tag_content[tag] = clean_text(content)

	# Find similarity
	# Now, this value might seem relatively arbitrary to you, it's hard
	# to determine if this value reflects that the content is
	# plagiarized or not. The larger the value is the less likely it is
	# to be considered plagiarized based on our understanding of 
	# levenshtein distance. However, it's difficult to determine that
	# threshold of what distance is not large enough.
	distances = similarity(user_article, tag_content)

	# Check plagiarism
	# We will check for plagiarism through a simple formulaic approach.
	# first we get the maximum length between the user submitted
	# article and the content. Then we check if the levenshtein
	# distance is less than or equal to that value multiplied by some
	# threshold (here we use 0.4), then that user submitted article 
	# can be considered plagiarized.
	print(
		json.dumps(
			is_plagiarism(user_article, tag_content, distances),
			indent=4
		)
	)

	# Caveats
	# There are a number of caveats to using the pipeline above.
	# 1) This pipeline does not identify which areas are plagiarized
	#	and which areas are not. It only yields an overall score of
	#	plagiarism.
	# 2) The process does not account of properly cited and quoted
	#	pieces of text. This would misleadingly increase the overall
	#	plagiarism score.
	# 3) It's difficult to determine a threshold of how small or large
	#	a distance should be to be considered plagiarized.

	# Final thoughts
	# Levenshtein distance is a great measure to use to identify
	# lexical similarity between a pair of texts, but there are also
	# other well performing similarity measures as well. The Jaro-
	# Winkler score in particular comes ot mind and can be easily
	# implemented in this pipeline. Be aware that Jaro similarly 
	# outputs a result which is interpreted differently than
	# Levenshtein distance.

	# Exit the program.
	exit(0)


# Get the wikipedia data associated to a certain user input tag.
# @param: tags (string), the item you want to search wikipedia for.
# @return: returns the contents associated with the user specified 
#	tags.
def fetch_wiki_data(tags):
	content = {}
	for tag in tags:
		# Get the wikipedia data for the tag.
		wiki_tag = wikipedia.search(tag)

		# Get the page info.
		page = wikipedia.page(wiki_tag[0])

		# Get the page content.
		content[tag] = page.content
	return content


# Remove punctuation from the input text.
def remove_punctuation(txt, punct=string.punctuation):
	return "".join([c for c in txt if c not in punct])


# Remove the stopwords from the input text.
def remove_stopwords(txt, sw=list(stopwords.words("english"))):
	return " ".join([w for w in txt.split() if w.lower() not in sw])


# Clean the text being passed by removing specific line feed characters
# like "\n", "\r", and "\".
def clean_text(txt):
	txt = txt.replace("\n", " ").replace("\r", " ").replace("\"", "")
	txt = remove_punctuation(txt)
	txt = remove_stopwords(txt)
	return txt.lower()


# Identify the similarities between the user_article and all the 
# content within tag_content.
# @param: user_article (string), the text submitted by the user.
# @param: tag_content (dict), key is the tag and the value is the
#	content you want to compare with the user_article.
# @return: returns a dictionary holding the levenshtein associated to
#	the user_article with each tag_content.
def similarity(user_article, tag_content):
	distances = {}
	for tag, content in tag_content.items():
		dist = lev.distance(user_article, content)
		distances[tag] = dist
	return distances


# Identify if the user_article is considered plagiarized for each of
# the tag_content based on the distances observered.
# @param: user_article (string), the text submitted by the user.
# @param: tag_content (dict), key is the tag and the value is the 
#	content you want to compare with the user_article
# @param: distances (dict), key is the tag and the value is the
#	levenshtein distance.
# @param: tf (float), the plagiarism threshold.
# @return: returns a dictionary associated to the plagiarism percentage
#	for each tag.
def is_plagiarism(user_article, tag_content, distances, th=0.4):
	ua_len = len(user_article)
	distances = {
		tag: [d, max(ua_len, len(tag_content[tag]))]
		for tag, d in distances.items()
	}

	for tag, d in distances.items():
		if d[0] <= d[1] * th:
			distances[tag] = "Plagiarized"
		else:
			distances[tag] = "Not Plagiarized"
	return distances


if __name__ == '__main__':
	main()