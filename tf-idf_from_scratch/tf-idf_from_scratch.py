# tf-ifd_from_scratch.py
# Implement the TF-IDF algorithm from scratch in python.
# Source: https://towardsdatascience.com/tf-idf-for-document-ranking-
# from-scratch-in-python-on-real-world-dataset-796d339a4089
# Source (github): https://github.com/williamscott701/Information-
# Retrieval
# Source (dataset): http://archives.textfiles.com/stories.zip
# Windows/MacOS/Linux
# Python 3.7


import os
import re
import math
import nltk
import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from num2words import num2words


def main():
	# Introduction
	# TF-IDF stands for "Term Frequency - Inverse Document Frequency".
	# This is a technique to quantify words in a set of documents. We
	# generally compute a score for each word to signify its importance
	# in the document and corpus. This method is a qidely used
	# technique in Information Retrieval and Text Mining.
	# Given a sentence (eg "this building is so tall"), we can easily
	# understand the sentence as we know the semantics of the words and
	# the sentence. It would be easier for a programming language to
	# understand textual data in the form of numerical values. For this
	# reason, we need to vectorize all of the text so that it is better
	# represented.
	# By vectorizing the documents, we can further perform multiple
	# tasks such as finding the relevant documents, ranking,
	# clustering, etc. This exact technique is used when you perform a
	# google search (now updated to newer transformer techniques). The
	# web pages are called documents and the search text with which
	# your search is called a query. The search engine maintains a
	# fixed representation of all the documents. When you search with a
	# query, the search engine will find the relevance of the query
	# with all of the documents, ranks them in the order of relevance,
	# and shows you the top k documents. All of this process is done
	# using the vectorized form of query and documents.
	# Now coming back to TF-IDF. TF-IDF stands for Term Frequency (TF) *
	# Inverse Document Frequency (IDF). 

	# Terminology
	# t -> term (word), d -> document (set of words), N -> count of
	# corpus, corpus -> the total document set.

	# Term Frequency
	# This measures the frequency of a word in a document. This highly
	# depends on the length of the document and the generality of the
	# word. For example, a very common word such as "was" can appear
	# multiple times in a document. But if we take two documents with
	# 100 words and 10,000 words respectively, there is a high
	# probability that the word "was" is present more in the 10,000
	# word document. But we cannot say that the longer document is more
	# important than the shorter document. For this extact reason, we
	# perform normalization on the frequency value, dividing the
	# frequency with the total number of words in the document.
	# Recall that we need to finally vectorize the document. When we
	# plan to vectorize documents, we cannot just consider the words
	# that are present in that particular document. If we do that, then
	# the vector length will be different for both the documents, and
	# it will not be feasible to compute the similarity. So, what we do
	# is we vectorize the documents on the vocab. Vocab is the list of
	# all possible words in the corpus.
	# We need the word counts of all the vocab words and the length of
	# the document to compute the TF. In case the term doesn't exist in
	# a particular document, that particular TF value will be 0 for
	# that particular document. In an extremem case, if all the words
	# in the document are the same, then TF will be 1. The final value
	# of the normalized TF value will be in the range of [0, 1].
	# TF is individual to each document and word, hence we can
	# formulate TF as follows: tf(t, d) = count of t in d / number of
	# words in d.
	# If we already computed the TF value and if this produces a
	# vectorized form of the document, why not just use TF to find the
	# relevance between documents? Why do we need IDF?
	# Words which are most common such as "is" and "are" will have very
	# high values, given those words very high importance. But using
	# these words to compute the relevance produces bad results. These
	# kinds of common words are called stop-words. Although we will
	# remove the stop words later in the pre-processing step, finding
	# the presence of the word across the documents and somehow reduce
	# their weightage is more ideal.

	# Document Frequency
	# This measures the importance of documents in a whole set of the
	# corpus. This is very similar to TF but the only difference is
	# that TF is the frequency counter for a term t in document d,
	# whereas DF is the count of occurrences of term t in the document
	# set N. In other words, DF is the number of documents in which the
	# word is present. We consider one occurrence if the term is
	# present in the document at least once, we do not need to know the
	# number of times the term is present.
	# df(t) = occurrence of t in N documents
	# To keep this also in a range, we normalize by dividing by the
	# total number of documents. our main goal is to know the
	# informativeness of a term, and DF is the exact inverse of it,
	# that is why we inverse DF.

	# Inverse Document Frequency
	# IDF is the inverse of the document frequency which measures the
	# informativeness of term t. When we calculate IDF, it will be very
	# low for the most occurring words such as stop words (because they
	# are present in almost all of the documents, and N/df will give a
	# very low value to that word). This finally gives what we want, a
	# relative weightage.
	# idf(t) = N/df
	# Now there are a few other problems with the IDF. When we have a 
	# large corpus size of say N = 10000, the IDF value explodes. So to
	# dampen the effect we take the log of IDF.
	# At query time, when the word is not present in the vocab, it will
	# simply be ignored. But in a few cases, we use a fixed vocab and a
	# few words of the vocab might be absent in the document, in such
	# cases, the df will be 0. As we cannot divide by 0, we smoothen
	# the value by adding 1 to the denominator.
	# idf(t) = log(N/(df + 1))
	# Finally, by taking a multiplicative value of TF and IDF, we get
	# the TF-IDF score. There are many different variations of TF-IDF
	# but for now, let us concentrate on this basic version.
	# tf-idf(t, d) = tf(t, d) * log(N/(df + 1))

	# Implementing on a real-world dataset
	# Now that we learned what TF-IDF is, let us compute the similarity
	# score on a dataset.
	# The dataset we are going to use are archives of a few stories.
	# This dataset has lots of documents in different formats. Download
	# the dataset (see link in source).

	# Step 1: Analyzing the Dataset
	# The first step in any machine learning task is to analyze the
	# data. So if we look at the dataset, at first glance, we see all
	# the documents with words in English. Each document has different
	# names and there are two folders in it.
	# Now one of the important tasks is to identify the title in the
	# body. If we analyze the documents, there are different patterns
	# of alignment of title. But most of the titles are center aligned.
	# Now we need to figure out a way to extract the title. But before
	# we get all pumped up and start coding, let us analyze the dataset
	# a little more.
	# Upon further inspection, we notice that there's an index.html in
	# each folder (including the root), which contains all the
	# document names and their titles. So, let us consider ourselves
	# lucky as the titles are given to us, without exhaustively
	# extracting titles from each document.

	# Step 2: Extracting title & body
	# There is no specific way to do this. This totally depends on the
	# problem statement at hand and on the analysis we do on the
	# dataset.
	# As we have already found that the titles and the document names
	# are in the index.html, we need to extract those names and titles.
	# We are lucky that index.html has tags that we can use as patterns
	# to extract our required content.
	# Before we start extracting the titles and file names, as we have
	# different folders, let's first crawl the folders to later read
	# all the index.html files at once.
	folders = [x[0] for x in os.walk(str(os.getcwd()) + "/stories/")]
	
	# os.walk gives us the files in the directory, os.getcwd gives us
	# the current directory and title we are going to search in the
	# current directory + stories folder as our data files are in the
	# stories folder. Assuming that you are dealing with a huge
	# dataset (usually the case), this helps automate the code. Now we
	# can find that folders give extra "/" for the root folder, so we
	# are going to remove it.
	folders[0] = folders[0][:len(folders[0]) - 1]

	# The above code removes the last character for the 0th index in
	# folders, which is the root folder.
	# Now, let's crawl through all the index.html to extract their
	# titles. To do that we need to find a pattern to take out the
	# title. As this is html, our job is a little simpler. We can 
	# clearly observe that each file name is enclosed between
	# (><A HERF=") and (") and each title is between (<BR><TD>) and
	# (\n).
	# We will use simple regular expressions to retrieve the name and
	# tile. the following code gives the list of all the values that
	# match that pattern. So names and titles variables have the list
	# of all names and titles.
	#names = re.findall('><A HREF="(.*)">', text)
	#titles = re.findall('<BR><TF> (.*)\n', text)

	# Now that we have code to retrieve the values from the index, we
	# just need to iterate to all the folders and get the title and
	# file from all the index.html files.
	# > read the file from index files
	# > extract title and names
	# > iterate to next folder
	# This prepares the indexes of the dataset, which is a tuple of the
	# file and its title. There is a small issue, the root folder
	# index.html also has folders and its links, we need to remove
	# those. Simply use a conditional check to remove it.
	dataset = []
	c = False
	for i in folders:
		file = open(i + "/index.html", "r")
		text = file.read().strip()
		file.close()

		file_name = re.findall('><A HREF="(.*)">', text)
		file_title = re.findall('<BR><TD> (.*)\n', text)
		
		if c == False:
			file_name = file_name[2:]
			c = True
		print(len(file_name), len(file_title))

		for j in range(len(file_name)):
			dataset.append(
				(str(i) + "/" + str(file_name[j]), file_title[j])
			)

	print(dataset)
	print(len(dataset))
	N = len(dataset)


	# Simple utility funtion.
	def print_doc(id):
		print(dataset[id])
		file = open(dataset[id][0], "r", encoding="cp1250")
		text = file.read().strip()
		file.close()
		print(text)

	# Step 3: Preprocessing
	# Preprocessing is one of the major steps when we are dealing with
	# any kind of text model. During this stage, we have to look at the
	# distribution of our data, what techniques are needed, and how
	# deep we should clean.
	# This step never has a one-hot rule and totally depends on the
	# problem statement. Few mandatory preprocessing steps are:
	# converting to lowercase, removing punctuation, removing stop
	# words, and lemmatization/stemming. in our problem statement, it
	# seems like the basic preprocessing steps will be sufficient.


	# Lowercase
	# During the text processing, each sentence is split into words and
	# each word is considered as a token after preprocessing.
	# Programming languages consider textual data as sensitive, which
	# means that "The" is different from "the". We humans know that
	# both those belong to the same token but due to the character
	# encoding those are considered as different tokens. Converting to
	# lowercase is a mandatory preprocessing step. As we have all our
	# data in the list, numpy has a method that can convert the list of
	# lists to lowercase at once.
	def lowercase(data):
		return np.char.lower(data)


	# Stop words
	# Stop words are the most commonly occuring words that dont give
	# any additional value to the document vector. In fact, removing
	# these will increase the computation and space efficiency. NLTK
	# library has a method to download the stopwords, so instead of
	# explicitly mentioning all the stopwords ourselves we can just
	# use the nltk library and iterate over all the words and remove
	# the stop words. There are many efficient ways to do this, but
	# I'll just give a simple method. We are going to iterate over all
	# the stop words and not append them to the list if it's a stop
	# word.
	def remove_stopwords(data):
		stop_words = stopwords.words("english")
		words = word_tokenize(str(data))
		new_text = ""
		for w in words:
			if w not in stop_words and len(w) > 1:
				new_text = new_text + " " + w
		return new_text


	# Punctuation
	# Punctuation is the set of unnecessary symbols that are in our
	# corpus of documents. We should be a little careful with what we
	# are doing with this, theremight be a few problems such as U.S. ->
	# "United States" being converted to "us" after the preprocessing.
	# That and hyphens should usually be dealt with a little care. But
	# for this problem statement, we are just going to remove these.
	# We are going to store all our symbols in a variable and iterate
	# that variable removing that particular symbol in the whole
	# dataset. We are using numpy here because our dataset is stored in
	# a list of lists, and numpy is our best bet.
	def remove_punctuation(data):
		symbols = "|!"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
		for i in symbols:
			data = np.char.replace(data, i, " ")
			data = np.char.replace(data, "  ", " ")
		data = np.char.replace(data, ",", "")
		return data


	# Note that ther is no open apostrophe in the punctuation symbols.
	# Because when we remove punctuation first it will convert don't to
	# dont, and it is a stop word that wont be removed. What we will do
	# instead, is removing the stop words first followed by symbols and
	# then finally repeat stopword removal as few words might still
	# have an apostrophe that are not stopwords.
	def remove_apostrophe(data):
		return np.char.replace(data, "'", "")


	# Single characters
	# Single characters are not much useful in knowing the importance
	# of the document and few final single characters might be
	# irrelevant symbols, so it is always good to remove the single
	# characters.
	# We just need to iterate to all the words and not append the word
	# if the length is not greater than 1.
	# def remove_single_characters(data):
	# 	new_text = ""
	# 	for w in words:
	# 		if len(w) > 1:
	# 			new_text = new_text + " " + w
	# 	return data


	# Stemming
	# This is the final and most important part of the preprocessing.
	# Stemming converts words to their stem.
	# For example, "playing" and "played" are the same type of words
	# that basically indicated an action "play" Stemmer does exactly
	# this, it reduces the word to its stem. We are going to use a
	# library called porter-stemmer which is a rule-based stemmer.
	# Porter-Stemmer identifies and removes the suffix or affix of a
	# word. The words given by the stemmer need not be a meaningful few
	# times, but it will be identified as a single token for the model.
	def stemming(data):
		stemmer = PorterStemmer()
		tokens = word_tokenize(str(data))
		new_text = ""
		for w in tokens:
			new_text = new_text + " " + stemmer.stem(w)
		return new_text


	# Lemmatization
	# Lemmatization is a way to reduce the word to the root synonym of
	# a word. Unlike Stemming, Lemmatization makes sure that the
	# reduced word is again a dictionary word (word present in the same
	# language). WordNetLemmatizer can be used to lemmatize any word.

	# Stemming vs lemmatization
	# stemming - need not be a dictionary word, removes prefix and
	# affix based on few rules.
	# lemmatization - will be a dictionary word. Reduces to a root
	# synonym.
	# Word 		Lemmatization 	Stemming
	# was 		be 				wa
	# studies 	study 			studi
	# studying 	study 			study
	# A better efficient way to proceed is to first lemmatise and then
	# stem, but stemming alone is also fine for a few problem
	# statements, here we will not lemmatise.

	# Converting numbers
	# When a user gives a query such as "100 dollars" or "hundred
	# dollars". For the user, both those search terms are the same.
	# But, our IR model treats them separately, as we are storing 100,
	# dollars, hundred as different tokens. So to make our IR model a
	# little better, we need to convert 100 to hundred. To achieve this
	# we are going to use a library called num2word.
	# num2words(100500) => "on hundred thousand, five hundred"
	# If we look a little close to the above output, it is giving us a
	# few symbols and sentences such as "one hundred *and* two", but we
	# just cleaned our data, then how do we handle this? We will just
	# run the punctuation and stop words again after converting numbers
	# to words.
	def convert_numbers(data):
		tokens = word_tokenize(str(data))
		new_text = ""
		for w in tokens:
			try:
				w =num2words(int(w))
			except:
				a = 0
			new_text = new_text + " " + w
		new_text = np.char.replace(new_text, "-", " ")
		return new_text


	# Preprocessing
	# Finally, we are going to put in all those preprocessing methods
	# above in another method and we will call that preprocess method.
	# If you look closely, a few of the preprocessing methods are
	# repeated again. As discussed, this just helps clean the data a
	# little deep. Now we need to read the documents and store their
	# title and the body separately as we are going to use them later.
	# In our problem statement, we have very different types of
	# documents, this can cause few erros in reading the documents due
	# to encoding compatibility. To resolve this, just use encoding =
	# "utf-8", errors = "ignore" in the open() method.
	def preprocess(data):
		data = lowercase(data)
		data = remove_punctuation(data) # remove comma separately
		data = remove_apostrophe(data)
		data = remove_stopwords(data)
		data = convert_numbers(data)
		data = stemming(data)
		data = remove_punctuation(data)
		data = convert_numbers(data)
		data = stemming(data) # needed again as we need to stem the words
		data = remove_punctuation(data) # needed again as num2word is giving few hyphens
		data = remove_stopwords(data) # needed again as num2word is giving stop words
		return data


	# Step 4: Calculating TF-IDF
	# Recall that we need to give different weights to title and body.
	# Now how are we going to handle that issue? How will the
	# calculation of TF-IDF work in this case?
	# Giving different weights to title and body is a very common
	# approach. We just need to consider the document as body + title,
	# using this we can find the vocab. And we need to give different
	# weights to words in the title and different weights to the words
	# in the body. To better explain this, let us consider an example:
	# title = "This is a novel paper"
	# body = "This paper consists of survey of many papers"
	# Now, we need to calculate the TF-IDF for body and for the title.
	# For the time being let us consider only the word "paper", and
	# forget about removing stop words.
	# What is the TF of the word "paper" in the title? 1/4?
	# No, it's 3/13. How? The word "paper" appears in title and body 3
	# times and the total number of words in title and body is 13. As
	# mentioned before, we just *consider* the word in the title to
	# have different weights, but still, we consider the whole document
	# when calculating TF-IDF.
	# Then the TF of "paper" in both title and body is the same? Yes,
	# it's the same! it's just the difference in weights that we are
	# going to give. If the word is present in both title and body,
	# then there wouldn't be any reduction in the TF-IDF value. If the
	# word is present only in the title, then the weight of the body
	# for that particular word will not add to the TF of that word, and
	# vice versa.
	# document = body + text
	# TF-IDF(document) = TF-IDF(title) * alpha + TF-IDF(body) * (1 - alpha)
	processed_text = []
	processed_title = []
	for i in dataset[:N]:
		file = open(i[0], "r", encoding='utf8', errors='ignore')
		text = file.read().strip()
		file.close()

		processed_text.append(word_tokenize(str(preprocess(text))))
		processed_title.append(word_tokenize(str(preprocess(i[1]))))


	# Calculating DF
	# Let us be smart and calculate DF beforehand. We need to iterate
	# through all the words in all the documents and store the document
	# id's for each word. For this, we will use a dictionary as we can
	# use the word as the key and a set of documents as the value. I
	# mentioned set because, even if we are trying to add the document
	# multiple times, a set will not just take duplicate values.
	# We are going to create a set if the word doesnt have a set yet,
	# else add it to the set. Ths condition is checked by the try
	# block. Here processed_text is the body of the document, and we
	# are going to repeat the same for the title as well, as we need to
	# consider the DF of the whole document.
	# len(DF) will give the unique words
	# DF will have the word as the key and the list of doc id's as the
	# value. But for DF we dont actually need the list of docs, we just
	# need the count. So we are going to replace the list with its
	# count.
	DF = {}
	for i in range(N):
		tokens = processed_text[i]
		for w in tokens:
			try:
				DF[w].add(i)
			except:
				DF[w] = {i}

		tokens = processed_title[i]
		for w in tokens:
			try:
				DF[w].add(i)
			except:
				DF[w] = {i}

	for i in DF:
		DF[i] = len(DF[i])

	# There we have it, the count we need for all the words. To find
	# the total unique words in out vocabulary, we need to take all the
	# keys of DF.
	total_vocab = [x for x in DF]
	total_vocab_size = len(DF)
	print(total_vocab_size)
	print(total_vocab[:20])


	# Calculating TF-IDF
	# Recall that we need to maintain different weights for title and
	# body. To calculate TF-IDF of body or title we need to consider
	# both the title and body. To make our job a little easier, let's
	# use a dictionary with (document, token) pair as key and any
	# TF-IDF score as the value. We just need to iterate over all the
	# documents. We can use the Counter which can give us the frequency
	# of the tokens, calculate TF and IDF and finally store as a
	# (document, token) pair in tf_idf. The tf_idf dictionary is for
	# the body, we will use the same logic to build a dictionary
	# tf_idf_title for the words in the title.
	def doc_freq(word):
		c = 0
		try:
			c = DF[word]
		except:
			pass
		return c


	doc = 0
	tf_idf = {}
	for i in range(N):
		tokens = processed_text[i]
		counter = Counter(tokens + processed_title[i])
		words_count = len(tokens + processed_title[i])

		for token in np.unique(tokens):
			tf = counter[token] / words_count
			df = doc_freq(token)
			idf = np.log((N + 1) / (df + 1))

			tf_idf[doc, token] = tf * idf

		doc += 1

	doc = 0
	tf_idf_title = {}
	for i in range(N):
		tokens = processed_title[i]
		counter = Counter(tokens + processed_text[i])
		words_count = len(tokens + processed_text[i])

		for token in np.unique(tokens):
			tf = counter[token] / words_count
			df = doc_freq(token)
			idf = np.log((N + 1) / (df + 1))

			tf_idf_title[doc, token] = tf * idf

		doc += 1

	print(tf_idf[(0, "go")])
	print(tf_idf_title[(0, "go")])

	# Coming to the calculation of different weights. Firstly, we need
	# to maintain a value alpha, which is the weight for the body, then
	# obviously 1 - alpha will be the weight for the title. Now let us
	# dive into a little math, we discussed that the TF-IDF value of a
	# word will be the same for both body and title if the word is 
	# present in both places. We will maintian two different tf-idf
	# dictionaries, one for the body and the other for the title.
	# What we are going to do is calculated the TF-IDF for the body,
	# multiply the whole body TF-IDF values with alpha, iterate the
	# tokens in the title, replace the title TF-IDF value in the body 
	# of the (document, token) pair exists.
	# Flow:
	# -> calculate TF-IDF for Body for all docs
	# -> calculate TF-IDF for Title for all docs
	# -> multiply the Body TF-IDF with alpha
	# -> iterate title TF-IDF for every (document, token)
	# -- If token is in body, replace the Body(document, token) value
	#	in Title(document, token).
	# TF-IDF = body_tf-idf * body_weight + title_tf-idf * title_weight
	# body_weight + title_weight = 1
	# When a token is in both places, then the final TF-IDF will be the
	# same as taking either body or title tf_idf. That is exactly what
	# we are doing in the above flow. So, finally, we have a dictionary
	# tf_idf which has the values as a (document, token) pair.
	alpha = 0.3
	for i in tf_idf:
		tf_idf[i] *= alpha
	for i in tf_idf_title:
		tf_idf[i] = tf_idf_title[i]

	print(len(tf_idf))


	# Step 5: Ranking using Matching Score
	# matching score is the simplest wait to calculate the similarity.
	# In this method, we add tf_idf values of the tokens that are in
	# query for every document. For example, for the query "hello
	# world", we need to check in every document if these words exist
	# and if the word exists, then the tf_idf value is added to the
	# matchig score of that particular doc_id. In the end, we will sort
	# and take the top k documents.
	# Mentioned above is the theoretical concept, but as we are using a
	# dictionary to hold our dataset, what we are going to do is we
	# will iterate over all of the values in the dictionary and check
	# if the value is present in the token. As our dictionary is a 
	# (document, token) key, when we find a token that is in the query
	# we will add the document id to another dictionary along with the
	# tf-idf value. Finally, we will just take the top k documents
	# again.
	def matching_score(k, query):
		preprocessed_query = preprocess(query)
		tokens = word_tokenize(str(preprocessed_query))

		print("Matching Score")
		print("\nQuery:", query)
		print("")
		print(tokens)

		query_weights = {}

		for key in tf_idf:
			if key[1] in tokens:
				try:
					query_weights[key[0]] += tf_idf[key] # key[0] is the document id, key[1] is the token
				except:
					query_weights[key[0]] = tf_idf[key]

		query_weights = sorted(
			query_weights.items(), key=lambda x: x[1], reverse=True
		)
		print("")

		l = []
		for i in query_weights[:10]:
			l.append(i[0])

		print(l)


	query_string = "Without the drive of Rebeccah's insistence," +\
		" Kate lost her momentum. She stood next a slatted oak" +\
		" bench, canisters still clutched, surveying"
	matching_score(10, query_string)


	# Step 6: Ranking using Cosine Similarity
	# When we have a perfectly working Matching Score, why do we need
	# cosine similarity again? Though Matching Score gives relevant
	# documents, it fails when we give long queries and will not be
	# able to rank them properly. What cosine similarity does is that
	# it will mark all the documents as vectors of tf-idf tokens and
	# measures the similarity in cosine space (the angle between the
	# vectors). Few times the query length would be small but it might
	# be closely related to the document in such cases cosine
	# similarity is the best to find relevance.
	def cosine_sim(a, b):
		cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
		return cos_sim


	# Vectorization
	# To compute any of the above, the simplest way is to convert
	# everything to a vector and then compute the cosine similarity.
	# So, let's convert the query and document to vectors. We are going
	# to use total_vocab variable which has all the list of unique
	# tokens to generate an index for each token, and we will use numpy
	# of shape (documents, total_vocab) to store the document vectors.
	# For the vector, we need to calculate the TF-IDF values, TF we can
	# calculate from the query itself, and we can make use of DF that
	# we created for the document frequency. Finally, we will store it
	# in a (1, vocab_size) numpy array to store the tf-idf values,
	# index of the token will be decided from the total_vocab list.
	# Now,all we have to do is calculate the cosine similarity for all
	# the documents and return the maximum k documents. Cosine
	# similarity is defined as follows:
	# np.dot(a, b) / (norm(a) * norm(b))
	D = np.zeros((N, total_vocab_size))
	for i in tf_idf:
		try:
			ind = total_vocab.index(i[1])
			D[i[0]][ind] = tf_idf[i]
		except:
			pass


	def gen_vector(tokens):
		Q = np.zeros((len(total_vocab)))

		counter = Counter(tokens)
		words_count = len(tokens)

		query_weights = {}

		for token in np.unique(tokens):
			tf = counter[token] / words_count
			df = doc_freq(token)
			idf = math.log((N + 1) / (df + 1))

			try: 
				ind = total_vocab.index(token)
				Q[ind] = tf * idf
			except:
				pass

		return Q


	def cosine_similarity(k, query):
		print("Cosine Similarity")
		preprocessed_query = preprocess(query)
		tokens = word_tokenize(str(preprocessed_query))

		print("\nQuery:", query)
		print("")
		print(tokens)

		d_cosines = []
		query_vector = gen_vector(tokens)
		for d in D:
			d_cosines.append(cosine_sim(query_vector, d))

		out = np.array(d_cosines).argsort()[-k:][::-1]
		print("")
		print(out)


	Q = cosine_similarity(10, query_string)

	print_doc(200)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()