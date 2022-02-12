# semantic_doc_search.py
# Build a semantic document search engine with TF-IDF (Term Frequency -
# Inverse Document Frequency) and Google's USE (Universal Sentence
# Encoder).
# Source: https://medium.com/analytics-vidhya/build-your-semantic-
# document-search-engine-with-tf-idf-and-google-use-c836bf5f27fb
# Source (Github): https://github.com/zayedrais/DocumentSearchEngine
# Tensorflow 2.7
# Python 3.7
# Windows/MacOS/Linux


import os
import re
import pickle
import string
import operator
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt


def main():
	# Data collection
	# Use the 20newsgroup open-source dataset for the analysis of a 
	# text search engine giving input keywords/sentence input. This
	# dataset is a collection of approximately 11K newsgroup documents,
	# partitioned (nearly) evenly across 20 different newsgroups.
	news = pd.read_json(
		'https://raw.githubusercontent.com/zayedrais/DocumentSearchEngine/master/data/newsgroups.json'
	)

	# Clean the data
	# Retrieve the subject of the document from the text.
	for i, txt in enumerate(news["content"]):
		subject = re.findall("Subject:(.*\n)", txt)
		if len(subject) != 0:
			news.loc[i, "Subject"] = str(i) + " " + subject[0]
		else:
			news.loc[i, "Subject"] = "NA"
	df_news = news[["Subject", "content"]]

	# Remove unwanted data from text content and the subject of a
	# dataset.
	df_news.content = df_news.content.replace(
		to_replace="from:(.*\n)", value="", regex=True
	) # remove from to email
	df_news.content = df_news.content.replace(
		to_replace="lines:(.*\n)", value="", regex=True
	)
	df_news.content = df_news.content.replace(
		to_replace="[!\"#$%&'()*+,/:;<=>?@[\\]^_`{!}~]", value="", 
		regex=True
	) # remove punctuation
	df_news.content = df_news.content.replace(
		to_replace="-", value="", regex=True
	) 
	df_news.content = df_news.content.replace(
		to_replace="\s+", value="", regex=True
	) # remove new line
	df_news.content = df_news.content.replace(
		to_replace="  ", value="", regex=True
	) # remove double white space
	df_news.content = df_news.content.apply(lambda x: x.strip()) # Ltrim and Rtrim of whitespace.

	df_news.Subject = df_news.Subject.replace(
		to_replace="Re:", value="", regex=True
	)
	df_news.Subject = df_news.Subject.replace(
		to_replace="[!\"#$%&'()*+,/:;<=>?@[\\]^_`{!}~]", value="", regex=True
	)
	df_news.Subject = df_news.Subject.replace(
		to_replace="\s+", value="", regex=True
	)
	df_news.Subject = df_news.Subject.replace(
		to_replace="  ", value="", regex=True
	)
	df_news.Subject = df_news.Subject.apply(lambda x: x.strip())

	# Data preprocessing
	# Have a look at the distribution of the data. Convert the text to
	# lowercase.
	df_news["content"] = [
		entry.lower() for entry in df_news["content"]
	]

	# Drop empty data.
	for i, sen in enumerate(df_news.content):
		if len(sen.strip()) == 0:
			df_news = df_news.drop(str(i), axis=0).reset_index()\
				.drop("index", axis=1)

	# Tokenize the sentence into words (aka word tokenization).
	df_news["Word tokenize"] = [
		word_tokenize(entry) for entry in df_news.content
	]

	# Stop words. Stop words are commonly occuring words which don't
	# give any additional value to the document vector. In fact,
	# removing these wil increase computation and space efficiency. The
	# nltk library has a method to download stopwords.
	print(stopwords.words("english"))


	# Word lemmatization. Lemmatization is a way to reduce the word to
	# root synonym of a word. Unlike stemming, lemmatization makes sure
	# that the reduced word is again a dictionary word (word present in
	# the same language). WordNetLemmatizer can be used to lemmatize
	# any word. Here, create a wordLemmatizer function to remove a
	# single character, stopwords and lemmatize the words. Note:
	# WordNetLemmatizer requires POS tags to understand if the word is
	# a noun or verb or adjective etc. By default, it is set to noun.
	def wordLemmatizer(data):
		tag_map = defaultdict(lambda: wn.NOUN)
		tag_map["J"] = wn.ADJ
		tag_map["V"] = wn.VERB
		tag_map["R"] = wn.ADV
		file_clean_k = pd.DataFrame()
		for index, entry in enumerate(data):
			# Declaring an empty list to store the words that follow
			# the rules for this step.
			final_words = []

			# Initialize WordNetLemmatizer().
			word_lemmatized = WordNetLemmatizer()

			# pos_tag function below will provide the "tag" i.e. if the
			# word is Noun (N) or Verb (V) or something else.
			for word, tag in pos_tag(entry):
				# The below condition is to check for stop words and
				# consider only alphabets.
				if len(word) > 1 and word not in stopwords.words("english"):
					word_final = word_lemmatized.lemmatize(
						word, tag_map[tag[0]]
					)
					final_words.append(word_final)

					# The final processed set of words for each
					# iteration will be stored in text_final.
					file_clean_k.loc[index, "Keyword_final"] = str(final_words)
					file_clean_k = file_clean_k.replace(
						to_replace="\[.", value="", regex=True
					)
					file_clean_k = file_clean_k.replace(
						to_replace="'", value="", regex=True
					)
					file_clean_k = file_clean_k.replace(
						to_replace=" ", value="", regex=True
					)
					file_clean_k = file_clean_k.replace(
						to_replace="\].", value="", regex=True
					)
		return file_clean_k


	# The above function took ~13 hours to check and lemmatize the
	# words of 11K documents of the 20newsgroup datasest. The below
	# JSON file contains the lemmatized words.
	# https://raw.githubusercontent.com/zayedrais/DocumentSearch
	# Engine/master/data/WordLemmatize20NewsGroup.json
	# df_clean = WordNetLemmatizer(df_news["Word tokenize"])
	df_clean = pd.read_json(
		"https://raw.githubusercontent.com/zayedrais/DocumentSearchEngine/master/data/WordLemmatize20NewsGroup.json"
	)
	print(df_clean.Clean_Keyword[0])
	df_news["Clean_Keyword"] = df_clean["Clean_Keyword"]

	# Document Search Engine
	# There are two approaches to understanding text analysis:
	# 1) Document search engine with TF-IDF.
	# 2) Document search engine with Google Universal Sentence Encoder.
	# Calculating ranking by using cosine similarity. It is the most
	# common metric used to calculate the similarity between document
	# text from input keywords/sentences. Mathematically, it measures
	# the cosine of the angle between two vectors projected in multi-
	# dimensional space. 

	# 1) Document search engine with TF-IDF
	# TF-IDF stands for "Term Frequency - Inverse Document Frequency".
	# This is a technique to calculate the weight of each word
	# signifies the importance of the word in the document and corpus.
	# This algorithm is mostly used for retrieval of information and
	# text mining field.
	# Term Frequency (TF). The number of times a word appears in a
	# document divided by the total number of words in the document.
	# Every document has its term frequency.
	# Inverse Document Frequency (IDF). The log of the number of
	# documents divided by the number of documents that contain the
	# word w. Inverse data frequency determines the weight of rare
	# words across all documents in the corpus.
	# Lastly, the TF-IDF is simply the TF multiplied by the IDF. Rather
	# than manually implement TF-IDF ourselves, we can use the class
	# provided by sklearn.

	# The following will create TF-IDF weight of the whole dataset.

	# Create vocabulary.
	vocabulary = set()
	for doc in df_news.Clean_Keyword:
		vocabulary.update(doc.split(','))
	vocabulary = list(vocabulary)

	# Initializing the TF-IDF model.
	tfidf = TfidfVectorizer(vocabulary=vocabulary)

	# Fit the TF-IDF model.
	tfidf.fit(df_news.Clean_Keyword)

	# Transform the TF-IDF model.
	tfidf_tran = tfidf.transform(df_news.Clean_Keyword)

	# Save/load trained TF-IDF model.
	os.makedirs("./TF-IDFModel", exist_ok=True)
	with open("./TF-IDFModel/tfidf.pkl", "wb") as handle:
		pickle.dump(tfidf_tran, handle)
	tfidf_tran = pickle.load(open("./TF-IDFModel/tfidf.pkl", "rb"))

	# Save/load vocabulary.
	with open("./TF-IDFModel/vocab_news20group.txt", "w") as file:
		file.write(str(vocabulary))
	with open("./TF-IDFModel/vocab_news20group.txt", "r") as file:
		vocabulary = eval(file.readline())

	# Create a function to generate a vector for the input query.
	def gen_vector_T(tokens):
		Q = np.zeros((len(vocabulary)))
		x = tfidf.transform(tokens)
		for tokens in tokens[0].split(','):
			try:
				ind = vocabulary.index(tokens)
				Q[ind] = x[0, tfidf.vocabulary_[token]]
			except:
				pass
		return Q


	# Cosine similarity function for the calculation.
	def cosine_sim(a, b):
		cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
		return cos_sim


	# Cosine similarity between document to query function.
	def cosine_similarity_T(k, query):
		preprocessed_query = re.sub("\W+", " ", query).strip()
		tokens = word_tokenize(str(preprocessed_query))
		q_df = pd.DataFrame(columns=["q_clean"])
		q_df.loc[0, "q_clean"] = tokens
		q_df["q_clean"] = wordLemmatizer(q_df.q_clean)
		d_cosines = []

		query_vector = gen_vector_T(q_df["q_clean"])
		for d in tfidf_tran.A:
			d_cosines.append(cosine_sim(query_vector, d))

		out = np.array(d_cosines).argsort()[-k:][::-1]
		d_cosines.sort()
		a = pd.DataFrame()
		for i, index in enumerate(out):
			a.loc[i, "index"] = str(index)
			a.loc[i, "Subject"] = df_news["Subject"][index]
		for j, simScore in enumerate(d_cosines[-k:][::-1]):
			a.loc[j, "Score"] = simScore
		return a


	# Testing the function.
	cosine_similarity_T(10, "computer science")

	# 2) Document search engine with Google Universal Sentence Encoder.
	# Introduction to Google USE. The pre-trained Universal Sentence
	# Encoder (https://ai.googleblog.com/2018/05/advances-in-semantic-
	# textual-similarity.html) is publicly available in Tensorflow-Hub.
	# It comes with two variations (i.e. one trained with Transformer
	# Encoder and the other trained with Deep Averaging Network (DAN)).
	# They are pre-trained on a large corpus and can be used in a
	# variety of tasks (sentiment analysis, classification, and so on).
	# The two have a trade-off of accuracy and computational resource
	# requirement. While the one with the Transformer Encoder has
	# higher accuracy, it is computationally more expensive. The one
	# with DAN encoding is computationally less expensive and with a
	# little lower accuracy. Here we are using the DAN universal
	# sentence encoder from this url: https://tfhub.dev/google/
	# universal-sentence-encoder/4
	# Both models take a word, sentence or paragraph as input and
	# output a 512 dimension vector.

	# Download the model from Tensorflow-Hub.
	model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"

	# Load the Google DAN Universal Sentence Encoder.
	model = hub.load(model_url)


	# Create function for using model training.
	def embed(input):
		return model(input)


	# Use case 1: Word semantic similarity.
	wordmessage = [
		"big data", "millions of data", "millions of records",
		"cloud computing", "aws", "azure", "saas", "bank", "account"
	]

	# Use case 2: Sentence semantic similarity.
	sentmessage = [
		"How old are you?", "What is your age?", "How are you?",
		"How you doing?"
	]

	# Use case 3: Word, sentence, and paragraph semantic similariy.
	word = "Cloud computing"
	sentence = "what is cloud computing"
	paragraph = (
		"Cloud computing is the latest generation technology with a "
		"high IT infrastructure that provides us a means by which we "
		"can use and utilize the applications as utilizes via the "
		"Internet. Cloud computing make IT infrastructure along with "
		"their services available 'on-need' basis. The cloud "
		"technology includes - a development platform, hard disk, "
		"computing power, software application, and database."
	)
	paragraph5 = (
		"Universal Sentence Encoder embeddings also support short "
		"paragraphs. There is no hard limit on how long the paragraph "
		" is. Roughly, the longer the more 'diluted' the embedding "
		"will be."
	)
	paragraph6 = (
		"Azure is a cloud computing platform which was launched by "
		"Microsoft in February 2010. It is an open and flexible cloud "
		"platform which helps in development, data storage, service "
		"hosting, and service management. The Azure tool hosts web "
		"applications over the internet with the help of Microsoft "
		"data centers."
	)
	case4Message = [word, sentence, paragraph, paragraph5, paragraph6]

	# Training the model. Here, we train the model at batch-wise 
	# because it takes a long time to generate the graph of the
	# dataset. 
	ls = []
	chunkwise = 1000
	le = len(df_news.content)
	for i in range(0, le, chunkwise):
		if (i + chunkwise > le):
			chunkwise = le
			ls.append(chunkwise)
		else:
			a = i + chunkwise
			ls.append(a)

	j = 0
	print("Training:")
	for i in ls:
		directory = "./googleUSEModel/TrainModel/" + str(i)
		if not os.path.exists(directory):
			os.makedirs(directory)
		print(i, j)
		m = embed(df_news.content[i:j])
		exported_m = tf.train.Checkpoint(v=tf.Variable(m))
		exported_m.f = tf.function(
			lambda x: exported_m.v * x,
			input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)]
		)
		tf.saved_model.save(exported_m, directory)
		j = i

	# Batch-wise load model.
	ar = []
	for i in ls:
		directory = "./googleUSEModel/TrainModel/" + str(i)
		if os.path.exists(directory):
			print(directory)
			imported_m = tf.saved_model.load(directory)
			a = imported_m.v.numpy()
			exec(f"load{i} = a")
			ar.append(a)

	# Concatenate the array from batch-wise loaded model.
	con_a = np.concatenate(ar)

	# Training the model for a single time.
	model_USE = embed(df_news.content[0:2500])

	# Save the model for reuse.
	exported = tf.train.Checkpoint(v=tf.Variable(model_USE))
	exported.f = tf.function(
		lambda x:  exported.v * x,
		input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)]
	)
	save_path = "./googleUSEModel/trained_model"
	tf.saved_model.save(exported, save_path)

	# Load model from path.
	imported = tf.saved_model.load(save_path)
	loaded_model = imported.v.numpy()


	# Function for document search:
	def searchDocument(query):
		q = [query]

		# Embed the query for calculating the similarity.
		Q_Train = embed(q)

		# Calculate the similarity.
		linear_similarities = linear_kernel(Q_Train, con_a).flatten()

		# Sort top 10 index with similarity score.
		top_index_doc = linear_similarities.argsort()[:-11:-1]

		# Sort by similarity score.
		linear_similarities.sort()
		a = pd.DataFrame()
		for i, index in enumerate(top_index_doc):
			a.loc[i, "index"] = str(index)
			a.loc[i, "File_Name"] = df_news["Subject"][index] # Read file name with index from file_data DF
		for j, simScore in enumerate(linear_similarities[:-11:-1]):
			a.loc[j, "Score"] = simScore
		return a


	# test the search:
	print(searchDocument("computer science"))

	# Conclusion
	# The Google USE model is providing the semantic search result
	# while TF-IDF model doesn't know the meaning of the word, just
	# giving the result based on words available in the documents.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()