# bert_tfhub.py
# Code from tensorflow that is meant to demonstrate how to:
# -> Load BERT models from Tensorflow Hub that have been trained on
#	different tasks including MNLI, SQuAD, and PubMed.
# -> Use a matching preprocessing model to tokenize raw text and 
#	convert it to ids.
# -> Generate the pooled and sequence output from the token input ids
#	using the loaded model.
# -> Look at the semantic similarity of the pooled outputs of different
#	sentences.
# Note: For some reason, running this program on bare metal on Windows
# gives an error. This error does not occur when running on Docker/
# linux.
# Source: https://www.tensorflow.org/hub/tutorials/bert_experts
# Tensorflow 2.7
# Python 3.7
# Windows/MacOS/Linux


import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text # Imports TF ops for preprocessing.
from sklearn.metrics import pairwise


def main():
	# Model
	# Model links for the BERT model.
	# ["https://tfhub.dev/google/experts/bert/wiki_books/2", 
	# "https://tfhub.dev/google/experts/bert/wiki_books/mnli/2", 
	# "https://tfhub.dev/google/experts/bert/wiki_books/qnli/2", 
	# "https://tfhub.dev/google/experts/bert/wiki_books/qqp/2", 
	# "https://tfhub.dev/google/experts/bert/wiki_books/squad2/2", 
	# "https://tfhub.dev/google/experts/bert/wiki_books/sst2/2",  
	# "https://tfhub.dev/google/experts/bert/pubmed/2", 
	# "https://tfhub.dev/google/experts/bert/pubmed/squad2/2"]
	BERT_MODEL = "https://tfhub.dev/google/experts/bert/wiki_books/2"

	# Preprocessing must match the model, but all the above use the same.
	PREPROCESS_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

	# Sentences
	# Take some sentences from Wikipedia to run through model.
	sentences = [
		"Here We Go Then, You And I is a 1999 album by Norwegian pop artist Morten Abel. It was Abel's second CD as a solo artist.",
		"The album went straight to number one on the Norwegian album chart, and sold to double platinum.",
		"Among the singles released from the album were the songs \"Be My Lover\" and \"Hard To Stay Awake\".",
		"Riccardo Zegna is an Italian jazz musician.",
		"Rajko Maksimović is a composer, writer, and music pedagogue.",
		"One of the most significant Serbian composers of our time, Maksimović has been and remains active in creating works for different ensembles.",
		"Ceylon spinach is a common name for several plants and may refer to: Basella alba Talinum fruticosum",
		"A solar eclipse occurs when the Moon passes between Earth and the Sun, thereby totally or partly obscuring the image of the Sun for a viewer on Earth.",
		"A partial solar eclipse occurs in the polar regions of the Earth when the center of the Moon's shadow misses the Earth.",
	]

	# Run the model
	# Load the BERT model from TF-Hub, tokenize the sentences using the
	# matching preprocessing model from TF-Hub, then feed in the
	# tokenized sentences to the model. It is recommended to run the
	# program on GPU.
	preprocess = hub.load(PREPROCESS_MODEL)
	bert = hub.load(BERT_MODEL)
	inputs = preprocess(sentences)
	outputs = bert(inputs)

	print("Sentences:")
	print(sentences)

	print("\nBERT inputs:")
	print(inputs)

	print("\nPooled embeddings:")
	print(outputs["pooled_output"])

	print("\nPer token embeddings:")
	print(outputs["sequence_output"])


	'''
	# Semantic similarity
	# Take a look at the pooled_output embeddings of the sentences and
	# compare how similar they are across sentences.
	def plot_similarity(features, labels):
		"""Plot a similarity matrix of the embeddings."""
		cos_sim = pairwise.cosine_similarity(features)
		sns.set(font_scale=1.2)
		cbar_kws=dict(use_gridspec=False, location="left")
		g = sns.heatmap(
			cos_sim, xticklabels=labels, yticklabels=labels,
			vmin=0, vmax=1, cmap="Blues", cbar_kws=cbar_kws
		)
		g.tick_params(labelright=True, labelleft=False)
		g.set_yticklabels(labels, rotation=0)
		g.set_title("Semantic Textual Similarity")


	plot_similarity(outputs["pooled_output"], sentences)
	'''


	# Learn more
	# -> Find more BERT models on Tensorflow hub.
	# -> This program demonstrates simple inference with BERT, and can
	#	find a more advanced tutorial about fine-tuning BERT at
	#	tensorflow.org/official_models/fine_tuning_bert.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()