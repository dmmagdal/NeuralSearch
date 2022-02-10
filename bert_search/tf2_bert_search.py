# tf2_bert_search.py
# Using the code from the medium article to implement a BERT search
# engine (L2 Retrieveer where L2 refers to the L2 or euclidean
# distance).
# Tensorflow 2.7
# Python 3.7
# Windows/MacOS/Linux


import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_text as text
import nltk
from nltk.corpus import reuters


def main():
	BERT_PREPROCESSOR = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1" # this is maked as a text preprocessing model
	BERT_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3" # this is marked as a text embedding model
	PREPROCESS_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3" # this is marked as a text preprocessing model

	preprocessor1 = hub.load(BERT_PREPROCESSOR)
	preprocessor2 = hub.load(PREPROCESS_MODEL)
	encoder = hub.load(BERT_MODEL)

	nltk.download("reuters")
	nltk.download("punkt")

	max_samples = 256
	categories = [
		"wheat", "tea", "strategic-metal", "housing", "money-supply",
		"fuel"
	]

	# Get the article embeddings.
	X, Y, S = [], [], []
	for category in categories:
		sents = reuters.sents(categories=category)
		sents = [" ".join(sent) for sent in sents][:max_samples]
		# X.append(bert_vectorizer(sents, verbose=True))
		# Y += [category] * len(sents)
		S += sents
	print(S)
	print(len(S))

	input_sample = S[0]

	# BERT with model BERT_PREPROCESSOR layer
	text_input = keras.layers.Input(shape=(), dtype=tf.string)
	preprocessor1_layer = hub.KerasLayer(BERT_PREPROCESSOR)
	encoder_inputs = preprocessor1_layer(text_input)
	encoder = hub.KerasLayer(BERT_MODEL, trainable=True)
	outputs = encoder(encoder_inputs)
	pooled_output = outputs["pooled_output"] # [batch_size, 768] (represent each input sequence as a whole)
	sequence_output = outputs["sequence_output"] # [batch_size, seq_length, 768] (represent each input token in context)
	model_1 = keras.Model(text_input, pooled_output, name="Bert1")
	model_1.summary()

	print(S[:3])
	exit()
	sentences = tf.constant(["Hello there"])
	print(model_1(sentences))


	exit(0)
	print(type(preprocessor))
	preprocessor(["Hello there"])
	exit(0)
	input_sample = S[:3]
	print(input_sample)
	print(tf.convert_to_tensor(input_sample))
	processed_input = preprocessor(tf.convert_to_tensor(input_sample))
	output = encoder(processed_input)
	print(type(output))
	print(output)

	# First load BERT.
	# Use 

	# Take knowledgebase and vectorize it.

	# Initialize retriever with knowledgebase.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()