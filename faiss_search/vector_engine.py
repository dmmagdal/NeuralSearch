# vector_engine
# Helper module to stream line searches and match them to the dataset.
# Tensorflow 2.7.0
# Windows/MacOS/Linux
# Python 3.7


import numpy as np


# Transforms query to vector using a pre-trained model and finds
# similar vectors using faiss.
# @param: query, (str) user query that should be more than a sentence
#	long.
# @param: model, model to encode the query text into vector embeddings.
# @param: index, (numpy.ndarray) faiss index that needs to be 
#	deserialized.
# @param: num_results, (int) number of results to return.
# @return: returns D (numpy.array of float) distance between results
#	and query and I (numpy.array of int) the Paper ID of the results. 
def vector_search(query, model, index, num_results=10):
	vector = model.encode(list(query))
	D, I = index.search(
		np.array(vector).astype("float32"), k=num_results
	)
	return D, I


# Returns the paper titles based on the paper index.
def id2details(df, I, column):
	return [list(df[df.id == idx][column]) for idx in I[0]]