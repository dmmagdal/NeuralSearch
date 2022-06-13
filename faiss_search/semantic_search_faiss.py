# semantic_search_faiss.py
# Build a semantic search engine with transformers (such as BERT) and
# faiss.
# Source: https://towardsdatascience.com/how-to-build-a-semantic-
# search-engine-with-transformers-and-faiss-dcbea307a0e8
# Source (GitHub): https://github.com/kstathou/vector_engine
# Tensorflow 2.7.0
# Windows/MacOS/Linux
# Python 3.7


import os
import json
import pickle
import faiss
import s3fs
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from vector_engine import vector_search, id2details


def main():
	# This example works with real-world data (a custom dataset created
	# by the author hosted on AWS S3) containing 8,430 academic
	# articles on misinformation, disinformation, and fake news
	# published between 2010 and 2020 by querying the Microsoft 
	# Academic Graph with Orion. The data contains the papers'
	# abstract, title, citations, publication year, and ID. Minimal
	# text cleaning/preprocessing has already been applied.

	# Pull the data from s3 if a local copy does not already exist.
	if not os.path.exists("misinformation_papers.csv"):
		df = pd.read_csv(
			's3://vector-search-blog/misinformation_papers.csv'
		)
		df.to_csv("misinformation_papers.csv")
	else:
		df = pd.read_csv("misinformation_papers.csv")

	# Next, encode the paper abstracts with a pre-trained model from
	# the Sentence Transformer library. The author has linked a
	# spreadsheet of these models here:
	# (https://docs.google.com/spreadsheets/d/
	# 14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0). In this
	# example, the author used the "distilbert-base-nli-stsb-mean-
	# tokens" model which performs well in semantic text similarity
	# tasks and is much faster than BERT (as it is considerably
	# smaller). Note that (text-embedding) models from tensorflow hub
	# can also be used here (Universal Sentence Encoder, BERT, etc)
	# though there would need to be additional processing (all input
	# would best be batched before entering the model). Sentence
	# Transformers does use Pytorch as part of its dependencies.

	# Instantiate the model.
	model = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")

	# Convert abstracts to embedding vectors (note that running the
	# model on a GPU will result in faster inference).
	embeddings = model.encode(df.abstract.tolist()) # use to_list() depending on pandas version

	# After that, we index the documents with Faiss. Faiss contains
	# algorithms that search in sets of vectors of any size, even ones
	# that do not fit in RAM. Faiss is built around the Index object
	# which contains (and sometimes preprocesses) the searchable
	# vectors. It handles collections of vectors of a fixed
	# dimensionality 'd', typically a few 10s to a few 100s. Faiss uses
	# only 32-bit floating point matrices. This means that the data
	# type of the input must be changed to match that before building
	# the index and as a result, the outputs would need to be stacked
	# back together; also the output of the model will have to be
	# converted from tensorflow tensors to numpy arrays). 
	
	# In this example, we use the IndexFlatL2 index that performs a
	# brute-force L2 distance search. It works well with this dataset,
	# however, it can be very slow with larger datasets as it scales
	# linearly with the number of indexed vectors. Faiss offers faster
	# indexes too (though those require some "training" before the
	# vector embeddings can be added).

	# Create the index
	# Step 1: Change data type to float32.
	embeddings = np.array(
		[embedding for embedding in embeddings]
	).astype("float32")

	# Step 2: Instantiate the index.
	index = faiss.IndexFlatL2(embeddings.shape[1])

	# Step 3: Pass the index to IndexIDMap.
	index = faiss.IndexIDMap(index)

	# Step 4: Add vectors and their IDs.
	index.add_with_ids(embeddings, df.id.values)

	# To test that the index works as expected, query it with an
	# indexed vector and retrieve its most similar documents as well as
	# their distance. The first result should be the query.

	# Retrieve the 10 nearest neighbors.
	D, I = index.search(np.array([embeddings[5415]]), k=10)

	# Query paper abstract and title.
	print("Query title:")
	print(df.iloc[5415, 0])
	print("Query abstract:")
	print(df.iloc[5415, 1])

	# Distances and indices returned by the search.
	print(f"L2 distance: {D.flatten().tolist()}")
	print(f"MAG paper IDs: {I.flatten().tolist()}")

	# Fetch paper titles based on the index.
	print(
		json.dumps(id2details(df, I, "original_title"), indent=4)
	)

	# We can try and also find relevant academic articles for new,
	# unseen search queries. This example query pulled from the first
	# paragraph of the "Can WhatsApp benefit from debunked fact-checked
	# stories to reduce misinformation?" article published on HKS
	# Misinformation Review. The process is similar to how we just
	# queried the index above and has already been abstracted/coded 
	# with the vector_search() function.

	# Original search query.
	query_search = '''
WhatsApp was alleged to have been widely used to spread misinformation and propaganda during the 2018 elections in Brazil and the 2019 elections in India. Due to the private encrypted nature of the messages on WhatsApp, it is hard to track the dissemination of misinformation at scale. In this work, using public WhatsApp data from Brazil and India, we observe that misinformation has been largely shared on WhatsApp public groups even after they were already fact-checked by popular fact-checking agencies. This represents a significant portion of misinformation spread in both Brazil and India in the groups analyzed. We posit that such misinformation content could be prevented if WhatsApp had a means to flag already fact-checked content. To this end, we propose an architecture that could be implemented by WhatsApp to counter such misinformation. Our proposal respects the current end-to-end encryption architecture on WhatsApp, thus protecting users’ privacy while providing an approach to detect the misinformation that benefits from fact-checking efforts.
'''
	print(f"Query search: {query_search}")

	# Query the index.
	D, I = vector_search([query_search], model, index, num_results=10)

	# Checking the paper title, most of the results look quite relevant
	# to the search query.
	print(
		json.dumps(id2details(df, I, "original_title"), indent=4)
	)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()