# Neural Search

Description: Create a semantic search engine with a neural network (i.e. BERT) whose knowledge base can be updated. This engine can later be used for downstream tasks in NLP such as Q&A, summarization, generation, and natural language understanding (NLU).

### bert_search
 - Status: WIP
 - Source: https://towardsdatascience.com/building-a-search-engine-with-bert-and-tensorflow-c6fdc0186c8a
 - Description: Use a pre-trained BERT model checkpoint to build a general-purpose text feature extractor, which will be applied to a task of nearest neighbor search.

### bert_tfhub
 - Status: Completed
 - Source: https://www.tensorflow.org/hub/tutorials/bert_experts
 - Description: Use a matching preprocessing model to tokenize raw text and convert it to ids, generate the pooled and sequence output from the token input ids using the loaded (BERT) model, and look at the semantic similarity of the pooled outputs of different sentences.

### download_tfds_datasets
 - Status: Completed
 - Source: N/A
 - Description: Not an example of neural networks for semantic search but instead is for downloading tensorflow datasets locally to specified locations. The Dockerfile is required for use on Windows machines (must also use Docker volumes to persist data from the container to the local machine). Running the download_data.py script on Linux or MacOS does not require said Dockerfile and can be run natively.

### finetune_bert
 - Status: Completed
 - Source: https://www.tensorflow.org/text/tutorials/fine_tune_bert
 - Description: Work through fine-tuning a BERT model using the tensorflow-models pip package. The pretrained BERT model is on Tensorflow Hub.

### finetune_bert4search
 - Status: Abandoned (No dataset used in example to reference)
 - Source: https://towardsdatascience.com/fine-tuning-a-bert-model-for-search-applications-33a7a442b9d0
 - Description: Fine-tuning a BERT model for search applications.

### text_summarization_encoderdecoder
 - Status: Abandoned (No source code to reference)
 - Source: https://towardsdatascience.com/text-summarization-from-scratch-using-encoder-decoder-network-with-attention-in-keras-5fa80d12710e
 - Description: Summarizing text from new articles to generate meaningful headlines using an Encoder-Decoder with Attention in Keras.

### tf-idf_from_scratch
 - Status: Completed
 - Source: https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089
 - Description: Implement the TF-IDF algorithm from scratch in python.

### tf-idf_use_search_engine
 - Status: Completed
 - Source: https://medium.com/analytics-vidhya/build-your-semantic-document-search-engine-with-tf-idf-and-google-use-c836bf5f27fb
 - Description: Build a semantic document search engine with TF-IDF and Google's USE. 
 - Note: Works on TF 2.7.0 bare metal and 2.4.0 Docker (apparently has some issues running bare metal on TF 2.4.0 on Windows laptop for some reason).


## Additional reads:
 - https://medium.com/@blogsupport/neural-information-retrieval-google-bert-6ce3cbabf7ff