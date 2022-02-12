# Neural Search

Description: Create a semantic search engine with a neural network (i.e. BERT) whose knowledge base can be updated. This engine can later be used for downstream tasks in NLP such as Q&A, summarization, generation, and natural language understanding (NLU).

### bert_search
 - Status: WIP
 - Source: https://towardsdatascience.com/building-a-search-engine-with-bert-and-tensorflow-c6fdc0186c8a
 - Description: Use a pre-trained BERT model checkpoint to build a general-purpose text feature extractor, which will be applied to a task of nearest neighbor search.

### bert_tfhub
 - Status: Completed
 - Source: Use a matching preprocessing model to tokenize raw text and convert it to ids, generate the pooled and sequence output from the token input ids using the loaded (BERT) model, and look at the semantic similarity of the pooled outputs of different sentences.
 - Description: https://www.tensorflow.org/hub/tutorials/bert_experts

### finetune_bert
 - Status: Completed
 - Source: https://www.tensorflow.org/text/tutorials/fine_tune_bert
 - Description: Work through fine-tuning a BERT model using the tensorflow-models pip package. The pretrained BERT model is on Tensorflow Hub.

### text_summarization_encoderdecoder
 - Status: Abandoned (No source code to reference)
 - Source: https://towardsdatascience.com/text-summarization-from-scratch-using-encoder-decoder-network-with-attention-in-keras-5fa80d12710e
 - Description: Summarizing text from new articles to generate meaningful headlines using an Encoder-Decoder with Attention in Keras.

### tf-idf_use_search_engine
 - Status WIP
 - Source: https://medium.com/analytics-vidhya/build-your-semantic-document-search-engine-with-tf-idf-and-google-use-c836bf5f27fb
 - Description: Build a semantic document search engine with TF-IDF and Google's USE. 
