# NOTES


### Notes

 - `bm25_search.py` is based around the "Understanding the BM25 Ranking Algorithm" medium article.
     - For some reason, this program is outputting scores of 0.0 for all searches when it shouldn't. Will have to debug later.
 - `bm25_alt_search.py` is based around the "BM25 Distilled: Unraveling the Essence of a Robust Ranking Function" medium article.
     - Works as expected.
 - None of the above programs require 3rd party python packages. Both implement BM25 with vanilla/stock python packages.
 - Key Points about BM25 (from the "BM25 Distilled: Unraveling the Essence of a Robust Ranking Function" medium article):
     - BM25 (Best Matching 25) is a ranking algorithm used in information retrieval systems to determine the relevance of documents to a given query.
     - It was developed in the late 1990s as an extension of the BM11 algorithm to address limitations in existing ranking models.
     - BM25 incorporates two fundamental principles: term frequency-inverse document frequency (TF-IDF) weighting and document length normalization.
     - TF-IDF assigns weights to terms based on their frequency in a document and their importance in the entire collection, emphasizing rare terms and downplaying common ones.
     - Document length normalization ensures that longer documents are not favored over shorter ones by normalizing their scores.
     - BM25 consists of three key components: term frequency (TF), inverse document frequency (IDF), and document length normalization.
     - The TF component uses a logarithmic term frequency function to mitigate the impact of excessively frequent terms.
     - IDF measures the significance of a term across the entire document collection and assigns higher weights to rare terms and lower weights to common terms.
     - Document length normalization divides the document length by an average document length to prevent longer documents from having an unfair advantage.
     - BM25's parameters, such as the k1 parameter and the b parameter, can be tuned to optimize its performance for specific retrieval tasks and datasets.
     - The k1 parameter determines the term frequency saturation point, and a higher value promotes higher term frequency saturation.
     - The b parameter governs the impact of document length normalization, influencing the extent to which longer documents are penalized.
     - BM25's simplicity, effectiveness, and versatility have led to its widespread adoption in various information retrieval systems, including search engines, question-answering systems, and recommender systems.
     - It continues to play a crucial role in facilitating efficient information retrieval and ensuring a balance between precision and recall.
     - BM25's distilled essence makes it a reliable tool for ranking documents and assisting users in navigating the vast landscape of information in the digital world.


### References

 - Okapi BM25 [wikipedia article](https://en.wikipedia.org/wiki/Okapi_BM25)
 - Understanding the BM25 Ranking Algorithm ([medium article](https://pub.aimind.so/understanding-the-bm25-ranking-algorithm-19f6d45c6ce)) (premium article)
     - [freedium link](https://freedium.cfd/https://pub.aimind.so/understanding-the-bm25-ranking-algorithm-19f6d45c6ce)
 - BM25 Distilled: Unraveling the Essence of a Robust Ranking Function ([medium article](https://ai.plainenglish.io/bm25-distilled-unraveling-the-essence-of-a-robust-ranking-function-5a0a6393c058)) (premium)
     - [freedium link](https://freedium.cfd/https://ai.plainenglish.io/bm25-distilled-unraveling-the-essence-of-a-robust-ranking-function-5a0a6393c058)
 - Build your own NLP based search engine Using BM25 ([Analytics Vidhya post](https://www.analyticsvidhya.com/blog/2021/05/build-your-own-nlp-based-search-engine-using-bm25/))
     - disable javascript 
 - Finding relevant patents via a simple BM25 search engine in Python ([medium article](https://foongminwong.medium.com/finding-relevant-patents-via-a-simple-bm25-search-engine-in-python-b84a62ae87ee))
 - rank-bm25 [pypi module](https://pypi.org/project/rank-bm25/)
