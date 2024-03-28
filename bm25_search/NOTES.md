# NOTES


### Notes

 - `bm25_search.py` is based around the "Understanding the BM25 Ranking Algorithm" medium article.
     - For some reason, this program is outputting scores of 0.0 for all searches when it shouldn't. Will have to debug later.
 - `bm25_alt_search.py` is based around the "BM25 Distilled: Unraveling the Essence of a Robust Ranking Function" medium article.
     - Works as expected.
 - None of the above programs require 3rd party python packages. Both implement BM25 with vanilla/stock python packages.


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
