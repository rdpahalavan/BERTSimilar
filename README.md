# BERTSimilarWords
## Find Similar Words using BERT

BERTSimilarWords is a python library that is used to find similar words using BERT. It uses a pre-trained BERT base model (cased) and cosine similarity to find the closest neighbor to the given words.

BERT generates contextual word embeddings, so the word embedding for the same word will differ based on its context. For example, the word **Apple** in *"Apple is a good fruit"* and *"Apple is a good phone"* have different word embeddings. Generating word embeddings for all vocabulary in the English language based on context is time-consuming and needs many resources. So, this library requires the text for generating word embeddings.
