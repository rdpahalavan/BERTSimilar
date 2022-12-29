# BERTSimilarWords
## Find Similar Words using BERT

BERTSimilarWords is a python library that is used to find similar words using BERT. It uses a pre-trained BERT base model (cased) and cosine similarity to find the closest neighbor to the given words.

BERT generates contextual word embeddings, so the word embedding for the same word will differ based on its context. For example, the word **Apple** in *"Apple is a good fruit"* and *"Apple is a good phone"* have different word embeddings. Generating word embeddings for all vocabulary in the English language based on context is time-consuming and needs many resources. So, this library requires the vocabulary for generating word embeddings beforehand.

Vocabularies used to generate word embeddings can be given in two ways:

* Using Wikipedia pages
* Using text files (.docx and .txt)

## Install

Install the library using
```
pip install BERTSimilarWords
```

Import it using
```python
>>> from BERTSimilarWords import BERTSimilarWords
```

## Providing the Vocabulary

Provide the literature (in terms of paragraphs), so the BERT model can generate the word embeddings for all the words present in the text.

### Using Wikipedia pages

Either the name of the Wikipedia pages or the query can be given. If the query is given, the Wikipedia pages related to that query will be taken.

```python
>>> wikipedia_pages = ['Apple', 'Apple Inc.']
>>> similar = BERTSimilarWords().load_dataset(wikipedia_page_list=wikipedia_pages)
# To get the Wikipedia pages used,
>>> similar.wikipedia_dataset_info
{'Apple': 'https://en.wikipedia.org/wiki/Apple',
 'Apple Inc.': 'https://en.wikipedia.org/wiki/Apple_Inc.'}
```
or
```python
# Get 5 Wikipedia pages based on the query
>>> similar = BERTSimilarWords().load_dataset(wikipedia_query='Apple', wikipedia_query_limit=5)
# To get the Wikipedia pages used (duplicate pages are ignored),
>>> similar.wikipedia_dataset_info
{'Apple': 'https://en.wikipedia.org/wiki/Apple',
 'Apple Inc.': 'https://en.wikipedia.org/wiki/Apple_Inc.',
 'Apples to Apples': 'https://en.wikipedia.org/wiki/Apples_to_Apples',
 'MacOS': 'https://en.wikipedia.org/wiki/MacOS'}
```
or
```python
# Get 5 Wikipedia pages based on each query
>>> similar = BERTSimilarWords().load_dataset(wikipedia_query=['Apple', 'Banana'], wikipedia_query_limit=5)
# To get the Wikipedia pages used (duplicate pages are ignored),
>>> similar.wikipedia_dataset_info
{'Apple': 'https://en.wikipedia.org/wiki/Apple',
 'Apple Inc.': 'https://en.wikipedia.org/wiki/Apple_Inc.',
 'Apples to Apples': 'https://en.wikipedia.org/wiki/Apples_to_Apples',
 'MacOS': 'https://en.wikipedia.org/wiki/MacOS',
 'Banana': 'https://en.wikipedia.org/wiki/Banana',
 'Cooking banana': 'https://en.wikipedia.org/wiki/Cooking_banana',
 'Banana republic': 'https://en.wikipedia.org/wiki/Banana_republic',
 'Banana ketchup': 'https://en.wikipedia.org/wiki/Banana_ketchup'}
```

### Using text files

File extensions supported are .docx and .txt .

```python
similar = BERTSimilarWords().load_dataset(dataset_path='Book_1.docx')
```
or
```python
similar = BERTSimilarWords().load_dataset(dataset_path=['Book_1.docx','Book_1.txt'])
```
