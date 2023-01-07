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
>>> similar = BERTSimilarWords().load_dataset(dataset_path='Book_1.docx')
```
or
```python
>>> similar = BERTSimilarWords().load_dataset(dataset_path=['Book_1.docx','Book_1.txt'])
```

## Find Similar Words

Similar words can be obtained using the `find_similar_words` method. This method calculates the cosine similarity between the average of the input words based on the given context and all the words present in the given vocabulary. The parameters for this method are
- **input_words** - the input words as (list of strings)
- **input_context** - the input context (string) (optional) (default: None)
- **output_words_ngram** - n-gram words expected as output (integer) (optional) (default: 1)
  - if 1, means output like {'apple', 'car'}
  - if 2, means output like {'apple cake', 'modern car'}
  - Likewise, maximum value is 9
  - if 0, all n-grams combined like {'apple', 'apple laptop', 'red color car'}
- **max_output_words** - the maximum output words to be generated (integer) (optional) (default: 10)
- **pos_to_exclude** - the output words are excluded if these part of speech tags are present in it (list of strings) (optional) (default: None)
  - if ['VBN'], the output word "used car" will be excluded as 'used' is a verb (VBN means past particible verb)
  - availabe POS tags can be found in the [Useful Methods](#useful-methods) section
- **context_similarity_factor** - uses to tune the context matching process (optional) (default: 0.25)
- **output_filter_factor** - uses to exclude similar words in the output (optional) (default: 0.5)
- **single_word_split** - whether to split n-gram words when given as input (optional) (default: True)
- **uncased_lemmatization** - whether to uncase and lemmatize the input (optional) (default: True)

## Examples

### Example 1

```python
>>> from BERTSimilarWords import BERTSimilarWords
>>> similar = BERTSimilarWords().load_dataset(wikipedia_query='Apple', wikipedia_query_limit=5)

>>> similar.find_similar_words(input_context='company',input_words=['Apple'])
{'iPhone': 0.7655301993367924,
 'Microsoft': 0.7644559773925612,
 'Samsung': 0.7483747939272186,
 'Nokia': 0.7418908483628721,
 'Macintosh': 0.7415292245659537,
 'iOS': 0.7409453358937249,
 'AppleCare': 0.7381210698272941,
 'iPadOS': 0.7112217377139232,
 'iTunes': 0.7007508157223745,
 'macOS': 0.69984740983893}

>>> similar.find_similar_words(input_context='fruit',input_words=['Apple'])
{'applejack': 0.8045216200651304,
 'Trees': 0.7926505935113519,
 'trees': 0.7806807879003239,
 'berries': 0.7689437435792672,
 'seeds': 0.7540070238557037,
 'peaches': 0.7381803534675645,
 'Orange': 0.733131237417253,
 'orchards': 0.7296196594053761,
 'juice': 0.7247635163014543,
 'nuts': 0.724424004884171}
```

### Example 2

```python
>>> from BERTSimilarWords import BERTSimilarWords
>>> similar = BERTSimilarWords().load_dataset(wikipedia_query='Tesla', wikipedia_query_limit=10)

>>> similar.find_similar_words(input_context='Tesla Motors', input_words=['CEO'], output_words_ngram=5, max_output_words=5)
{'Chief Executive Elon Musk handing': 0.7596588355056113,
 '2018 CEO Elon Musk briefly': 0.751011374230985,
 'August 2018 CEO Elon Musk': 0.7492089016517951,
 '2021 CEO Elon Musk revealed': 0.7470401856896459,
 'SEC questioned Tesla CFO Zach': 0.738144930474394}

>>> similar.find_similar_words(input_words=['Nikola Tesla'], output_words_ngram=0, max_output_words=5)
{'Tesla Nikola Tesla Corner': 0.9203870154998232,
 'IEEEThe Nikola Tesla Memorial': 0.8932847992637643,
 'electrical engineer Nikola Tesla': 0.8811208719958945,
 'Serbian American inventor Nikola Tesla': 0.8766566716046287,
 'Nikola Tesla Technical Museum': 0.8759513407776292}
```

## Useful Methods
