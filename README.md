# BERTSimilarWords

## Find Similar Words using BERT

BERTSimilarWords is a python library that is used to find similar words using BERT. It uses a pre-trained BERT base model (cased) and cosine similarity to find the closest neighbor to the given words. It is similar to the Gensim Word2Vec similar words, but with context.

BERT generates contextual word embeddings, so the word embedding for the same word will differ based on its context. For example, the word **Apple** in *"Apple is a good fruit"* and *"Apple is a good phone"* have different word embeddings. Generating word embeddings for all vocabulary in the English language based on context is time-consuming and needs many resources. So, this library requires the vocabulary for generating word embeddings beforehand.

Vocabularies used to generate word embeddings can be given in two ways:

* [Using Wikipedia Pages](#using-wikipedia-pages)
* [Using Text Files](#using-text-files)

## Install and Import

Install the Python package using
```
pip install BERTSimilarWords
```

Import the module using
```python
>>> from BERTSimilarWords import BERTSimilarWords
```

## Providing the Vocabulary

Provide the text (in terms of paragraphs), so the BERT model can generate the word embeddings for all the words present in the text.

### Using Wikipedia Pages

1) Using Wikipedia page names as a list (the content of the pages will be taken as input and processed)

```python
>>> wikipedia_pages = ['Apple', 'Apple Inc.']
>>> similar = BERTSimilarWords().load_dataset(wikipedia_page_list=wikipedia_pages)

# To get the Wikipedia pages used,
>>> similar.wikipedia_dataset_info
{'Apple': 'https://en.wikipedia.org/wiki/Apple',
 'Apple Inc.': 'https://en.wikipedia.org/wiki/Apple_Inc.'}
```

2) Using Wikipedia search query as string (the content of the pages related to the query will be taken as input and processed)

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

3) Using Wikipedia search queries as a list (the content of the pages related to each query will be taken as input and processed)

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

### Using Text Files

File extensions supported are .docx and .txt (For other file types, please convert them to the supporting format)

1) Using single text file (the content of the file will be taken as input and processed)

```python
>>> similar = BERTSimilarWords().load_dataset(dataset_path='Book_1.docx')
```

2) Using multiple text files (the contents of each file will be taken as input and processed)

```python
>>> similar = BERTSimilarWords().load_dataset(dataset_path=['Book_1.docx','Book_1.txt'])
```

## Find Similar Words

Similar words can be generated using the `find_similar_words` method. This method calculates the cosine similarity between the average of the input words based on the given context and all the words present in the given vocabulary. The parameters for this method are

- **input_words** - the input words (list of strings)
- **input_context** - the input context (string) (optional) (default: None)
- **output_words_ngram** - n-gram words expected as output (integer) (optional) (default: 1)
  - if 1, means output like *{'apple', 'car'}*
  - if 2, means output like *{'apple cake', 'modern car'}*
  - likewise, maximum value is 10
  - if 0, all n-grams combined like *{'Apple', 'Apple laptop', 'red color car'}*
- **max_output_words** - the maximum number of output words to be generated (integer) (optional) (default: 10)
- **pos_to_exclude** - the words are ignored in the output if these part of speech tags are present in it (list of strings) (optional) (default: None)
  - if ['VBN'], the word *"used car"* will be ignored in the output as *"used"* is a verb (VBN means past particible verb)
  - availabe POS tags can be found in the [Useful Attributes](#useful-attributes) section
- **context_similarity_factor** - uses to tune the context-matching process, find the best paragraphs related to the given input words (float) (optional) (default: 0.25)
  - possible valuse are from 0 to 1
  - value closer to 0 will do a strict context-matching and closer to 1 will do lenient context-matching
- **output_filter_factor** - uses to ignore words that are similar to the given input in the output (float) (optional) (default: 0.5)
  - possible values are from 0 to 1
  - value closer to 0 will do strick comparison and value closer to 1 will do lenient comparison
- **single_word_split** - whether to split n-gram words when given as input (boolean) (optional) (default: True)
  - whether to split the n-gram words given as input into single words
  - if True, *"Apple phones"* given as input will be split into *"Apple"* and *"phones"* separately and processed
- **uncased_lemmatization** - whether to uncase and lemmatize the input (boolean) (optional) (default: True)
  - whether to uncase and lemmatize the input
  - if True, *"Apple phones"* given as input will be converted to *"apple phone"* and processed

## Full Examples

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

## Useful Attributes

These attributes can be used to get values or modify default values, and can be used after the `load_dataset` method. For example, to get the maximum n-gram supported
```python
>>> similar = BERTSimilarWords().load_dataset(dataset_path='Book_1.docx')

# This will give the maximum n-gram supported (default: 10)
>>> similar.max_ngram
10

# To change this to only support up to 5-gram words
>>> similar.max_ngram = 5
```

- **bert_words_ngram** - to get the n-gram words
  - bert_words_ngram[0] gives unigram words
  - bert_words_ngram[1] gives bigram words
  - bert_words_ngram[n-1] gives n-gram words
- **bert_words_ngram** - to get the BERT word embeddings for the n-gram words
  - bert_words_ngram[0] gives word embeddings of the unigram words
  - bert_words_ngram[1] gives word embeddings of the bigram words
  - bert_words_ngram[n-1] gives word embeddings of the n-gram words
- **document_list** - to get the paragraphs
- **punctuations** - to get the punctuations to be removed from the dataset (can be modified)
  - default: '''!"#$%&\'()*+,-./:—;<=>−?–@[\\]^_`{|}~'''
- **doc_regex** - the regular expression to be used to process the text files (can be modified)
  - default: "[\([][0-9]+[\])]|[”“‘’‛‟]|\d+\s"
- **stop_words** - the stop words to be ignored in the output (can be modified)
- **max_heading_length** - total words in a paragraph less than this length will be considered as heading (can be modified)
  - default: 10
- **pos_tags_info()** - to get the POS tags and information to be used in the `find_similar_words` method

## References

1) Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Le Scao, T., Gugger, S., Drame, M., Lhoest, Q., & Rush, A. M. (2020). Transformers: State-of-the-Art Natural Language Processing [Conference paper]. 38–45. https://www.aclweb.org/anthology/2020.emnlp-demos.6
