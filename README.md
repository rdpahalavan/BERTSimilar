# BERTSimilar

## Get Similar Words and Embeddings using BERT Models

BERTSimilar is used to get similar words and embeddings using BERT models. It uses **bert-base-cased** model as default and cosine similarity to find the closest word to the given words.

BERT generates contextual word embeddings, so the word embedding for the same word will differ based on its context. For example, the word **Apple** in *"Apple is a good fruit"* and *"Apple is a good phone"* have different word embeddings. Generating word embeddings for all vocabulary in the English language based on context is time-consuming and needs many resources. So, this library requires the vocabulary for generating word embeddings beforehand.

Vocabularies used to generate word embeddings can be given in two ways:

* Using Wikipedia Pages
* Using Text Files (.docx and .txt)

## Install and Import

Install the Python package using
```
pip install BERTSimilar
```

Import the module using
```python
>>> from BERTSimilar import SimilarWords
```

## Providing the Vocabulary

Provide the text (in terms of paragraphs), so the BERT model can generate the word embeddings for all the words present in the text.

### Using Wikipedia Pages

1) Using Wikipedia page names as a list (the content of the pages will be taken as input and processed)

```python
>>> wikipedia_pages = ['Apple', 'Apple Inc.']
>>> similar = SimilarWords().load_dataset(wikipedia_page_list=wikipedia_pages)

# To get the Wikipedia pages used,
>>> similar.wikipedia_dataset_info
{'Apple': 'https://en.wikipedia.org/wiki/Apple',
 'Apple Inc.': 'https://en.wikipedia.org/wiki/Apple_Inc.'}
```

2) Using Wikipedia search query as a string (the content of the pages related to the query will be taken as input and processed)

```python
# Get 5 Wikipedia pages based on the query
>>> similar = SimilarWords().load_dataset(wikipedia_query='Apple', wikipedia_query_limit=5)

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
>>> similar = SimilarWords().load_dataset(wikipedia_query=['Apple', 'Banana'], wikipedia_query_limit=5)

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

1) Using a single text file (the content of the file will be taken as input and processed)

```python
>>> similar = SimilarWords().load_dataset(dataset_path='Book_1.docx')
```

2) Using multiple text files (the contents of each file will be taken as input and processed)

```python
>>> similar = SimilarWords().load_dataset(dataset_path=['Book_1.docx','Book_1.txt'])
```

### BERTSimilar() Parameters

You can pass these parameters to customize the initialization.

- **model** - the BERT model to use (default: bert-base-cased)
- **max_heading_length** - the maximum heading length. Lengths more than this are considered paragraphs (default: 10)
- **max_document_length** - the maximum paragraph length. Lengths more than this are split into multiple paragraphs (default: 300)
- **exclude_stopwords** - by default all stopwords are excluded from tags. To include stopwords, pass the stopwords as a list of strings to include (default: None)
- **embeddings_scaler** - Scaler to standardize the embeddings (default: None)

## Find Similar Words

Similar words are generated using the `find_similar_words` method. This method calculates the cosine similarity between the average of the input words based on the given context and all the words present in the given vocabulary. The similar words and the embedding used to select the nearest words will be returned. This embedding is the representation of input words and context. The parameters for this method are

- **input_words** - the input words (list of strings)
- **input_context** - the input context (string) (optional) (default: None)
- **input_embedding** - an embedding can be given in place of input words and context (numpy array) (default: None)
- **output_words_ngram** - n-gram words expected as output (integer) (optional) (default: 1)
  - if 1, means output like *{'apple', 'car'}*
  - if 2, means output like *{'apple cake', 'modern car'}*
  - likewise, the maximum value is 10
  - if 0, all n-grams combined like *{'Apple', 'Apple laptop', 'red color car'}*
- **max_output_words** - the maximum number of output words to be generated (integer) (optional) (default: 10)
- **pos_to_exclude** - the words are ignored in the output if these part of speech tags are present in it (list of strings) (optional) (default: None)
  - if ['VBN'], the word *"used car"* will be ignored in the output as *"used"* is a verb (VBN means past participle verb)
  - available POS tags can be found in the [Attributes](#Attributes) section
- **context_similarity_factor** - used to tune the context-matching process, and find the best paragraphs related to the given input words (float) (optional) (default: 0.25)
  - possible values are from 0 to 1
  - value closer to 0 will do a strict context-matching and a closer to 1 will do lenient context-matching
- **output_filter_factor** - uses to ignore words that are similar to the given input in the output (float) (optional) (default: 0.5)
  - possible values are from 0 to 1
  - value closer to 0 will do a strict comparison and a value closer to 1 will do a lenient comparison
- **single_word_split** - whether to split n-gram words when given as input (boolean) (optional) (default: True)
  - whether to split the n-gram words given as input into single words
  - if True, *"Apple phones"* given as input will be split into *"Apple"* and *"phones"* separately and processed
- **uncased_lemmatization** - whether to uncase and lemmatize the input (boolean) (optional) (default: True)
  - whether to uncase and lemmatize the input
  - if True, *"Apple phones"* given as input will be converted to *"apple phone"* and processed

## Demo

### Example 1

```python
>>> from BERTSimilar import SimilarWords
>>> similar = SimilarWords().load_dataset(wikipedia_query='Apple', wikipedia_query_limit=5)

>>> words, embedding = similar.find_similar_words(input_context='company',input_words=['Apple'])
>>> words
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

>>> words, embedding = similar.find_similar_words(input_context='fruit',input_words=['Apple'])
>>> words
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
>>> from BERTSimilar import SimilarWords
>>> similar = SimilarWords().load_dataset(wikipedia_query='Tesla', wikipedia_query_limit=10)

>>> words, embedding = similar.find_similar_words(input_context='Tesla Motors', input_words=['CEO'], output_words_ngram=5, max_output_words=5)
>>> words
{'Chief Executive Elon Musk handing': 0.7596588355056113,
 '2018 CEO Elon Musk briefly': 0.751011374230985,
 'August 2018 CEO Elon Musk': 0.7492089016517951,
 '2021 CEO Elon Musk revealed': 0.7470401856896459,
 'SEC questioned Tesla CFO Zach': 0.738144930474394}

>>> words, embedding = similar.find_similar_words(input_words=['Nikola Tesla'], output_words_ngram=0, max_output_words=5)
>>> words
{'Tesla Nikola Tesla Corner': 0.9203870154998232,
 'IEEEThe Nikola Tesla Memorial': 0.8932847992637643,
 'electrical engineer Nikola Tesla': 0.8811208719958945,
 'Serbian American inventor Nikola Tesla': 0.8766566716046287,
 'Nikola Tesla Technical Museum': 0.8759513407776292}
```

## Attributes

These attributes can be used to get values or modify default values of the SimilarWords class.

To get the value of the attributes,

```python
>>> similar = SimilarWords().load_dataset(dataset_path='Book_1.docx')

# This will return all the words
>>> similar.bert_words

# This will return the embeddings for all the words
>>> similar.bert_vectors
```

To change the values of the attributes,

```python
>>> similar = SimilarWords()
>>> similar.max_ngram
10

>>> similar.max_ngram = 12
>>> similar = similar.load_dataset(dataset_path='Book_1.docx')
>>> similar.max_ngram
12
```

- **tokenizer** -  to get the BERT tokenizer
- **model** - to get the BERT model
- **bert_words** -  to get all words
- **bert_vectors** -  to get embeddings of all words
- **bert_words_ngram** - to get the n-gram words
  - bert_words_ngram[0] gives unigram words
  - bert_words_ngram[1] gives bigram words
  - bert_words_ngram[n-1] gives n-gram words
- **bert_vectors_ngram** - to get the BERT word embeddings for the n-gram words
  - bert_words_ngram[0] gives word embeddings of the unigram words
  - bert_words_ngram[1] gives word embeddings of the bigram words
  - bert_words_ngram[n-1] gives word embeddings of the n-gram words
- **bert_words_all** - to get all n-gram words as a flattened list
- **bert_vectors_all** - to get all embeddings as a flattened list
- **document_list** - to get the paragraphs
- **max_ngram** - maximum n-gram words to generate
  - default: 10 (10-gram words)
- **punctuations** - to get the punctuations to be removed from the dataset
  - default: '''!"#$%&\'()*+,-./:—;<=>−?–@[\\]^_`{|}~'''
- **doc_regex** - the regular expression to be used to process the text files
  - default: "[\([][0-9]+[\])]|[”“‘’‛‟]|\d+\s"
- **stop_words** - the stop words to be ignored in the output (can be modified)
- **max_heading_length** - total words in a paragraph less than this length will be considered as heading
  - default: 10
- **max_document_length** - total words in a paragraph greater than this will be split into multiple paragraphs
  - default: 300
- **pos_tags_info()** - to get the POS tags and information to be used in the `find_similar_words` method
