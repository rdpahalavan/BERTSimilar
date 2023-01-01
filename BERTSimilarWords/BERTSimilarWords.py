import torch
from transformers import BertModel, AutoTokenizer, logging
import numpy as np
import warnings
import docx
import re
import nltk
import wikipedia
import itertools
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
try:
  ipython = get_ipython()
  from tqdm.notebook import tqdm
except:
  from tqdm import tqdm

logging.set_verbosity_error()
logging.disable_progress_bar()
warnings.filterwarnings('ignore')
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('tagsets', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class BERTSimilarWords:

    def __init__(self):

        for i in tqdm(range(2), unit=' it', desc='Initializing', postfix='Tokenizer and Model'): pass
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.lemmatizer = WordNetLemmatizer()
        self.min_max_scaler = MinMaxScaler()
        self.model = BertModel.from_pretrained('bert-base-cased')
        if torch.cuda.is_available():
            self.processor = 'GPU'
            self.cuda_current_device = torch.cuda.current_device()
            self.model = self.model.to(self.cuda_current_device)
        else:
            self.processor = 'CPU'
        self.max_document_length = 300
        self.max_heading_length = 10
        self.max_ngram = 10
        self.wikipedia_dataset_info = {}
        self.document_list = []
        self.bert_words = []
        self.bert_vectors = []
        self.bert_documents = []
        self.continous_words = []
        self.temporary_ngram_words = []
        self.count_vectorizer_words = []
        self.cv_counts = []
        self.cv_words = []
        self.count_vectorizer = CountVectorizer(analyzer=self._custom_analyzer)
        self.stop_words = stopwords.words()
        self.punctuations = '''!"#$%&'()*+,-./:—;<=>−?–@[\]^_`{|}~'''
        self.doc_regex = "[\([][0-9]+[\])]|[”“‘’‛‟]|\d+\s"
        self.punctuations_continuity_exclude = '''—-–,−'''
        self.pos_tags_info = nltk.help.upenn_tagset
        self.bert_words_ngram = [[] for _ in range(self.max_ngram)]
        self.bert_vectors_ngram = [[] for _ in range(self.max_ngram)]
        self.bert_documents_ngram = [[] for _ in range(self.max_ngram)]

    def load_dataset(self, dataset_path=None, wikipedia_query=None, wikipedia_query_limit=10, wikipedia_page_list=None):

        if wikipedia_query is not None or wikipedia_page_list is not None:
            if wikipedia_query is not None:
                query_results = []
                if type(wikipedia_query) == str:
                    wikipedia_query = [wikipedia_query]
                for query in wikipedia_query:
                    query_results += wikipedia.search(query, results=wikipedia_query_limit)
            else:
                query_results = wikipedia_page_list
            page_content = []
            for result in tqdm(query_results, unit=' pages', desc='Extracting', postfix='Data from Wikipedia'):
                if '(disambiguation)' not in result and result not in self.wikipedia_dataset_info.keys():
                    try:
                        page = wikipedia.page(result, auto_suggest=False)
                    except:
                        continue
                    page_content += ['== New page =='] + page.content.split('\n\n\n')
                    self.wikipedia_dataset_info[page.title] = page.url
            self.document_list = self._process_wikipedia_dataset(page_content)
        elif dataset_path is not None:
            if type(dataset_path) == str:
                dataset_path = [dataset_path]
            for path in dataset_path:
                if path.endswith('.docx'):
                    docx_content = docx.Document(path)
                    self.document_list += self._process_docx_dataset(docx_content)
                elif path.endswith('.txt'):
                    self.document_list += self._process_txt_dataset(path)
                else:
                    raise ValueError("Files supported: .docx / .txt")
        for words, vectors, document, continous in self._tokenize_and_embeddings(self.document_list):
            self.temporary_ngram_words = []
            for i in range(len(words)):
                self._generate_n_grams(i, words, vectors, document, continous)
            self.bert_words.extend(words)
            self.bert_vectors.extend(vectors)
            self.bert_documents.extend(document)
            self.continous_words.extend(continous)
            self.count_vectorizer_words.append(words + self.temporary_ngram_words)
        self.bert_words_ngram[0] = self.bert_words
        self.bert_vectors_ngram[0] = self.bert_vectors
        self.bert_documents_ngram[0] = self.bert_documents
        self.cv_counts = self.count_vectorizer.fit_transform(self.count_vectorizer_words)
        self.cv_words = self.count_vectorizer.get_feature_names_out()
        return self

    def _process_wikipedia_dataset(self, page_content):

        document_list = []
        for section in page_content:
            if not any(exclude in section for exclude in
                       ['== Further reading ==', '== References ==', '== External links ==', '== See also ==',
                        '== Notes ==']):
                if "==" in section[:self.max_heading_length] and "===" not in section[:self.max_heading_length]:
                    flag = 0
                paragraph = section.split('\n')
                for sentence in paragraph:
                    sentence_words = sentence.split()
                    sentence_length = len(sentence_words)
                    if sentence_length > self.max_heading_length:
                        if len(document_list) != 0 and flag == 1 and len(
                                document_list[-1].split() + sentence_words) < self.max_document_length:
                            document_list[-1] += ' ' + sentence
                        else:
                            document_list = self._process_dataset_long_paragraph(document_list, sentence,
                                                                                 sentence_length)
                            flag = 1
        return document_list

    def _process_docx_dataset(self, docx_content):

        document_list = []
        for paragraph in tqdm(docx_content.paragraphs, unit=' paragraphs', desc='Extracting',
                              postfix='Data from Dataset'):
            if 'Heading' in str(paragraph.style):
                text = re.sub(self.doc_regex, '', paragraph.text)
                if len(document_list) != 0 and len(document_list[-1].split()) <= self.max_heading_length:
                    document_list[-1] = text + '.'
                else:
                    document_list.append(text + '.')
            if 'Body Text' in str(paragraph.style):
                sentence = re.sub(self.doc_regex, '', paragraph.text)
                sentence_length = len(sentence.split())
                if sentence_length > self.max_heading_length:
                    if len(document_list) != 0 and len(
                            document_list[-1].split()) + sentence_length < self.max_document_length:
                        document_list[-1] += ' ' + sentence
                    else:
                        document_list = self._process_dataset_long_paragraph(document_list, sentence, sentence_length)
        return document_list

    def _process_txt_dataset(self, path):

        document_list = []
        with open(path) as file:
            for line in tqdm(file.readlines(), unit=' paragraphs', desc='Extracting', postfix='Data from Dataset'):
                line_text = line.strip()
                line_text = re.sub(self.doc_regex, '', line_text)
                line_length = len(line_text.split())
                if 0 < line_length <= self.max_heading_length:
                    if len(document_list) != 0 and len(document_list[-1].split()) <= self.max_heading_length:
                        document_list[-1] = line_text + '.'
                    else:
                        document_list.append(line_text + '.')
                elif line_length > self.max_heading_length:
                    if len(document_list) != 0 and len(
                            document_list[-1].split()) + line_length < self.max_document_length:
                        document_list[-1] += ' ' + line_text
                    else:
                        document_list = self._process_dataset_long_paragraph(document_list, line_text, line_length)
        return document_list

    def _process_dataset_long_paragraph(self, document_list, sentence, sentence_length):

        if sentence_length > self.max_document_length:
            for i in range(2, sentence_length):
                div = sentence_length / i
                if div < self.max_document_length:
                    break
            temp_sent = ''
            sm_sent = sent_tokenize(sentence)
            for sent in sm_sent:
                if len(temp_sent.split() + sent.split()) > div:
                    document_list.append(temp_sent)
                    temp_sent = ''
                temp_sent = temp_sent + sent
            if len(document_list[-1].split() + temp_sent.split()) < self.max_document_length:
                document_list[-1] += ' ' + temp_sent
            else:
                document_list.append(temp_sent)
        else:
            document_list.append(sentence)
        return document_list

    def _tokenize_and_embeddings(self, document_list):

        continous_index = 0
        document_index = 0
        for document in tqdm(document_list, unit=' documents', desc='Processing', postfix='Word Embeddings'):
            if self.processor == 'GPU':
                tokens = self.tokenizer(document, truncation=True, return_tensors='pt').to(self.cuda_current_device)
            else:
                tokens = self.tokenizer(document, truncation=True, return_tensors='pt')
            words = self.tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
            word_ids = tokens.word_ids()
            output = self.model(**tokens)
            if self.processor == 'GPU':
                vectors = output.last_hidden_state[0].cpu().detach().numpy()
            else:
                vectors = output.last_hidden_state[0].detach().numpy()
            word_list = []
            vector_list = []
            continous_words = []
            word_index = -1
            for i in range(len(words)):
                if word_ids[i] is None or words[i] in self.punctuations:
                    if words[i] in self.punctuations_continuity_exclude:
                        pass
                    else:
                        continous_index = continous_index + 1
                    continue
                if word_ids[i] > word_index:
                    if len(word_list) != 0 and word_list[-1].lower() in self.stop_words:
                        word_list.pop()
                        vector_list.pop()
                        continous_words.pop()
                        continous_index = continous_index + 1
                    word_list.append(words[i])
                    vector_list.append(vectors[i])
                    continous_words.append(continous_index)
                    word_index = word_ids[i]
                elif word_ids[i] == word_index:
                    sub_word = words[i].replace('##', "")
                    word_list[-1] = word_list[-1] + sub_word
                    vector_list[-1] = (vector_list[-1] + vectors[i])
                    if word_ids[i + 1] != word_ids[i]:
                        vector_list[-1] = vector_list[-1] / word_ids.count(word_index)
            yield word_list, vector_list, [document_index] * len(word_list), continous_words
            document_index += 1

    def _generate_n_grams(self, i, words, vectors, document, continous, n=1):

        if i > n - 1 and n < self.max_ngram and continous[i] == continous[i - n]:
            temp_word = ''
            temp_vector = np.zeros([len(vectors[i])])
            for j in range(n, -1, -1):
                temp_word = temp_word + ' ' + words[i - j]
                temp_vector = temp_vector + vectors[i - j]
            self.temporary_ngram_words.append(temp_word.strip())
            self.bert_words_ngram[n].append(temp_word.strip())
            self.bert_vectors_ngram[n].append(temp_vector / (n + 1))
            self.bert_documents_ngram[n].append(document[i])
            self._generate_n_grams(i, words, vectors, document, continous, n=n + 1)
        return

    def _custom_analyzer(self, words):

        final_list = []
        for word in words:
            final_list.append(word)
            lemmatized_word = ' '.join([self.lemmatizer.lemmatize(token.lower()) for token in word.split()])
            if word != lemmatized_word:
                final_list.append(lemmatized_word)
        return final_list

    def _context_similarity_measurement(self, features, context_length):

        context_total = 0
        word_total = 0
        for i in range(context_length):
            if features[i] != 0:
                context_total += 1
        for i, x in enumerate(features[context_length:]):
            if x != 0:
                word_total += 1
        word_mean = 0.5 * np.mean(features[context_length:])
        if len(features[:context_length]) == 0:
            context_mean = 0
        else:
            context_mean = 0.5 * np.mean(features[:context_length])
        return int(str(context_total) + str(word_total)) + context_mean + word_mean

    def _get_article_words_vectors(self, similar_documents, similarity_scores, similarity_factor, input_words_max):

        document_words = []
        document_vectors = []
        for article in similar_documents:
            if similarity_scores[article] < similarity_scores[similar_documents[0]] - similarity_factor:
                break
            if article == len(similar_documents) - 1:
                for i in range(input_words_max):
                    document_words += self.bert_words_ngram[i][self.bert_documents_ngram[i].index(article):]
                    document_vectors += self.bert_vectors_ngram[i][self.bert_documents_ngram[i].index(article):]
            else:
                for i in range(input_words_max):
                    document_words += self.bert_words_ngram[i][
                                      self.bert_documents_ngram[i].index(article):self.bert_documents_ngram[i].index(
                                          article + 1)]
                    document_vectors += self.bert_vectors_ngram[i][
                                        self.bert_documents_ngram[i].index(article):self.bert_documents_ngram[i].index(
                                            article + 1)]
        return document_words, document_vectors

    def _calculate_input_word_embedding(self, input_words, document_words, document_vectors, uncased_lemmatization):

        average_list = np.zeros([len(input_words), len(document_vectors[0])])
        mean_index = []
        for i_index, i_word in enumerate(input_words):
            a_count = 0
            for a_index, a_word in enumerate(document_words):
                if uncased_lemmatization and i_word == self.lemmatizer.lemmatize(a_word.lower()):
                    average_list[i_index] += document_vectors[a_index]
                    a_count = a_count + 1
                elif i_word == a_word:
                    average_list[i_index] += document_vectors[a_index]
                    a_count = a_count + 1
            if average_list[i_index].any():
                average_list[i_index] = average_list[i_index] / a_count
                mean_index.append(i_index)
        average = np.mean(average_list[mean_index], axis=0)
        return average

    def _context_similarity_document_scores(self, input_context_words, input_context_length, input_words_length,
                                            context_similarity_factor):

        cv_list = []
        cv_counts = self.cv_counts.toarray()
        index = [i for i in np.searchsorted(self.cv_words, input_context_words) if
                 self.cv_words[i] in input_context_words]

        for i in range(len(self.document_list)):
            cv_list.append(cv_counts[i][index].tolist())

        cv_list = self.min_max_scaler.fit_transform(cv_list)
        similarity_scores = [self._context_similarity_measurement(counts, input_context_length) for counts in cv_list]
        similarity_factor = context_similarity_factor * input_words_length
        similar_documents = np.flip(np.argsort(similarity_scores))
        return similar_documents, similarity_scores, similarity_factor

    def _find_nearest_cosine_words(self, input_context_words, cosine_sim, cosine_words, pos_to_exclude,
                                   max_output_words, output_filter_factor):

        output_dict = {}
        sorted_list = np.flip(np.argsort(cosine_sim))
        lemmatized_words = {self.lemmatizer.lemmatize(token.lower()) for word in input_context_words for token in
                            word.split()}

        for i in range(len(cosine_words)):
            stop = 0
            pop_list = []
            original_word = cosine_words[sorted_list[i]]
            pos_tags = [pos[1] for pos in nltk.pos_tag(original_word.split())]
            lemmatized_word = {self.lemmatizer.lemmatize(token.lower()) for token in original_word.split()}
            if len(lemmatized_words.intersection(lemmatized_word)) > output_filter_factor * len(original_word.split()):
                continue
            if any(pos in pos_tags for pos in pos_to_exclude):
                continue
            if original_word not in output_dict.keys():
                for word in output_dict.keys():
                    if original_word in word:
                        stop = 1
                        break
                    elif word in original_word:
                        pop_list.append(word)
                        stop = 0
                if stop == 0:
                    pop = [output_dict.pop(key) for key in pop_list]
                    output_dict[original_word] = cosine_sim[sorted_list[i]]
                    if len(output_dict.keys()) == max_output_words:
                        break
        return output_dict

    def _process_input_context_words(self, input_context, input_words, single_word_split, uncased_lemmatization):

        if single_word_split:
            input_context_split = input_context.split()
            input_words_split = list(itertools.chain.from_iterable([word.split() for word in input_words]))
            input_words_max = 1
        else:
            input_context_split = [] if input_context == '' else [input_context]
            input_words_split = input_words
            input_words_max = max([len(word.split()) for word in input_words])
        if uncased_lemmatization:
            input_context_split = [' '.join([self.lemmatizer.lemmatize(token.lower()) for token in word.split()]) for
                                   word in input_context_split]
            input_words_split = [' '.join([self.lemmatizer.lemmatize(token.lower()) for token in word.split()]) for word
                                 in input_words_split]
        input_context_words = input_context_split + input_words_split
        input_context_words_max = max([len(word.split()) for word in input_context_words])
        return input_context_split, input_words_split, input_words_max, input_context_words, input_context_words_max

    def find_similar_words(self,
                           input_context='',
                           input_words=[],
                           output_words_ngram=1,
                           pos_to_exclude=[],
                           max_output_words=10,
                           context_similarity_factor=0.25,
                           output_filter_factor=0.5,
                           single_word_split=True,
                           uncased_lemmatization=True
                           ):

        input_context_split, input_words_split, input_words_max, input_context_words, input_context_words_max = self._process_input_context_words(
            input_context, input_words, single_word_split, uncased_lemmatization)
        similar_documents, similarity_scores, similarity_factor = self._context_similarity_document_scores(
            input_context_words, len(input_context_split), len(input_words_split), context_similarity_factor)
        document_words, document_vectors = self._get_article_words_vectors(similar_documents, similarity_scores,
                                                                           similarity_factor, input_words_max)
        input_embedding = self._calculate_input_word_embedding(input_words_split, document_words, document_vectors,
                                                               uncased_lemmatization)
        if output_words_ngram == 0:
            cosine_sim = cosine_similarity(list(itertools.chain.from_iterable(self.bert_vectors_ngram)),
                                           [input_embedding]).flatten()
            cosine_words = list(itertools.chain.from_iterable(self.bert_words_ngram))
        else:
            cosine_sim = cosine_similarity(self.bert_vectors_ngram[output_words_ngram - 1], [input_embedding]).flatten()
            cosine_words = self.bert_words_ngram[output_words_ngram - 1]
        output_dictionary = self._find_nearest_cosine_words(input_context_words, cosine_sim, cosine_words,
                                                            pos_to_exclude, max_output_words, output_filter_factor)
        return output_dictionary