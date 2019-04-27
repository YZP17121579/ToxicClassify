import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from DeepNet.deeptoxic.config import dataset_config
from DeepNet.deeptoxic.data_helper import data_loader
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from collections import defaultdict

import time


class DataTransformer(object):
    """
    You shall change data in the file data_config.
    """
    def __init__(self, max_num_words, max_sequence_length, char_level, pad_seq=True, lemmatize=True, count_null_words=True,
                 clean_wiki_tokens=True, remove_stopwords=False, stem_words=False, error_correct=False, convert_typo=False):
        self.remove_stopwords = remove_stopwords
        self.stem_words = stem_words
        self.lemmatize = lemmatize
        self.clean_wiki_tokens = clean_wiki_tokens
        self.error_correct = error_correct
        self.convert_typo = convert_typo
        self.pad_seq = pad_seq

        self.data_loader = data_loader.DataLoader()
        self.clean_word_dict = self.data_loader.load_clean_words(dataset_config.CLEAN_WORDS_PATH)
        self.train_df = self.data_loader.load_dataset(dataset_config.TRAIN_PATH)
        self.test_df = self.data_loader.load_dataset(dataset_config.TEST_PATH)

        self.max_num_words = max_num_words
        self.max_sequence_length = max_sequence_length
        self.char_level = char_level
        self.tokenizer = None

        self.word_count_dict = defaultdict(int)
        self.count_null_words = count_null_words

    def prepare_data(self):
        """
        For projects different from Toxic, changing the column names is essential.
        """
        list_sentences_train = self.train_df["comment_text"].fillna("no comment").values
        list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        list_sentences_test = self.test_df["comment_text"].fillna("no comment").values

        time_start = time.time()
        print("Doing preprocessing...")
        # clean_text
        self.train_comments = [self.clean_text(text) for text in list_sentences_train]
        self.test_comments = [self.clean_text(text) for text in list_sentences_test]

        # prepare tokenizer
        self.tokenizer = Tokenizer(num_words=self.max_num_words, char_level=self.char_level)
        self.tokenizer.fit_on_texts(self.train_comments + self.test_comments)
        print('Found %s unique tokens' % len(self.tokenizer.word_index))

        train_sequences = self.tokenizer.texts_to_sequences(self.train_comments)  # return data of list
        training_labels = self.train_df[list_classes].values
        test_sequences = self.tokenizer.texts_to_sequences(self.test_comments)
        if self.pad_seq:
            """ If pad here , mini_batch generator cannot mask unnecessary words. """
            train_sequences = pad_sequences(train_sequences, maxlen=self.max_sequence_length)
            test_sequences = pad_sequences(test_sequences, maxlen=self.max_sequence_length)  # return data of np.array

        time_end = time.time()
        print("Preprocessed within {} seconds.".format(time_end-time_start))

        return train_sequences, training_labels, test_sequences

    def build_embedding_matrix(self, embeddings_index):
        """
        :param embeddings_index: dictionary of : word --> embed_vector
        :return: embedding_matrix of sentence
        """
        nb_words = min(self.max_num_words, len(embeddings_index))
        embedding_matrix = np.zeros((nb_words, 300))
        word_index = self.tokenizer.word_index
        null_words = open('null-word.txt', 'w', encoding='utf-8')

        for word, i in word_index.items():
            if i >= self.max_num_words:
                null_words.write(word + ', ' + str(self.word_count_dict[word]) + '\n')
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                null_words.write(word + ', ' + str(self.word_count_dict[word]) + '\n')
        print('Null word embeddings: %d , with %d words in total.' % (np.sum(np.sum(embedding_matrix, axis=1) == 0),
              nb_words))
        return embedding_matrix

    def clean_text(self, text):

        text = text.lower()
        text = re.sub(
            r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)", "",
            text)
        text = re.sub(r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}", "",
                      text)

        if self.clean_wiki_tokens:
            # pictures
            text = re.sub(r"image:[a-zA-Z0-9]*\.jpg", " ", text)
            text = re.sub(r"image:[a-zA-Z0-9]*\.png", " ", text)
            text = re.sub(r"image:[a-zA-Z0-9]*\.gif", " ", text)
            text = re.sub(r"image:[a-zA-Z0-9]*\.bmp", " ", text)

            # css
            text = re.sub(r"#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})", " ", text)
            text = re.sub(r"\{\|[^\}]*\|\}", " ", text)

            # templates
            text = re.sub(r"\[?\[user:.*\]", " ", text)
            text = re.sub(r"\[?\[user:.*\|", " ", text)
            text = re.sub(r"\[?\[wikipedia:.*\]", " ", text)
            text = re.sub(r"\[?\[wikipedia:.*\|", " ", text)
            text = re.sub(r"\[?\[special:.*\]", " ", text)
            text = re.sub(r"\[?\[special:.*\|", " ", text)
            text = re.sub(r"\[?\[category:.*\]", " ", text)
            text = re.sub(r"\[?\[category:.*\|", " ", text)

        # clean char type
        for typo, correct in self.clean_word_dict.items():
            text = re.sub(typo, " " + correct + " ", text)
            # text = re.sub(typo, correct, text)

        # abbr convert
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\?", " ? ", text)
        text = re.sub(r"\!", " ! ", text)
        text = re.sub(r"\"", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        # Numeric chars
        text = re.sub(r'\d+', " ", text)
        # Get ride of punctuation  (After processed with abbr, punctuation is useless)
        text = re.sub(r"^\w\s", "", text)

        # In toxic, words like fuckkkkk or fffffuck are explicit
        if self.convert_typo:
            convert_text = text.split()
            convert_text = [w if "fuck" not in w else "fuck" for w in convert_text]
            convert_text = [w if "dick" not in w else "dick" for w in convert_text]
            convert_text = [w if "bitch" not in w else "bitch" for w in convert_text]
            text = " ".join(convert_text)

        if self.stem_words:
            text = text.split()
            stemmer = SnowballStemmer('english')
            stemmed_words = [stemmer.stem(word) for word in text]
            text = " ".join(stemmed_words)

        if self.lemmatize:
            text = text.split()
            wnl = WordNetLemmatizer()
            lemmed_words = [wnl.lemmatize(word) for word in text]
            text = " ".join(lemmed_words)

        if self.remove_stopwords:
            raise NotImplementedError

        if self.error_correct:
            text = TextBlob(text).correct()

        if self.count_null_words:
            text = text.split()
            for t in text:
                self.word_count_dict[t] += 1
            text = " ".join(text)

        # Get ride of unnecessary blanks  (when char_level == True, too many blanks may hurt the result)
        if not self.count_null_words and not self.stem_words and not self.lemmatize:
            text = " ".join(text.split())

        return text
