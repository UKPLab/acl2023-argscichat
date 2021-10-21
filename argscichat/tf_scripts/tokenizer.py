"""

Tensorflow specific tokenizers

"""

import collections

import tensorflow as tf
from argscichat.utility.python_utils import merge

from argscichat.generic.tokenizer import BaseTokenizer, TokenizerFactory
from argscichat.utility.embedding_utils import build_embeddings_matrix, load_embedding_model


class KerasTokenizer(BaseTokenizer):

    def __init__(self, tokenizer_args=None, **kwargs):
        super(KerasTokenizer, self).__init__(**kwargs)

        tokenizer_args = {} if tokenizer_args is None else tokenizer_args

        assert isinstance(tokenizer_args, dict) or isinstance(tokenizer_args, collections.OrderedDict)

        self.tokenizer_args = tokenizer_args

    def build_vocab(self, data, **kwargs):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(**self.tokenizer_args)
        self.tokenizer.fit_on_texts(data)
        self.vocab = self.tokenizer.word_index

        if self.build_embedding_matrix:
            self.embedding_model = load_embedding_model(model_type=self.embedding_model_type,
                                                        embedding_dimension=self.embedding_dimension)

            self.embedding_matrix, self.vocab = build_embeddings_matrix(vocab_size=len(self.vocab) + 1,
                                                                        embedding_model=self.embedding_model,
                                                                        embedding_dimension=self.embedding_dimension,
                                                                        word_to_idx=self.vocab)

    def initialize_with_info(self, info):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(**self.tokenizer_args)
        self.tokenizer.word_index = info['vocab']
        self.embedding_model = info['embedding_model']
        self.embedding_matrix = info['embedding_matrix']

    def get_info(self):
        info = super(KerasTokenizer, self).get_info()
        info['vocab_size'] = len(self.vocab) + 1

        return info

    def tokenize(self, text):
        return text

    def convert_tokens_to_ids(self, tokens):
        if type(tokens) == str:
            return self.tokenizer.texts_to_sequences([tokens])[0]
        else:
            return self.tokenizer.texts_to_sequences(tokens)

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.sequences_to_texts(ids)


class TFTokenizerFactory(TokenizerFactory):

    @classmethod
    def get_supported_values(cls):
        supported_values = super(TFTokenizerFactory, cls).get_supported_values()
        return merge(supported_values, {
            'keras_tokenizer': KerasTokenizer
        })
