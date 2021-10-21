"""

Generic tokenizers

"""

import os

import numpy as np
from transformers import DistilBertTokenizer, AutoTokenizer

from argscichat.generic.factory import Factory
from argscichat.utility.log_utils import Logger


class BaseTokenizer(object):

    def __init__(self, build_embedding_matrix=False, embedding_dimension=None,
                 embedding_model_type=None, merge_vocabularies=False):
        if build_embedding_matrix:
            assert embedding_model_type is not None
            assert embedding_dimension is not None and type(embedding_dimension) == int

        self.build_embedding_matrix = build_embedding_matrix
        self.embedding_dimension = embedding_dimension
        self.embedding_model_type = embedding_model_type
        self.embedding_model = None
        self.embedding_matrix = None
        self.merge_vocabularies = merge_vocabularies
        self.vocab = None

    def build_vocab(self, data, **kwargs):
        raise NotImplementedError()

    def initialize_with_info(self, info):
        pass

    def tokenize(self, text):
        raise NotImplementedError()

    def convert_tokens_to_ids(self, tokens):
        raise NotImplementedError()

    def convert_ids_to_tokens(self, ids):
        raise NotImplementedError()

    def get_info(self):
        return {
            'build_embedding_matrix': self.build_embedding_matrix,
            'embedding_dimension': self.embedding_dimension,
            'embedding_model_type': self.embedding_model_type,
            'embedding_matrix': self.embedding_matrix,
            'embedding_model': self.embedding_model,
            'vocab_size': len(self.vocab) + 1,
            'vocab': self.vocab
        }

    def show_info(self, info=None):

        info = info if info is not None else self.get_info()
        info = {key: value for key, value in info.items() if key != 'vocab'}

        Logger.get_logger(__name__).info('Tokenizer info: {}'.format(info))

    def save_info(self, filepath, prefix=None):
        save_name = 'tokenizer_info'
        if prefix is not None:
            save_name += '_{}'.format(prefix)

        filepath = os.path.join(filepath, '{}.npy'.format(save_name))
        np.save(filepath, self.get_info())

    @staticmethod
    def load_info(filepath, prefix=None):
        load_name = 'tokenizer_info'
        if prefix is not None:
            load_name += '_{}'.format(prefix)

        filepath = os.path.join(filepath, '{}.npy'.format(load_name))
        return np.load(filepath, allow_pickle=True).item()


class SciBertBaseTokenizerWrapper(BaseTokenizer):

    def __init__(self, preloaded_name, **kwargs):
        super(SciBertBaseTokenizerWrapper, self).__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(preloaded_name, from_pt=True)
        self.vocab = self.tokenizer.get_vocab()
        self.preloaded_name = preloaded_name

    def build_vocab(self, data, **kwargs):
        pass

    def get_info(self):
        info = super(SciBertBaseTokenizerWrapper, self).get_info()
        info['vocab_size'] = self.vocab_size
        return info

    @property
    def vocab_size(self):
        return len(self.vocab)

    def tokenize(self, text):
        return self.tokenizer(text)

    def convert_tokens_to_ids(self, tokens):
        tokenized = self.tokenizer(tokens)['input_ids']
        tokenized = [seq[1:-1] for seq in tokenized]
        return tokenized

    @classmethod
    def from_pretrained(cls, preloaded_name):
        tokenizer = SciBertBaseTokenizerWrapper(preloaded_name)
        return tokenizer

    def initialize_with_info(self, info):
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.preloaded_name)
        self.tokenizer.word_index = info['vocab']

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)


class TokenizerFactory(Factory):

    @classmethod
    def get_supported_values(cls):
        return {
            'scibert_tokenizer': SciBertBaseTokenizerWrapper
        }
