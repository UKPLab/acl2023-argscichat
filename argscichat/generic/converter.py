"""

Generic data converters

"""

import os

import numpy as np

from argscichat.generic.factory import Factory
from argscichat.generic.features import FeaturesFactory


class BaseConverter(object):

    def __init__(self, feature_class_name, feature_factory=None, max_tokens_limit=None):
        self.feature_factory = feature_factory if feature_factory is not None else FeaturesFactory
        self.feature_class = self.feature_factory.get_supported_values()[feature_class_name]
        self.max_tokens_limit = max_tokens_limit if max_tokens_limit else 100000
        self.data_sizes = {}

    def get_instance_args(self):
        return {
            'max_tokens_limit': self.max_tokens_limit,
            'data_sizes': self.data_sizes
        }

    def convert_data(self, examples, tokenizer, label_list, output_file,
                     is_training=False, has_labels=True, suffix='data', additional_data={}):
        raise NotImplementedError()

    def convert_example(self, example, tokenizer, label_list, has_labels=True):
        example_conversion_method = self.feature_class.from_example
        feature = example_conversion_method(example,
                                            label_list,
                                            has_labels=has_labels,
                                            tokenizer=tokenizer,
                                            conversion_args=self.get_conversion_args(),
                                            converter_args=self.get_instance_args(),
                                            wrapper_state={})
        return feature.to_tensor_format()

    def get_conversion_args(self):
        return vars(self)

    def set_conversion_args(self, conversion_args):
        for key, value in conversion_args.items():
            setattr(self, key, value)

    def training_preparation(self, examples, label_list, tokenizer):
        raise NotImplementedError()

    def save_conversion_args(self, filepath, prefix=None):
        if prefix is not None:
            filename = 'converter_info_{}.npy'.format(prefix)
        else:
            filename = 'converter_info.npy'
        filepath = os.path.join(filepath, filename)
        np.save(filepath, self.get_conversion_args())

    def save_instance_args(self, filepath, prefix=None):
        if prefix is not None:
            filename = 'converter_instance_info_{}.npy'.format(prefix)
        else:
            filename = 'converter_instance_info.npy'
        filepath = os.path.join(filepath, filename)
        np.save(filepath, self.get_instance_args())

    @staticmethod
    def load_conversion_args(filepath, prefix=None):
        if prefix:
            filename = 'converter_info_{}.npy'.format(prefix)
        else:
            filename = 'converter_info.npy'
        filepath = os.path.join(filepath, filename)
        return np.load(filepath, allow_pickle=True).item()

    def load_instance_args(self, filepath, prefix=None):
        if prefix:
            filename = 'converter_instance_info_{}.npy'.format(prefix)
        else:
            filename = 'converter_instance_info.npy'
        filepath = os.path.join(filepath, filename)
        loaded_args = np.load(filepath, allow_pickle=True).item()
        for key, value in loaded_args.items():
            setattr(self, key, value)


# Populate with generic converters (that do not require a specific framework)
class ConverterFactory(Factory):

    @classmethod
    def get_supported_values(cls):
        return {
        }
