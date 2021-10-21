import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from argscichat.generic.converter import BaseConverter, ConverterFactory
from argscichat.generic.features import BaseFeatures
from argscichat.generic.examples import ExampleList
from argscichat.tf_scripts.features import TFFeaturesFactory
from argscichat.utility.log_utils import Logger
from argscichat.utility.python_utils import merge

logger = Logger.get_logger(__name__)


class TFBaseConverter(BaseConverter):

    def __init__(self, feature_class, max_tokens_limit=None):
        super(TFBaseConverter, self).__init__(feature_class_name=feature_class,
                                              feature_factory=TFFeaturesFactory,
                                              max_tokens_limit=max_tokens_limit)

    def convert_data(self, examples, tokenizer, label_list, output_file,
                     is_training=False, has_labels=True, suffix='data', additional_data={}):

        assert issubclass(self.feature_class, BaseFeatures)

        if is_training:
            logger.info('Retrieving training set info...this may take a while...')
            self.training_preparation(examples=examples,
                                      label_list=label_list,
                                      tokenizer=tokenizer)

        example_conversion_method = self.feature_class.from_example
        example_feature_method = self.feature_class.get_feature_records

        writer = tf.io.TFRecordWriter(output_file)

        self.data_sizes[suffix] = len(examples)

        for ex_index, example in enumerate(tqdm(examples, leave=True, position=0)):
            if 'checkpoint' in additional_data and additional_data['checkpoint'] is not None\
                    and ex_index % additional_data['checkpoint'] == 0:
                logger.info('Writing example {0} of {1}'.format(ex_index, len(examples)))

            feature = example_conversion_method(example,
                                                label_list,
                                                has_labels=has_labels,
                                                tokenizer=tokenizer,
                                                conversion_args=self.get_conversion_args(),
                                                converter_args=self.get_instance_args(),
                                                wrapper_state=examples.get_added_state())

            features = example_feature_method(feature)

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())

        writer.close()

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
        return {
            'max_seq_length': self.max_seq_length,
            'label_map': self.label_map,
        }

    # TODO: we can do better here by defining attribute-defining functions: one for each conversion_args key
    def training_preparation(self, examples, label_list, tokenizer):

        assert isinstance(examples, ExampleList)

        example_conversion_method = self.feature_class.convert_example

        max_seq_length = None
        for example in tqdm(examples):
            features = example_conversion_method(example, label_list, tokenizer,
                                                 converter_args=self.get_instance_args(),
                                                 wrapper_state=examples.get_added_state())
            text_ids, label_id, label_map = features
            features_ids_len = len(text_ids)

            if max_seq_length is None:
                max_seq_length = features_ids_len
            elif max_seq_length < features_ids_len <= self.max_tokens_limit:
                max_seq_length = features_ids_len

        self.label_map = label_map
        self.max_seq_length = max_seq_length

    def save_conversion_args(self, filepath, prefix=None):
        if prefix:
            filename = 'converter_info_{}.npy'.format(prefix)
        else:
            filename = 'converter_info.npy'
        filepath = os.path.join(filepath, filename)
        np.save(filepath, self.get_conversion_args())

    def save_instance_args(self, filepath, prefix=None):
        if prefix:
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


class TFBaseTokensConverter(TFBaseConverter):

    def get_conversion_args(self):
        conversion_args = super(TFBaseTokensConverter, self).get_conversion_args()
        conversion_args['label_list'] = self.label_list

        return conversion_args

    def training_preparation(self, examples, label_list, tokenizer):

        assert isinstance(examples, ExampleList)

        max_seq_length = None

        for example in tqdm(examples):

            features = self.feature_class.convert_example(example, label_list, tokenizer=tokenizer,
                                                          converter_args=self.get_instance_args(),
                                                          wrapper_state=examples.get_added_state())
            input_ids, label_id = features
            features_ids_len = len(input_ids)

            if max_seq_length is None:
                max_seq_length = features_ids_len
            elif max_seq_length < features_ids_len <= self.max_tokens_limit:
                max_seq_length = features_ids_len

        self.max_seq_length = min(max_seq_length, self.max_tokens_limit)
        self.label_map = label_list.get_labels_mapping()
        self.label_list = label_list


class TFTransformerTokensConverter(TFBaseConverter):

    def get_conversion_args(self):
        conversion_args = super(TFTransformerTokensConverter, self).get_conversion_args()
        conversion_args['label_list'] = self.label_list

        return conversion_args

    def training_preparation(self, examples, label_list, tokenizer):

        assert isinstance(examples, ExampleList)

        max_seq_length = None

        for example in tqdm(examples):

            features = self.feature_class.convert_example(example, label_list, tokenizer=tokenizer,
                                                          converter_args=self.get_instance_args(),
                                                          wrapper_state=examples.get_added_state())
            input_ids, attention_mask, label_id = features
            features_ids_len = len(input_ids)

            if max_seq_length is None:
                max_seq_length = features_ids_len
            elif max_seq_length < features_ids_len <= self.max_tokens_limit:
                max_seq_length = features_ids_len

        self.max_seq_length = min(max_seq_length, self.max_tokens_limit)
        self.label_map = label_list.get_labels_mapping()
        self.label_list = label_list


class TFBaseComponentsConverter(TFBaseConverter):

    def get_conversion_args(self):
        conversion_args = super(TFBaseComponentsConverter, self).get_conversion_args()
        conversion_args['label_list'] = self.label_list

        return conversion_args

    def training_preparation(self, examples, label_list, tokenizer):

        assert isinstance(examples, ExampleList)

        max_seq_length = None

        for example in tqdm(examples):

            features = self.feature_class.convert_example(example, label_list, tokenizer=tokenizer,
                                                          converter_args=self.get_instance_args(),
                                                          wrapper_state=examples.get_added_state())
            source_ids, target_ids, label_id = features
            features_ids_len = max(len(source_ids), len(target_ids))

            if max_seq_length is None:
                max_seq_length = features_ids_len
            elif max_seq_length < features_ids_len <= self.max_tokens_limit:
                max_seq_length = features_ids_len

        self.max_seq_length = min(max_seq_length, self.max_tokens_limit)
        self.label_map = label_list.get_labels_mapping()
        self.label_list = label_list


class TFTransformerComponentsConverter(TFBaseConverter):

    def get_conversion_args(self):
        conversion_args = super(TFTransformerComponentsConverter, self).get_conversion_args()
        conversion_args['label_list'] = self.label_list
        conversion_args['max_digits_length'] = self.max_digits_length

        return conversion_args

    def training_preparation(self, examples, label_list, tokenizer):

        assert isinstance(examples, ExampleList)

        max_seq_length = None
        max_digits_length = None

        for example in tqdm(examples):

            features = self.feature_class.convert_example(example, label_list, tokenizer=tokenizer,
                                                          converter_args=self.get_instance_args(),
                                                          wrapper_state=examples.get_added_state())
            source_input_ids, source_attention_mask, target_input_ids, target_attention_mask, distance, label_id = features
            features_ids_len = max(len(source_input_ids), len(target_input_ids))

            if features_ids_len <= self.max_tokens_limit:
                if max_seq_length is None:
                    max_seq_length = features_ids_len
                elif max_seq_length < features_ids_len:
                    max_seq_length = features_ids_len

                if max_digits_length is None:
                    max_digits_length = len(distance)
                elif max_digits_length < len(distance):
                    max_digits_length = len(distance)

        self.max_seq_length = min(max_seq_length, self.max_tokens_limit)
        self.max_digits_length = max_digits_length
        self.label_map = label_list.get_labels_mapping()
        self.label_list = label_list


class TFConverterFactory(ConverterFactory):

    @classmethod
    def get_supported_values(cls):
        supported_values = super(TFConverterFactory, cls).get_supported_values()
        return merge(supported_values, {
            'tf_base_converter': TFBaseConverter,

            'tf_base_tokens_converter': TFBaseTokensConverter,
            'tf_transformer_tokens_converter': TFTransformerTokensConverter,

            'tf_transformer_components_converter': TFTransformerComponentsConverter,
            'tf_base_components_converter': TFBaseComponentsConverter,
        })
