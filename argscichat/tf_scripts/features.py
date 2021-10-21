"""

Simple wrappers for each dataset example

"""

from collections import OrderedDict

import tensorflow as tf

from argscichat.generic.examples import TextExample, TokensExample, PairedTextExample
from argscichat.generic.features import BaseFeatures, FeaturesFactory
from argscichat.utility.log_utils import Logger
from argscichat.utility.python_utils import merge

logger = Logger.get_logger(__name__)


# Utility

def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f


# Features


class TFBaseFeatures(BaseFeatures):

    @classmethod
    def _retrieve_default_label_mappings(cls, mappings, conversion_args, converter_args=None, has_labels=True):
        label_list = conversion_args['label_list']

        if has_labels:
            for label in label_list:
                mappings[label.name] = tf.io.FixedLenFeature([1], tf.int64)

        return mappings

    @classmethod
    def _retrieve_default_label_feature_records(cls, feature, features, converter_args=None):
        if feature.label_id is not None:
            # single label group
            if type(feature.label_id) == OrderedDict:
                for key, value in feature.label_id.items():
                    if type(value) == list:
                        features[key] = create_int_feature(value)
                    else:
                        features[key] = create_int_feature([value])

            # sequence label groups
            else:
                keys = feature.label_id[0].keys()
                for key in keys:
                    key_values = [item[key] for item in feature.label_id]
                    features[key] = create_int_feature(key_values)

        return features


class TFTextFeatures(TFBaseFeatures):

    def __init__(self, text_ids, label_id):
        self.text_ids = text_ids
        self.label_id = label_id

    @classmethod
    def get_mappings(cls, conversion_args, converter_args=None, has_labels=True):
        max_seq_length = conversion_args['max_seq_length']

        mappings = dict()
        mappings['text_ids'] = tf.io.FixedLenFeature([max_seq_length], tf.int64)

        mappings = cls._retrieve_default_label_mappings(mappings=mappings,
                                                        conversion_args=conversion_args,
                                                        converter_args=converter_args,
                                                        has_labels=has_labels)

        return mappings

    @classmethod
    def get_feature_records(cls, feature, converter_args=None):
        features = OrderedDict()
        features['text_ids'] = create_int_feature(feature.text_ids)

        features = cls._retrieve_default_label_feature_records(feature=feature,
                                                               features=features,
                                                               converter_args=converter_args)

        return features

    @classmethod
    def get_dataset_selector(cls, label_list):
        def _selector(record):
            x = {
                'text': record['text_ids'],
            }
            return cls._retrieve_default_label_dataset_selector(x, record, label_list)

        return _selector

    @classmethod
    def convert_example(cls, example, label_list, tokenizer, has_labels=True, converter_args=None, conversion_args=None,
                        wrapper_state={}):
        label_id, label_map = cls._convert_labels(example_label=example.label,
                                                  label_list=label_list,
                                                  has_labels=has_labels,
                                                  converter_args=converter_args)

        tokens = tokenizer.tokenize(example.text)
        text_ids = tokenizer.convert_tokens_to_ids(tokens)

        return text_ids, label_id, label_map

    @classmethod
    def from_example(cls, example, label_list, tokenizer, conversion_args, has_labels=True, converter_args=None,
                     wrapper_state={}):
        if not isinstance(example, TextExample):
            raise AttributeError('Expected TextExample instance, got: {}'.format(type(example)))

        max_seq_length = conversion_args['max_seq_length']

        text_ids, label_id, label_map = TFTextFeatures.convert_example(example=example, label_list=label_list,
                                                                       tokenizer=tokenizer, has_labels=has_labels,
                                                                       converter_args=converter_args,
                                                                       conversion_args=conversion_args,
                                                                       wrapper_state=wrapper_state)

        # Padding
        text_ids += [0] * (max_seq_length - len(text_ids))
        text_ids = text_ids[:max_seq_length]

        assert len(text_ids) == max_seq_length

        feature = TFTextFeatures(text_ids=text_ids, label_id=label_id)
        return feature


class TFBaseTokensFeatures(TFBaseFeatures):

    def __init__(self, input_ids, label_id):
        self.input_ids = input_ids
        self.label_id = label_id

    def to_tensor_format(self):
        to_return = {
            'input_ids': tf.expand_dims(tf.convert_to_tensor(self.input_ids), 0),
        }

        if self.label_id is not None:
            to_return['label_id'] = tf.expand_dims(tf.convert_to_tensor(self.label_id), 0)

        return to_return

    @classmethod
    def _retrieve_default_label_mappings(cls, mappings, conversion_args, converter_args=None, has_labels=True):
        label_list = conversion_args['label_list']
        max_sequence_length = conversion_args['max_seq_length']

        if has_labels:
            for label in label_list:
                mappings[label.name] = tf.io.FixedLenFeature([max_sequence_length], tf.int64)

        return mappings

    @classmethod
    def get_mappings(cls, conversion_args, converter_args=None, has_labels=True):
        max_seq_length = conversion_args['max_seq_length']

        mappings = {
            'input_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
        }

        mappings = cls._retrieve_default_label_mappings(mappings=mappings,
                                                        conversion_args=conversion_args,
                                                        converter_args=converter_args,
                                                        has_labels=has_labels)

        return mappings

    @classmethod
    def get_feature_records(cls, feature, converter_args=None):

        features = OrderedDict()
        features['input_ids'] = create_int_feature(feature.input_ids)

        features = cls._retrieve_default_label_feature_records(feature=feature,
                                                               features=features,
                                                               converter_args=converter_args)

        return features

    @classmethod
    def get_dataset_selector(cls, label_list):

        def _selector(record):
            x = {
                'input_ids': record['input_ids'],
            }
            return cls._retrieve_default_label_dataset_selector(x, record, label_list)

        return _selector

    @classmethod
    def convert_example(cls, example, label_list, tokenizer, has_labels=True, converter_args=None, conversion_args=None,
                        wrapper_state={}):

        label_id = cls._convert_labels(example_label=example.tokens_labels,
                                       label_list=label_list,
                                       has_labels=has_labels,
                                       converter_args=converter_args)

        input_ids = tokenizer.tokenize(example.tokens)
        input_ids = tokenizer.convert_tokens_to_ids(input_ids)

        if label_id:
            ext_label_id = [[label for _ in tokens] for tokens, label in zip(input_ids, label_id)]
            ext_label_id = [item for seq in ext_label_id for item in seq]
        else:
            ext_label_id = label_id

        input_ids = [item for seq in input_ids for item in seq]

        if label_id:
            assert len(input_ids) == len(ext_label_id)

        return input_ids, ext_label_id

    @classmethod
    def from_example(cls, example, label_list, tokenizer, conversion_args, has_labels=True, converter_args=None,
                     wrapper_state={}):

        if not isinstance(example, TokensExample):
            raise AttributeError('Expected TokensExample instance, got: {}'.format(type(example)))

        max_seq_length = conversion_args['max_seq_length']

        input_ids, label_id = cls.convert_example(example=example,
                                                  label_list=label_list,
                                                  tokenizer=tokenizer,
                                                  has_labels=has_labels,
                                                  converter_args=converter_args,
                                                  conversion_args=conversion_args,
                                                  wrapper_state=wrapper_state)

        # Padding

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # attention_mask = [1 if converter_args['mask_padding_with_zero'] else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + [0] * padding_length

        if label_id:
            label_id = label_id + [label_id[-1] for _ in range(padding_length)]
            label_id = label_id[:max_seq_length]

        input_ids = input_ids[:max_seq_length]

        assert len(input_ids) == max_seq_length, "Error with input length {} vs {}".format(len(input_ids),
                                                                                           max_seq_length)

        feature = cls(input_ids=input_ids, label_id=label_id)
        return feature


class TFTransformerTokensFeatures(TFBaseFeatures):

    def __init__(self, input_ids, attention_mask, label_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label_id = label_id

    def to_tensor_format(self):
        to_return = {
            'input_ids': tf.expand_dims(tf.convert_to_tensor(self.input_ids), 0),
            'attention_mask': tf.expand_dims(tf.convert_to_tensor(self.attention_mask), 0)
        }

        if self.label_id is not None:
            to_return['label_id'] = tf.expand_dims(tf.convert_to_tensor(self.label_id), 0)

        return to_return

    @classmethod
    def _retrieve_default_label_mappings(cls, mappings, conversion_args, converter_args=None, has_labels=True):
        label_list = conversion_args['label_list']
        max_sequence_length = conversion_args['max_seq_length']

        if has_labels:
            for label in label_list:
                mappings[label.name] = tf.io.FixedLenFeature([max_sequence_length], tf.int64)

        return mappings

    @classmethod
    def get_mappings(cls, conversion_args, converter_args=None, has_labels=True):
        max_seq_length = conversion_args['max_seq_length']

        mappings = {
            'input_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
            'attention_mask': tf.io.FixedLenFeature([max_seq_length], tf.int64),
        }

        mappings = cls._retrieve_default_label_mappings(mappings=mappings,
                                                        conversion_args=conversion_args,
                                                        converter_args=converter_args,
                                                        has_labels=has_labels)

        return mappings

    @classmethod
    def get_feature_records(cls, feature, converter_args=None):

        features = OrderedDict()
        features['input_ids'] = create_int_feature(feature.input_ids)
        features['attention_mask'] = create_int_feature(feature.attention_mask)

        features = cls._retrieve_default_label_feature_records(feature=feature,
                                                               features=features,
                                                               converter_args=converter_args)

        return features

    @classmethod
    def get_dataset_selector(cls, label_list):

        def _selector(record):
            x = {
                'input_ids': record['input_ids'],
                'attention_mask': record['attention_mask'],
            }
            return cls._retrieve_default_label_dataset_selector(x, record, label_list)

        return _selector

    @classmethod
    def convert_example(cls, example, label_list, tokenizer, has_labels=True, converter_args=None, conversion_args=None,
                        wrapper_state={}):

        label_id = cls._convert_labels(example_label=example.tokens_labels,
                                       label_list=label_list,
                                       has_labels=has_labels,
                                       converter_args=converter_args)

        inputs = tokenizer.tokenizer(example.tokens)
        start_token, end_token = inputs['input_ids'][0][0], inputs['input_ids'][0][-1]
        merged_input_ids = [item[1:-1] for item in inputs['input_ids']]
        merged_attention_mask = [item[1:-1] for item in inputs['attention_mask']]

        if has_labels:
            adjusted_label_id = []
            for index in range(len(label_id)):
                if len(merged_input_ids[index]) == 1:
                    adjusted_label_id.append(label_id[index])
                else:
                    for sub_token_idx in range(len(merged_input_ids[index])):
                        adjusted_label_id.append(label_id[index])
            adjusted_label_id = [adjusted_label_id[0]] + adjusted_label_id + [adjusted_label_id[-1]]
        else:
            adjusted_label_id = label_id

        merged_input_ids = [item for seq in merged_input_ids for item in seq]
        merged_attention_mask = [item for seq in merged_attention_mask for item in seq]

        merged_input_ids = [start_token] + merged_input_ids + [end_token]
        merged_attention_mask = [1] + merged_attention_mask + [1]

        if has_labels:
            assert len(merged_input_ids) == len(merged_attention_mask) == len(adjusted_label_id)
        else:
            assert len(merged_input_ids) == len(merged_attention_mask)

        return merged_input_ids, merged_attention_mask, adjusted_label_id

    @classmethod
    def from_example(cls, example, label_list, tokenizer, conversion_args, has_labels=True, converter_args=None,
                     wrapper_state={}):

        if not isinstance(example, TokensExample):
            raise AttributeError('Expected TokensExample instance, got: {}'.format(type(example)))

        max_seq_length = conversion_args['max_seq_length']

        input_ids, attention_mask, label_id = cls.convert_example(example=example,
                                                                  label_list=label_list,
                                                                  tokenizer=tokenizer,
                                                                  has_labels=has_labels,
                                                                  converter_args=converter_args,
                                                                  conversion_args=conversion_args,
                                                                  wrapper_state=wrapper_state)

        # Padding

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # attention_mask = [1 if converter_args['mask_padding_with_zero'] else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + [0] * padding_length
        attention_mask = attention_mask + [0] * padding_length

        if has_labels:
            label_id = label_id + [label_id[-1] for _ in range(padding_length)]
            label_id = label_id[:max_seq_length]

        input_ids = input_ids[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]

        assert len(input_ids) == max_seq_length, "Error with input length {} vs {}".format(len(input_ids),
                                                                                           max_seq_length)
        assert len(attention_mask) == max_seq_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                                max_seq_length)

        feature = cls(input_ids=input_ids, label_id=label_id,
                      attention_mask=attention_mask)
        return feature


class TFBaseComponentsFeatures(TFBaseFeatures):

    def __init__(self, source_ids, target_ids, distance, label_id):
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.distance = distance
        self.label_id = label_id

    @classmethod
    def get_mappings(cls, conversion_args, converter_args=None, has_labels=True):
        max_seq_length = conversion_args['max_seq_length']
        max_distance_digits = conversion_args['max_distance_digits']

        mappings = {
            'source_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
            'target_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
            'distance': tf.io.FixedLenFeature([max_distance_digits], tf.int64)
        }

        mappings = cls._retrieve_default_label_mappings(mappings=mappings,
                                                        conversion_args=conversion_args,
                                                        converter_args=converter_args,
                                                        has_labels=has_labels)

        return mappings

    @classmethod
    def get_feature_records(cls, feature, converter_args=None):
        features = OrderedDict()
        features['source_ids'] = create_int_feature(feature.source_ids)
        features['target_ids'] = create_int_feature(feature.target_ids)
        features['distance'] = create_int_feature(feature.distance)

        features = cls._retrieve_default_label_feature_records(feature=feature,
                                                               features=features,
                                                               converter_args=converter_args)

        return features

    @classmethod
    def get_dataset_selector(cls, label_list):
        def _selector(record):
            x = {
                'source_ids': record['source_ids'],
                'target_ids': record['target_ids'],
                'distance': record['distance']
            }
            return cls._retrieve_default_label_dataset_selector(x, record, label_list)

        return _selector

    @classmethod
    def convert_example(cls, example, label_list, tokenizer, has_labels=True, converter_args=None, conversion_args=None,
                        wrapper_state={}):
        label_id = cls._convert_labels(example_label=example.labels,
                                       label_list=label_list,
                                       has_labels=has_labels,
                                       converter_args=converter_args)

        source_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example.source))
        target_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example.target))

        return source_ids, target_ids, example.distance, label_id

    @classmethod
    def from_example(cls, example, label_list, tokenizer, conversion_args, has_labels=True, converter_args=None,
                     wrapper_state={}):
        if not isinstance(example, PairedTextExample):
            raise AttributeError('Expected PairedTextExample instance, got: {}'.format(type(example)))

        max_seq_length = conversion_args['max_seq_length']
        max_distance_digits = conversion_args['max_distance_digits']

        source_ids, target_ids, distance, label_id = cls.convert_example(example=example,
                                                                         label_list=label_list,
                                                                         tokenizer=tokenizer,
                                                                         has_labels=has_labels,
                                                                         converter_args=converter_args,
                                                                         conversion_args=conversion_args,
                                                                         wrapper_state=wrapper_state)

        # Padding

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # attention_mask = [1 if converter_args['mask_padding_with_zero'] else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - min(len(source_ids), len(target_ids))

        source_ids = source_ids + [1] * padding_length
        target_ids = target_ids + [1] * padding_length

        source_ids = source_ids[:max_seq_length]
        target_ids = target_ids[:max_seq_length]

        distance = [0] * (max_distance_digits - len(distance)) + distance
        distance = distance[:max_distance_digits]

        assert len(source_ids) == max_seq_length, "Error with source input length {0} vs {1}".format(
            len(source_ids),
            max_seq_length)
        assert len(target_ids) == max_seq_length, "Error with target input length {0} vs {1}".format(
            len(target_ids),
            max_seq_length)
        assert len(distance) == max_distance_digits, "Error with distance digits length {0} vs {1}".format(
            len(distance), max_distance_digits)

        feature = cls(source_ids=source_ids, target_ids=target_ids, distance=distance, label_id=label_id)
        return feature


class TFTransformerComponentsFeatures(TFBaseFeatures):

    def __init__(self, source_input_ids, source_attention_mask, target_input_ids, target_attention_mask, distance,
                 label_id):
        self.source_input_ids = source_input_ids
        self.source_attention_mask = source_attention_mask
        self.target_input_ids = target_input_ids
        self.target_attention_mask = target_attention_mask
        self.distance = distance
        self.label_id = label_id

    @classmethod
    def get_mappings(cls, conversion_args, converter_args=None, has_labels=True):
        max_seq_length = conversion_args['max_seq_length']
        max_digits_length = conversion_args['max_digits_length']

        mappings = {
            'source_input_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
            'source_attention_mask': tf.io.FixedLenFeature([max_seq_length], tf.int64),
            'target_input_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
            'target_attention_mask': tf.io.FixedLenFeature([max_seq_length], tf.int64),
            'distance': tf.io.FixedLenFeature([max_digits_length], tf.int64),
        }

        mappings = cls._retrieve_default_label_mappings(mappings=mappings,
                                                        conversion_args=conversion_args,
                                                        converter_args=converter_args,
                                                        has_labels=has_labels)

        return mappings

    @classmethod
    def get_feature_records(cls, feature, converter_args=None):
        features = OrderedDict()
        features['source_input_ids'] = create_int_feature(feature.source_input_ids)
        features['source_attention_mask'] = create_int_feature(feature.source_attention_mask)
        features['target_input_ids'] = create_int_feature(feature.target_input_ids)
        features['target_attention_mask'] = create_int_feature(feature.target_attention_mask)
        features['distance'] = create_int_feature(feature.distance)

        features = cls._retrieve_default_label_feature_records(feature=feature,
                                                               features=features,
                                                               converter_args=converter_args)

        return features

    @classmethod
    def get_dataset_selector(cls, label_list):
        def _selector(record):
            x = {
                'source_input_ids': record['source_input_ids'],
                'source_attention_mask': record['source_attention_mask'],
                'target_input_ids': record['target_input_ids'],
                'target_attention_mask': record['target_attention_mask'],
                'distance': record['distance']
            }
            return cls._retrieve_default_label_dataset_selector(x, record, label_list)

        return _selector

    @classmethod
    def convert_example(cls, example, label_list, tokenizer, has_labels=True, converter_args=None, conversion_args=None,
                        wrapper_state={}):
        label_id = cls._convert_labels(example_label=example.labels,
                                       label_list=label_list,
                                       has_labels=has_labels,
                                       tokenizer=tokenizer,
                                       converter_args=converter_args)

        source_inputs = tokenizer.tokenizer(example.source)
        source_input_ids, source_attention_mask = source_inputs['input_ids'], source_inputs['attention_mask']

        target_inputs = tokenizer.tokenizer(example.target)
        target_input_ids, target_attention_mask = target_inputs['input_ids'], target_inputs['attention_mask']

        return source_input_ids, source_attention_mask, target_input_ids, target_attention_mask, example.distance, label_id

    @classmethod
    def from_example(cls, example, label_list, tokenizer, conversion_args, has_labels=True, converter_args=None,
                     wrapper_state={}):
        if not isinstance(example, PairedTextExample):
            raise AttributeError('Expected PairedTextExample instance, got: {}'.format(type(example)))

        max_seq_length = conversion_args['max_seq_length']
        max_digits_length = conversion_args['max_digits_length']

        source_input_ids, source_attention_mask, \
        target_input_ids, target_attention_mask, distance, label_id = cls.convert_example(example=example,
                                                                                          label_list=label_list,
                                                                                          tokenizer=tokenizer,
                                                                                          has_labels=has_labels,
                                                                                          converter_args=converter_args,
                                                                                          conversion_args=conversion_args,
                                                                                          wrapper_state=wrapper_state)

        # Padding

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # attention_mask = [1 if converter_args['mask_padding_with_zero'] else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - min(len(source_input_ids), len(target_input_ids))

        source_input_ids = source_input_ids + [1] * padding_length
        target_input_ids = target_input_ids + [1] * padding_length

        source_attention_mask = source_attention_mask + [1] * padding_length
        target_attention_mask = target_attention_mask + [1] * padding_length

        source_input_ids = source_input_ids[:max_seq_length]
        target_input_ids = target_input_ids[:max_seq_length]

        source_attention_mask = source_attention_mask[:max_seq_length]
        target_attention_mask = target_attention_mask[:max_seq_length]

        distance = [0] * (max_digits_length - len(distance)) + distance
        distance = distance[:max_digits_length]

        assert len(source_input_ids) == max_seq_length, "Error with source input length {} vs {}".format(
            len(source_input_ids),
            max_seq_length)
        assert len(target_input_ids) == max_seq_length, "Error with target input length {} vs {}".format(
            len(target_input_ids),
            max_seq_length)

        assert len(source_attention_mask) == max_seq_length, "Error with source attention mask length {} vs {}".format(
            len(source_attention_mask),
            max_seq_length)
        assert len(target_attention_mask) == max_seq_length, "Error with target attention mask length {} vs {}".format(
            len(target_attention_mask),
            max_seq_length)

        assert len(distance) == max_digits_length, "Error with distance digits length {0} vs {1}".format(len(distance),
                                                                                                         max_digits_length)

        feature = cls(source_input_ids=source_input_ids, source_attention_mask=source_attention_mask,
                      target_input_ids=target_input_ids, target_attention_mask=target_attention_mask,
                      label_id=label_id, distance=distance)
        return feature


class TFFeaturesFactory(FeaturesFactory):

    @classmethod
    def get_supported_values(cls):
        supported_values = super(TFFeaturesFactory, cls).get_supported_values()
        return merge(supported_values, {
            'tf_text_features': TFTextFeatures,

            'tf_base_tokens_features': TFBaseTokensFeatures,
            'tf_transformer_tokens_features': TFTransformerTokensFeatures,

            'tf_base_components_features': TFBaseComponentsFeatures,
            'tf_transformer_components_features': TFTransformerComponentsFeatures,
        })
