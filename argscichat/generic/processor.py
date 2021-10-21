"""

Generic data processors

"""

from ast import literal_eval
from collections import OrderedDict

import pandas as pd
from tqdm import tqdm

from argscichat.generic.examples import ExampleList, \
    TextExample, TokensExample, PairedTextExample
from argscichat.generic.factory import Factory
from argscichat.utility import preprocessing_utils


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, loader_info, filter_names=None, retrieve_label=True):
        self.loader_info = loader_info
        self.filter_names = filter_names if filter_names is not None else preprocessing_utils.filter_methods
        self.retrieve_label = retrieve_label

    def _retrieve_default_label(self, row):
        if self.retrieve_label:
            label = OrderedDict([(label.name, row[label.name]) for label in self.loader_info['label']])
        else:
            label = None

        return label

    def get_train_examples(self, filepath=None, ids=None, data=None):
        """Gets a collection of `Example`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, filepath=None, ids=None, data=None):
        """Gets a collection of `Example`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, filepath=None, ids=None, data=None):
        """Gets a collection of `Example`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return self.loader_info['label'] if self.retrieve_label else None

    def get_processor_name(self):
        """Gets the string identifier of the processor."""
        return self.loader_info['data_name']

    def wrap_single_example(self, df_item):
        raise NotImplementedError()

    @classmethod
    def read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        df = pd.read_csv(input_file)
        return df


class TextProcessor(DataProcessor):

    def _get_examples_from_df(self, df, suffix):
        examples = ExampleList()
        for row_id, row in tqdm(df.iterrows()):
            text_data_key = self.loader_info['data_keys']['text']
            text = preprocessing_utils.filter_line(row[text_data_key], function_names=self.filter_names)
            label = self._retrieve_default_label(row)
            example = TextExample(text=text, label=label)
            examples.append(example)

        return examples

    def get_train_examples(self, filepath=None, ids=None, data=None):

        if filepath is None and data is None:
            raise AttributeError('Either filepath or data must be not None')

        if not isinstance(data, pd.DataFrame):
            raise AttributeError('Data must be a pandas.DataFrame')

        if filepath is not None:
            df = self.read_csv(filepath)
            return self._get_examples_from_df(df, suffix='train')
        else:
            if ids is not None:
                data = data.iloc[ids]

            return self._get_examples_from_df(data, suffix='train')

    def get_dev_examples(self, filepath=None, ids=None, data=None):

        if filepath is None and data is None:
            raise AttributeError('Either filepath or data must be not None')

        if not isinstance(data, pd.DataFrame):
            raise AttributeError('Data must be a pandas.DataFrame')

        if filepath is not None:
            df = self.read_csv(filepath)
            return self._get_examples_from_df(df, suffix='dev')
        else:
            if ids is not None:
                data = data.iloc[ids]

            return self._get_examples_from_df(data, suffix='dev')

    def get_test_examples(self, filepath=None, ids=None, data=None):

        if filepath is None and data is None:
            raise AttributeError('Either filepath or data must be not None')

        if not isinstance(data, pd.DataFrame):
            raise AttributeError('Data must be a pandas.DataFrame')

        if filepath is not None:
            df = self.read_csv(filepath)
            return self._get_examples_from_df(df, suffix='test')
        else:
            if ids is not None:
                data = data.iloc[ids]

            return self._get_examples_from_df(data, suffix='test')


class BaseTokensProcessor(TextProcessor):

    def _get_examples_from_df(self, df, suffix):
        examples = ExampleList()
        doc_key = self.loader_info['data_keys']['doc_id']
        grouped_df = df.groupby(doc_key)

        for group_idx, group in tqdm(grouped_df):
            text_data_key = self.loader_info['data_keys']['token']
            chunk_id_key = self.loader_info['data_keys']['chunk_id']
            chunks = group[chunk_id_key].values
            min_chunk_id, max_chunk_id = min(chunks), max(chunks)

            for chunk_id in range(min_chunk_id, max_chunk_id):
                chunk_group = group[group[chunk_id_key] == chunk_id]
                chunk_tokens = chunk_group[text_data_key].astype(str).values.tolist()
                chunk_labels = [self._retrieve_default_label(row) for _, row in chunk_group.iterrows()]

                example = TokensExample(tokens=chunk_tokens, tokens_labels=chunk_labels)
                examples.append(example)

        return examples

    def wrap_single_example(self, df_item):
        text_data_key = self.loader_info['data_keys']['token']
        tokens = df_item[text_data_key].values.tolist()
        labels = [self._retrieve_default_label(row) for _, row in df_item.iterrows()]

        example = TokensExample(tokens=tokens, tokens_labels=labels)
        return example


class BaseComponentsProcessor(TextProcessor):

    def _get_examples_from_df(self, df, suffix):
        examples = ExampleList()

        for row_idx, row in tqdm(df.iterrows()):
            source = row[self.loader_info['data_keys']['source']]
            target = row[self.loader_info['data_keys']['target']]
            distance = literal_eval(row[self.loader_info['data_keys']['distance']])
            labels = self._retrieve_default_label(row)

            example = PairedTextExample(source=source, target=target, labels=labels, distance=distance)
            examples.append(example)

        return examples


class ProcessorFactory(Factory):

    @classmethod
    def get_supported_values(cls):
        return {
            'text_processor': TextProcessor,
            'base_tokens_processor': BaseTokensProcessor,
            'base_components_processor': BaseComponentsProcessor
        }

