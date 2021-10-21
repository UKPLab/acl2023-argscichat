import os

import numpy as np
import pandas as pd

import argscichat.const_define as cd
from argscichat.generic.labels import ClassificationLabel, LabelList
from argscichat.utility.pickle_utils import load_pickle


def load_file_data(file_path):
    """
    Reads a file line by line.

    :param file_path: path to the file
    :return: list of sentences (string)
    """

    sentences = []

    with open(file_path, 'r') as f:
        for line in f:
            sentences.append(line)

    return sentences


def load_dataset(df_path):
    """
    Loads the ToS dataset given the DataFrame path.

    :param df_path: path to the .csv file
    :return pandas.DataFrame
    """

    df = pd.read_csv(df_path)
    return df


class DrInventorTokensLoader(object):

    def load(self):
        df_path = os.path.join(cd.LOCAL_DATASETS_DIR, 'drinventor-tokens', 'dataset_chunked.csv')
        df = load_dataset(df_path=df_path)
        train_df = df[df['Split'] == 'train']
        val_df = df[df['Split'] == 'val']
        test_df = df[df['Split'] == 'test']

        merged_labels = set(df['Merged_Arg_Label'].values)

        labels = LabelList(
            [
                ClassificationLabel(name='Merged_Arg_Label',
                                    values=list(merged_labels)),
            ]
        )

        return TrainAndTestDataHandle(train_data=train_df, validation_data=val_df, test_data=test_df,
                                      build_validation=False,
                                      data_name="DrInventorTokens",
                                      label=labels,
                                      data_keys={'token': 'Token', 'chunk_id': 'Chunk_Index', 'doc_id': 'Doc'})


class DrInventorComponentsLoader(object):

    def load(self, distinguish_reverse_pairs=False):
        base_path = os.path.join(cd.LOCAL_DATASETS_DIR, 'drinventor-components', '{}.pkl')
        train_df = load_pickle(base_path.format('train'))
        val_df = load_pickle(base_path.format('validation'))
        test_df = load_pickle(base_path.format('test'))

        # Count inv relations as None
        # Alternatively, keep them to see if performance improves
        if not distinguish_reverse_pairs:
            train_df.loc[train_df['relation_type'] == 'inv_supports', 'relation_type'] = 'link'
            val_df.loc[val_df['relation_type'] == 'inv_supports', 'relation_type'] = 'link'
            test_df.loc[test_df['relation_type'] == 'inv_supports', 'relation_type'] = 'link'

            train_df.loc[train_df['relation_type'] == 'inv_contradicts', 'relation_type'] = 'link'
            val_df.loc[val_df['relation_type'] == 'inv_contradicts', 'relation_type'] = 'link'
            test_df.loc[test_df['relation_type'] == 'inv_contradicts', 'relation_type'] = 'link'
        else:
            train_df.loc[train_df['relation_type'] == 'inv_supports', 'relation_type'] = 'inv_link'
            val_df.loc[val_df['relation_type'] == 'inv_supports', 'relation_type'] = 'inv_link'
            test_df.loc[test_df['relation_type'] == 'inv_supports', 'relation_type'] = 'inv_link'

            train_df.loc[train_df['relation_type'] == 'inv_contradicts', 'relation_type'] = 'inv_link'
            val_df.loc[val_df['relation_type'] == 'inv_contradicts', 'relation_type'] = 'inv_link'
            test_df.loc[test_df['relation_type'] == 'inv_contradicts', 'relation_type'] = 'inv_link'

        # Define link no-link task
        train_df.loc[train_df['relation_type'] == 'semantically_same', 'relation_type'] = 'link'
        val_df.loc[val_df['relation_type'] == 'semantically_same', 'relation_type'] = 'link'
        test_df.loc[test_df['relation_type'] == 'semantically_same', 'relation_type'] = 'link'

        train_df.loc[train_df['relation_type'] == 'supports', 'relation_type'] = 'link'
        val_df.loc[val_df['relation_type'] == 'supports', 'relation_type'] = 'link'
        test_df.loc[test_df['relation_type'] == 'supports', 'relation_type'] = 'link'

        train_df.loc[train_df['relation_type'] == 'contradicts', 'relation_type'] = 'link'
        val_df.loc[val_df['relation_type'] == 'contradicts', 'relation_type'] = 'link'
        test_df.loc[test_df['relation_type'] == 'contradicts', 'relation_type'] = 'link'

        component_types = list(set(train_df['source_type'].values))
        relation_types = list(set(train_df['relation_type'].values))

        labels = LabelList(
            [
                # ClassificationLabel(name='source_type',
                #                     values=component_types),
                # ClassificationLabel(name='target_type',
                #                     values=component_types),
                ClassificationLabel(name='relation_type',
                                    values=relation_types)
            ]
        )

        return TrainAndTestDataHandle(train_data=train_df, test_data=test_df, validation_data=val_df,
                                      data_name="DrInventorComponents",
                                      label=labels,
                                      build_validation=False,
                                      data_keys={'source': 'source_proposition', 'target': 'target_proposition',
                                                 'source_type': 'source_type', 'target_type': 'target_type',
                                                 'relation': 'relation_type', 'distance': 'Distance'})



class ArgSciChatModelComponentsPapersLoader(object):

    def load(self, threshold=None):

        if threshold is None:
            filename = 'papers_model_components_dataset.csv'
        else:
            filename = 'papers_model_components_dataset_{}.csv'.format(threshold)

        df_path = os.path.join(cd.LOCAL_DATASETS_DIR, 'arg_sci_chat', filename)
        df = pd.read_csv(df_path)

        data_keys = {'source': 'Source', 'target': 'Target', 'distance': 'Distance'}

        return UnseenDataHandle(data=df, data_name='ArgSciChatModelComponentsPapers', data_keys=data_keys)


class ArgSciChatTokensPapersLoader(object):

    def load(self):
        df_path = os.path.join(cd.LOCAL_DATASETS_DIR, 'arg_sci_chat', 'papers_token_dataset_chunked.csv')
        df = pd.read_csv(df_path)

        data_keys = {'token': 'Token', 'chunk_id': 'Chunk_Index', 'doc_id': 'Doc'}

        return UnseenDataHandle(data=df, data_name='ArgSciChatTokensPapers', data_keys=data_keys)


class DataLoaderFactory(object):
    supported_data_loaders = {
        'dr_inventor_tokens': DrInventorTokensLoader,
        'dr_inventor_components': DrInventorComponentsLoader,
        'arg_sci_chat_model_components_papers': ArgSciChatModelComponentsPapersLoader,
        'arg_sci_chat_tokens_papers': ArgSciChatTokensPapersLoader,
    }

    @staticmethod
    def factory(cl_type, **kwargs):
        """
        Returns an instance of specified type, built with given, if any, parameters.

        :param cl_type: string name of the classifier class (not case sensitive)
        :param kwargs: additional __init__ parameters
        :return: classifier instance
        """

        key = cl_type.lower()
        if DataLoaderFactory.supported_data_loaders[key]:
            return DataLoaderFactory.supported_data_loaders[key](**kwargs)
        else:
            raise ValueError('Bad type creation: {}'.format(cl_type))


#### Data Handles ####


class DataHandle(object):
    """
    General dataset wrapper. Additional behaviour can be expressed via attributes.
    """

    def __init__(self, data, data_name,
                 data_keys, label=None, has_fixed_test=False, test_data=None,
                 has_fixed_validation=False, val_data=None):
        self.data = data
        self.data_name = data_name
        self.has_fixed_test = has_fixed_test
        self.test_data = test_data
        self.has_fixed_validation = has_fixed_validation
        self.val_data = val_data
        self.label = label
        self.data_keys = data_keys

    def get_split(self, key_values, key=None, val_indexes=None, validation_percentage=None):
        if key is None:
            if self.data.index.name is None:
                self.data['index'] = np.arange(self.data.shape[0])
                self.data.set_index('index')
                key = 'index'
            else:
                key = self.data.index.name

            key_values = key_values.astype(self.data[key].values.dtype)

            if val_indexes is not None:
                val_indexes = val_indexes.astype(key_values.dtype)

        train_df = self.data[np.logical_not(self.data[key].isin(key_values))]

        if self.has_fixed_test:
            # self.data should not contain test_data
            test_df = self.test_data
            val_df = self.data[self.data[key].isin(key_values)]
        else:
            test_df = self.data[self.data[key].isin(key_values)]

            if self.has_fixed_validation:
                val_df = self.val_data
            else:
                if val_indexes is None:
                    assert validation_percentage is not None

                    validation_amount = int(len(train_df) * validation_percentage)
                    train_amount = len(train_df) - validation_amount
                    val_df = train_df[train_amount:]
                    train_df = train_df[:train_amount]
                else:
                    val_df = train_df[train_df[key].isin(val_indexes)]
                    train_df = train_df[np.logical_not(train_df[key].isin(val_indexes))]

        self.train_size = train_df.shape[0]
        self.val_size = val_df.shape[0]
        self.test_size = test_df.shape[0]

        return train_df, val_df, test_df

    def get_data(self, validation_percentage=None):
        raise NotImplementedError()

    def get_additional_info(self):
        return {
            'label': self.label,
            'data_keys': self.data_keys
        }


class TrainAndTestDataHandle(object):

    def __init__(self, train_data, test_data, data_name, label, data_keys,
                 validation_data=None, build_validation=True):
        self.train_data = train_data
        self.test_data = test_data
        self.data_name = data_name
        self.label = label
        self.validation_data = validation_data
        self.data_keys = data_keys
        self.build_validation = build_validation

    def get_data(self, validation_percentage=None):

        train_df = self.train_data
        test_df = self.test_data

        if self.build_validation:
            if validation_percentage is None:
                val_df = self.validation_data
            else:
                validation_amount = int(len(train_df) * validation_percentage)
                train_amount = len(train_df) - validation_amount
                val_df = train_df[train_amount:]
                train_df = train_df[:train_amount]
        else:
            val_df = self.validation_data

        return train_df, val_df, test_df

    def get_additional_info(self):
        return {
            'label': self.label,
            'data_keys': self.data_keys
        }


class UnseenDataHandle(object):

    def __init__(self, data, data_name, data_keys):
        self.data = data
        self.data_name = data_name
        self.data_keys = data_keys

    def get_data(self):
        return self.data

    def get_additional_info(self):
        return {
            'data_keys': self.data_keys
        }
