
import inspect
from argscichat.utility.json_utils import load_json, save_json
from argscichat.generic.data_loader import DataLoaderFactory
from argscichat.utility.python_utils import merge
from datetime import datetime
import os
import argscichat.const_define as cd
from copy import deepcopy
from functools import reduce
import shutil


class ConfigHelper(object):

    def __init__(self, test_type):
        self.test_type = test_type
        self.config_filename = self._get_test_name_from_config(test_type)
        self.config_folder = self._get_folder_from_config(test_type)
        self.config = load_json(os.path.join(cd.CONFIGS_DIR, self.config_filename))

        # Loading existing model
        if 'test_name' in self.config:
            model_path = os.path.join(self._get_folder_from_config(self['training_test']),
                                      self['model_type'],
                                      self['test_name'])
            self.path = model_path
        else:
            self.path = cd.CONFIGS_DIR

        self.training_config = None
        self.model_config = None
        self.data_loader_config = None
        self.callbacks_config = None
        self.model_name_additional_suffixes = {}

    def __getitem__(self, item):
        return self.config[item]

    def __contains__(self, item):
        return item in self.config

    def _get_test_name_from_config(self, test_type):
        return cd.TEST_INFO[test_type]['filename']

    def _get_folder_from_config(self, test_type):
        return cd.TEST_INFO[test_type]['folder']

    def extract_method_args_from_config(self, method):
        method_args = list(inspect.signature(method).parameters.keys())
        configs = [self.config]
        if self.training_config is not None:
            configs.append(self.training_config)
        if self.model_config is not None:
            configs.append(self.model_config)
        if self.data_loader_config is not None:
            configs.append(self.data_loader_config)

        merged_configs = reduce(lambda a, b: merge(a, b), configs)

        return {key: value for key, value in merged_configs.items() if key in method_args and key in merged_configs}

    def load_and_prepare_configs(self, network_class):
        # Training config
        training_config = load_json(os.path.join(self.path, cd.JSON_TRAINING_CONFIG_NAME))

        # Load model config
        model_config = network_class.load_config(**self.extract_method_args_from_config(network_class.load_config))

        # Loading data
        data_loader_config, data_loader_info, data_loader = self.get_data_loader_info(model_config=model_config)

        # Callbacks config
        callbacks_config = load_json(os.path.join(self.path, cd.JSON_CALLBACKS_NAME))
        callbacks_config = [callbacks_config[callback_name] for callback_name in self.config['callbacks_names']]

        self.training_config = training_config
        self.model_config = model_config
        self.data_loader_config = data_loader_config
        self.callbacks_config = callbacks_config

        return data_loader_info, data_loader

    def load_and_prepare_unseen_configs(self, network_class):

        # Training config
        training_config = load_json(os.path.join(self.path, cd.JSON_TRAINING_CONFIG_NAME))

        # Load model config
        model_config = network_class.load_config(**self.extract_method_args_from_config(network_class.load_config))

        # Loading data
        data_loader_config, data_loader_info, data_loader = self.get_data_loader_info(model_config=model_config)
        train_data_handle = data_loader.load()

        unseen_data_loader_config,\
        unseen_data_loader_info,\
        unseen_data_loader = self.get_unseen_data_loader_info(model_config=model_config)

        # Callbacks config
        callbacks_config = load_json(os.path.join(self.path, cd.JSON_CALLBACKS_NAME))
        callbacks_config = [callbacks_config[callback_name] for callback_name in self.config['callbacks_names']]

        self.training_config = training_config
        self.model_config = model_config
        self.data_loader_config = data_loader_config
        self.callbacks_config = callbacks_config

        return data_loader_info, data_loader, unseen_data_loader_info, unseen_data_loader

    def get_unseen_data_loader_info(self, model_config):
        data_loader_type = self['data_loader_type']
        data_loader_config = load_json(os.path.join(cd.CONFIGS_DIR, cd.JSON_DATA_LOADER_CONFIG_NAME))
        data_loader_info = data_loader_config['configs'][data_loader_type]
        loader_additional_info = {key: value['value'] for key, value in model_config.items()
                                  if 'data_loader' in value['flags']}
        data_loader_info = merge(data_loader_info, loader_additional_info)
        data_loader = DataLoaderFactory.factory(data_loader_type)

        return data_loader_config, data_loader_info, data_loader

    def get_data_loader_info(self, model_config):
        data_loader_config = load_json(os.path.join(self.path, cd.JSON_DATA_LOADER_CONFIG_NAME))
        data_loader_type = data_loader_config['type']
        data_loader_info = data_loader_config['configs'][data_loader_type]
        loader_additional_info = {key: value['value'] for key, value in model_config.items()
                                  if 'data_loader' in value['flags']}
        data_loader_info = merge(data_loader_info, loader_additional_info)
        data_loader = DataLoaderFactory.factory(data_loader_type)

        return data_loader_config, data_loader_info, data_loader

    def determine_save_base_path(self):
        if 'test_name' in self:
            save_base_path = os.path.join(cd.UNSEEN_DATA_DIR, self['model_type'], self['test_name'])
        elif not self['preloaded']:
            current_date = datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
            save_base_path = os.path.join(self.config_folder, self['model_type'], current_date)
        else:
            assert self['preloaded_model'] is not None
            save_base_path = os.path.join(self.config_folder, self['model_type'], self['preloaded_model'])
            if not os.path.isdir(save_base_path):
                msg = "Can't find given pre-trained model. Got: {}".format(save_base_path)
                raise RuntimeError(msg)

        if not os.path.isdir(save_base_path):
            os.makedirs(save_base_path)

        self.save_base_path = save_base_path
        return save_base_path

    def clear_temporary_data(self):
        if not self['save_model']:
            shutil.rmtree(self.save_base_path)

    def get_callbacks(self, callback_factory):
        callbacks = []
        for callback_config, callback_name in zip(self.callbacks_config, self.config['callbacks_names']):
            callback_conditions = callback_config['conditions']
            callback_args = callback_config['args']

            verified = True
            for key, value in callback_conditions.items():
                if key not in self or self[key] != value:
                    verified = False

            if verified:
                callbacks.append(callback_factory.factory(callback_name, **callback_args))

        return callbacks

    def save_configs(self, save_base_path, scores=None):
        if self['save_model']:
            save_json(os.path.join(save_base_path, cd.JSON_DISTRIBUTED_MODEL_CONFIG_NAME), self.model_config)
            save_json(os.path.join(save_base_path, cd.JSON_TRAINING_CONFIG_NAME), self.training_config)
            save_json(os.path.join(save_base_path, cd.JSON_TRAIN_AND_TEST_CONFIG_NAME), data=self.config)
            save_json(os.path.join(save_base_path, cd.JSON_CALLBACKS_NAME), data=self.callbacks_config)
            save_json(os.path.join(save_base_path, cd.JSON_DATA_LOADER_CONFIG_NAME), data=self.data_loader_config)

            if scores is not None:
                save_json(os.path.join(save_base_path, cd.JSON_VALIDATION_INFO_NAME), scores['validation_info'])
                save_json(os.path.join(save_base_path, cd.JSON_TEST_INFO_NAME), scores['test_info'])
                save_json(os.path.join(save_base_path, cd.JSON_PREDICTIONS_NAME), scores['predictions'])

    def save_unseen_predictions(self, save_base_path, predictions):
        if self['save_predictions']:
            network_name = self.get_network_name()
            save_json(os.path.join(save_base_path, '{0}_{1}'.format(network_name, cd.JSON_UNSEEN_PREDICTIONS_NAME)), predictions)

    def rename_folder(self, save_base_path):
        if self['save_model'] and \
                self['rename_folder'] is not None and \
                self['preloaded_model'] is None:
            renamed_base_path = os.path.join(cd.TRAIN_AND_TEST_DIR,
                                             self['model_type'],
                                             self['rename_folder'])
            os.rename(save_base_path, renamed_base_path)

    def register_model_suffix(self, suffix_name, suffix_value):
        if suffix_value is not None:
            self.model_name_additional_suffixes[suffix_name] = suffix_value

    def clear_iteration_status(self):
        self.model_name_additional_suffixes.clear()

    def get_network_name(self):
        name = cd.MODEL_CONFIG[self['model_type']]['save_suffix']
        for key, value in self.model_name_additional_suffixes.items():
            name += '_{0}_{1}'.format(key, value)

        return name

    def setup_model(self, data_handle):
        network_retrieved_args = {key: deepcopy(value['value']) for key, value in self.model_config.items()
                                  if 'model_class' in value['flags']}
        network_retrieved_args['additional_data'] = data_handle.get_additional_info()
        network_retrieved_args['name'] = self.get_network_name()

        return network_retrieved_args


