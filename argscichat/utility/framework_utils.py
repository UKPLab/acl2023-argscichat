import importlib
from collections import namedtuple
import os
import argscichat.const_define as cd
from argscichat.utility.json_utils import load_json
from argscichat.utility.log_utils import Logger
import numpy as np


HelperConfig = namedtuple('HelperConfig',
                          'module'
                          ' class_name')


class Helper(object):
    _SUPPORTED_FRAMEWORKS = {
        'tensorflow': HelperConfig('utility.tf_helper', 'TFHelper'),
        'torch': HelperConfig('utility.torch_helper', 'TorchHelper')
    }

    _MODULES = {
        'models': 'models',
        'callbacks': 'callbacks',
        'processor': 'processor',
        'converter': 'converter',
        'tokenizer': 'tokenizer',
        'features': 'features'
    }

    _instance = None
    _framework = None

    @classmethod
    def get_keyword(cls):
        return 'generic'

    @classmethod
    def get_factory_prefix(cls):
        return ''

    @classmethod
    def use(cls, framework='tensorflow'):
        assert framework in Helper._SUPPORTED_FRAMEWORKS
        cls._framework = framework
        cls._instance = cls._load_helper(framework=framework)
        return cls._instance

    @classmethod
    def _load_helper(cls, framework):
        helper_config = Helper._SUPPORTED_FRAMEWORKS[framework]
        module = importlib.import_module(helper_config.module)
        return cls._from_config(getattr(module, helper_config.class_name))

    @staticmethod
    def _build_framework_module(module_name, framework_keyword):
        return framework_keyword + '.' + module_name

    @staticmethod
    def _get_module_factory(factory_name, module_key, framework_keyword):
        models_module_name = Helper._MODULES[module_key]
        models_module_name = Helper._build_framework_module(module_name=models_module_name,
                                                            framework_keyword=framework_keyword)
        module = importlib.import_module(models_module_name)
        return getattr(module, factory_name)

    def setup(self, config_helper):
        pass

    def limit_gpu_usage(self, **kwargs):
        raise NotImplementedError()

    @classmethod
    def _from_config(cls, class_name):
        config_path = os.path.join(cd.CONFIGS_DIR, cd.JSON_HELPER_CONFIG_NAME)

        if os.path.isfile(config_path):
            config = load_json(config_path)[cls._framework]
            cls._instance = class_name(**config)
        else:
            Logger.get_logger(__name__).info('Helper config file not found! Attempting no arguments initialization...')
            cls._instance = class_name()

        return cls._instance

    def get_model_factory(self):
        factory_name = self.get_factory_prefix() + 'ModelFactory'
        return self._get_module_factory(factory_name=factory_name,
                                        module_key='models',
                                        framework_keyword=self.get_keyword())

    def get_callbacks_factory(self):
        factory_name = self.get_factory_prefix() + 'CallbackFactory'
        return self._get_module_factory(factory_name=factory_name,
                                        module_key='callbacks',
                                        framework_keyword=self.get_keyword())

    def setup_distributed_environment(self):
        raise NotImplementedError()

    def setup_pipeline(self, data_loader_info, config_helper, data_handle):
        raise NotImplementedError()

    def setup_unseen_data_pipeline(self, data_loader_info, config_helper, train_data_handle, data_handle):
        raise NotImplementedError()

    def get_pipeline_factories(self):
        raise NotImplementedError()

    def apply_pipeline(self, train_df, val_df, test_df, config_helper, save_prefix=None):
        raise NotImplementedError()

    def apply_unseen_data_pipeline(self, test_df, config_helper):
        raise NotImplementedError()

    def retrieve_dataset_iterators(self, config_helper):
        raise NotImplementedError()

    def retrieve_unseen_dataset_iterators(self, config_helper):
        raise NotImplementedError()

    def display_scores(self, scores, config_helper):
        # Validation
        if 'validation_info' in scores:
            if config_helper['repetitions'] > 1:
                Logger.get_logger(__name__).info('Average validation scores: {}'.format(
                    {key: np.mean(item) for key, item in scores['validation_info'].items() if key.startswith('avg')}))
            else:
                Logger.get_logger(__name__).info('Average validation scores: {}'.format(
                    {key: np.mean(item) for key, item in scores['validation_info'].items() if not key.startswith('avg')}))

        # Test
        if 'test_info' in scores:
            if config_helper['repetitions'] > 1:
                Logger.get_logger(__name__).info('Average test scores: {}'.format(
                    {key: np.mean(item) for key, item in scores['test_info'].items() if key.startswith('avg')}))
            else:
                Logger.get_logger(__name__).info('Average test scores: {}'.format(
                    {key: np.mean(item) for key, item in scores['test_info'].items() if not key.startswith('avg')}))

    def prepare_for_training(self, network, data_handle, config_helper,
                             train_data, fixed_train_data, val_data, test_data):
        raise NotImplementedError()

    def prepare_for_unseen_inference(self, network, data_handle, config_helper, test_data):
        raise NotImplementedError()

    def build_model(self, config_helper, network):
        raise NotImplementedError()

    def prepare_model(self, config_helper, network):
        raise NotImplementedError()

    def prepare_saved_model(self, test_path, config_helper, network):
        raise NotImplementedError()

    def save_model(self, test_path, config_helper, network):
        raise NotImplementedError()

    def save_ground_truth(self, test_path, config_helper):
        raise NotImplementedError()

    def clear_session(self):
        raise NotImplementedError()

    @classmethod
    def get_instance(cls):
        assert cls._instance is not None, "It seems like you didn't call Helper.use()!"
        return cls._instance


