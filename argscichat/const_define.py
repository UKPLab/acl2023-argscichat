"""

Constant python configuration script.

"""

import os

NAME_LOG = 'daily_log.log'

# DEFAULT DIRECTORIES
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(PROJECT_DIR, 'configs')
PATH_LOG = os.path.join(PROJECT_DIR, 'log')
RUNNABLES_DIR = os.path.join(PROJECT_DIR, 'runnables')
LOCAL_DATASETS_DIR = os.path.join(PROJECT_DIR, 'local_database')
TESTS_DATA_DIR = os.path.join(PROJECT_DIR, 'tests_data')

# EVALUATION METHOD DIRS
TRAIN_AND_TEST_DIR = os.path.join(PROJECT_DIR, 'train_and_test')
UNSEEN_DATA_DIR = os.path.join(PROJECT_DIR, 'unseen_data_test')

# EXT MODELS
EXT_MODELS_DIR = os.path.join(PROJECT_DIR, 'ext_models')

# JSON FILES
JSON_CALLBACKS_NAME = 'callbacks.json'
JSON_MODEL_CONFIG_NAME = 'model_config.json'
JSON_TRAINING_CONFIG_NAME = 'training_config.json'
JSON_LOO_TEST_CONFIG_NAME = 'loo_test_config.json'
JSON_TRAIN_AND_TEST_CONFIG_NAME = 'train_and_test_config.json'
JSON_UNSEEN_TEST_CONFIG_NAME = 'unseen_test_config.json'
JSON_CV_TEST_CONFIG_NAME = 'cv_test_config.json'
JSON_DATA_LOADER_CONFIG_NAME = 'data_loader.json'
JSON_CALIBRATOR_INFO_NAME = 'calibrator_info.json'
JSON_VALIDATION_INFO_NAME = 'validation_info.json'
JSON_TEST_INFO_NAME = 'test_info.json'
JSON_PREDICTIONS_NAME = 'predictions.json'
JSON_DISTRIBUTED_MODEL_CONFIG_NAME = 'model_config.json'
JSON_DISTRIBUTED_CONFIG_NAME = 'distributed_config.json'
JSON_MODEL_DATA_CONFIGS_NAME = 'data_configs.json'
JSON_UNSEEN_PREDICTIONS_NAME = 'unseen_predictions.json'
JSON_QUICK_SETUP_CONFIG_NAME = "quick_setup_config.json"
JSON_HELPER_CONFIG_NAME = 'helper_config.json'

# ALGORITHMS

MODEL_CONFIG = {
    'drinventor_tf_components_scibert': {
        'processor': 'base_components_processor',
        'tokenizer': 'scibert_tokenizer',
        'converter': 'tf_transformer_components_converter',
        'save_suffix': 'drinventor_tf_components_scibert',
    },
    'drinventor_tf_tokens_scibert': {
        'processor': 'base_tokens_processor',
        'tokenizer': 'scibert_tokenizer',
        'converter': 'tf_transformer_tokens_converter',
        'save_suffix': 'drinventor_tf_tokens_scibert',
    },
    'drinventor_tf_tokens_scibert_crf': {
        'processor': 'base_tokens_processor',
        'tokenizer': 'scibert_tokenizer',
        'converter': 'tf_transformer_tokens_converter',
        'save_suffix': 'drinventor_tf_tokens_scibert_crf',
    }
}

PIPELINE_INFO = {
    "processor": {
        'base_components_processor': 'generic',
        'base_tokens_processor': 'generic'
    },
    "tokenizer": {
        'keras_tokenizer': 'tf_scripts',
        'scibert_tokenizer': 'generic'
    },
    "converter": {
        'tf_transformer_components_converter': 'tf_scripts',
        'tf_transformer_tokens_converter': 'tf_scripts',
        'tf_base_tokens_converter': 'tf_scripts',
    }
}

SUFFIX_MAP = {
    'generic': '',
    'tf_scripts': 'TF',
    'torch_scripts': 'Torch'
}

TEST_INFO = {
    'train_and_test': {
        "filename": JSON_TRAIN_AND_TEST_CONFIG_NAME,
        "folder": TRAIN_AND_TEST_DIR
    },
    "unseen_test": {
        "filename": JSON_UNSEEN_TEST_CONFIG_NAME,
        "folder": UNSEEN_DATA_DIR
    }
}
