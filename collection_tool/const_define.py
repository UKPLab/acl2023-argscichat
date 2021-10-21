"""

Constant python configuration script.

"""

import os

NAME_LOG = 'daily_log.log'

# DEFAULT DIRECTORIES
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(PROJECT_DIR, 'configs')
LOCAL_DATASETS_DIR = os.path.join(PROJECT_DIR, 'local_database')

# JSON FILES
JSON_COLLECTION_CONFIG_NAME = 'collection_config.json'