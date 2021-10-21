"""

Simple inference script on a given test data
Check configs/unseen_test_config.json to for more info about customizing the routine behaviour

"""

from argscichat.utility.config_helper import ConfigHelper
from argscichat.utility.distributed_test_utils import unseen_data_test
from argscichat.utility.framework_utils import Helper
from argscichat.utility.log_utils import Logger

if __name__ == '__main__':

    # Helper
    test_type = 'unseen_test'
    config_helper = ConfigHelper(test_type=test_type)

    helper = Helper.use(framework=config_helper['framework'])
    helper.setup_distributed_environment()
    helper.setup(config_helper=config_helper)

    # Get model
    model_factory = helper.get_model_factory()
    network_class = model_factory.get_supported_values()[config_helper['model_type']]

    # Loading config data
    data_loader_info, data_loader,\
    unseen_data_loader_info, unseen_data_loader = config_helper.load_and_prepare_unseen_configs(network_class=network_class)
    train_data_handle = data_loader.load(**data_loader_info)
    unseen_data_handle = unseen_data_loader.load(**unseen_data_loader_info)

    # Test save path
    save_base_path = config_helper.determine_save_base_path()

    # Logging
    Logger.set_log_path(save_base_path)
    logger = Logger.get_logger(__name__)

    # Callbacks
    callbacks_factory = helper.get_callbacks_factory()
    callbacks = config_helper.get_callbacks(callback_factory=callbacks_factory)
    for callback in callbacks:
        callback.set_save_path(save_path=save_base_path)

    predictions = unseen_data_test(train_data_handle=train_data_handle,
                                   data_handle=unseen_data_handle,
                                   callbacks=callbacks,
                                   test_path=save_base_path,
                                   data_loader_info=data_loader_info,
                                   config_helper=config_helper,
                                   network_class=network_class)

    config_helper.save_unseen_predictions(save_base_path=save_base_path,
                                          predictions=predictions)
