"""

Performs a [repeated] train and test evaluation routine.
Check configs/train_and_test_config.json for more info about customizing the routine behaviour

"""

from argscichat.utility.distributed_test_utils import train_and_test
from argscichat.utility.framework_utils import Helper
from argscichat.utility.log_utils import Logger
from argscichat.utility.config_helper import ConfigHelper

if __name__ == '__main__':

    # Load config helper
    test_type = 'train_and_test'
    config_helper = ConfigHelper(test_type=test_type)

    helper = Helper.use(framework=config_helper['framework'])
    helper.setup_distributed_environment()
    helper.setup(config_helper=config_helper)

    # Get model
    model_factory = helper.get_model_factory()
    network_class = model_factory.get_supported_values()[config_helper['model_type']]

    # Loading config data
    data_loader_info, data_loader = config_helper.load_and_prepare_configs(network_class=network_class)
    data_handle = data_loader.load(**data_loader_info)

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

    scores = train_and_test(data_handle=data_handle,
                            callbacks=callbacks,
                            test_path=save_base_path,
                            data_loader_info=data_loader_info,
                            config_helper=config_helper,
                            network_class=network_class)

    helper.display_scores(scores=scores, config_helper=config_helper)
    config_helper.save_configs(save_base_path=save_base_path, scores=scores)

    # Rename folder
    config_helper.rename_folder(save_base_path=save_base_path)

    # Clear temporary data
    config_helper.clear_temporary_data()
