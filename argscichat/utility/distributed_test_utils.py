from __future__ import division

from collections import OrderedDict

import numpy as np

from argscichat.generic.data_loader import DataHandle
from argscichat.utility.config_helper import ConfigHelper
from argscichat.utility.evaluation_utils import build_metrics, compute_iteration_validation_error, update_cv_validation_info
from argscichat.utility.framework_utils import Helper
from argscichat.utility.log_utils import Logger
from argscichat.utility.python_utils import merge


def train_and_test(data_handle: DataHandle, test_path: str, data_loader_info: dict,
                   callbacks: list, network_class: type,
                   config_helper: ConfigHelper):
    helper = Helper.get_instance()

    # Step 0: Build metrics
    parsed_metrics = build_metrics(**config_helper.extract_method_args_from_config(build_metrics))

    # Step 1: Prepare pipeline: each step here is guaranteed to be idempotent (make sure of it!)
    processor, tokenizer, converter = helper.setup_pipeline(data_loader_info=data_loader_info,
                                                            data_handle=data_handle,
                                                            config_helper=config_helper)

    # Step 2: Train and test
    total_validation_info = OrderedDict()
    total_test_info = OrderedDict()
    total_preds = OrderedDict()

    for repetition in range(config_helper['repetitions']):
        Logger.get_logger(__name__).info('Repetition {0}/{1}'.format(repetition + 1, config_helper['repetitions']))

        validation_info = OrderedDict()
        test_info = OrderedDict()
        all_preds = OrderedDict()

        train_df, val_df, test_df = data_handle.get_data(config_helper['validation_percentage'])

        save_prefix = None
        helper.apply_pipeline(train_df=train_df,
                              val_df=val_df,
                              test_df=test_df,
                              config_helper=config_helper,
                              save_prefix=save_prefix)

        # Building network
        config_helper.register_model_suffix(suffix_name='repetition', suffix_value=repetition)
        network_retrieved_args = config_helper.setup_model(data_handle=data_handle)
        network = network_class(**network_retrieved_args)
        network.set_processor(processor)
        network.set_tokenizer(tokenizer)
        network.set_converter(converter)

        # Create Datasets
        train_data, fixed_train_data, val_data, test_data = helper.retrieve_dataset_iterators(
            config_helper=config_helper)
        network.set_data(train_data=train_data,
                         fixed_train_data=fixed_train_data,
                         val_data=val_data,
                         test_data=test_data)

        helper.prepare_for_training(network=network,
                                    data_handle=data_handle,
                                    config_helper=config_helper,
                                    train_data=train_data,
                                    fixed_train_data=fixed_train_data,
                                    val_data=val_data,
                                    test_data=test_data)

        # Custom callbacks only
        for callback in callbacks:
            if hasattr(callback, 'on_build_model_begin'):
                callback.on_build_model_begin(logs={'network': network})

        network = helper.build_model(config_helper=config_helper,
                                     network=network)

        # Custom callbacks only
        for callback in callbacks:
            if hasattr(callback, 'on_build_model_end'):
                callback.on_build_model_end(logs={'network': network})

        helper.prepare_model(config_helper=config_helper,
                             network=network)

        # Training
        network.fit(train_data=train_data, callbacks=callbacks, validation_data=val_data,
                    metrics=config_helper.training_config['metrics'],
                    additional_metrics_info=config_helper.training_config['additional_metrics_info'],
                    metrics_nicknames=config_helper.training_config['metrics_nicknames'],
                    label_metrics_map=config_helper.training_config['label_metrics_map'],
                    epochs=config_helper.training_config['epochs'],
                    verbose=config_helper.training_config['verbose'],
                    step_checkpoint=config_helper.training_config['step_checkpoint'],
                    train_steps=helper.train_steps,
                    val_steps=helper.val_steps,
                    val_y=helper.val_y,
                    train_y=helper.train_y)

        # Inference
        if val_data is not None:
            np_val_y = network._parse_labels(helper.val_y)
            val_predictions = network.predict(x=val_data,
                                              steps=helper.val_steps,
                                              callbacks=callbacks)

            iteration_validation_error = compute_iteration_validation_error(parsed_metrics=parsed_metrics,
                                                                            true_values=np_val_y,
                                                                            predicted_values=val_predictions)

            validation_info = update_cv_validation_info(test_validation_info=validation_info,
                                                        iteration_validation_info=iteration_validation_error)

            Logger.get_logger(__name__).info('Iteration validation info: {}'.format(iteration_validation_error))

        if config_helper['compute_test_info'] and test_data is not None:
            np_test_y = network._parse_labels(helper.test_y)
            test_predictions = network.predict(x=test_data,
                                               steps=helper.test_steps,
                                               callbacks=callbacks)

            all_preds[repetition] = test_predictions

            iteration_test_error = compute_iteration_validation_error(parsed_metrics=parsed_metrics,
                                                                      true_values=np_test_y,
                                                                      predicted_values=test_predictions)

            test_info = update_cv_validation_info(test_validation_info=test_info,
                                                  iteration_validation_info=iteration_test_error)

            Logger.get_logger(__name__).info('Iteration test info: {}'.format(iteration_test_error))

        # Save model
        helper.save_model(test_path=test_path,
                          config_helper=config_helper,
                          network=network)

        # Save ground truth
        helper.save_ground_truth(test_path=test_path,
                                 config_helper=config_helper)

        # Flush
        helper.clear_session()
        config_helper.clear_iteration_status()

        for key, item in validation_info.items():
            total_validation_info.setdefault(key, []).append(item)
        for key, item in test_info.items():
            total_test_info.setdefault(key, []).append(item)
        for key, item in all_preds.items():
            total_preds.setdefault(key, []).append(item)

    if config_helper['repetitions'] == 1:
        total_validation_info = {key: np.mean(item, 0) for key, item in total_validation_info.items()}
        total_test_info = {key: np.mean(item, 0) for key, item in total_test_info.items()}
    else:
        avg_validation_info = {}
        for key, item in total_validation_info.items():
            avg_validation_info['avg_{}'.format(key)] = np.mean(item, 0)
        total_validation_info = merge(total_validation_info, avg_validation_info)

        avg_test_info = {}
        for key, item in total_test_info.items():
            avg_test_info['avg_{}'.format(key)] = np.mean(item, 0)
        total_test_info = merge(total_test_info, avg_test_info)

    result = {
        'validation_info': total_validation_info,
        'predictions': total_preds
    }

    if config_helper['compute_test_info']:
        result['test_info'] = total_test_info

    return result


def unseen_data_test(data_handle: DataHandle, test_path: str, data_loader_info: dict,
                     train_data_handle: DataHandle, config_helper: ConfigHelper,
                     callbacks: list, network_class: type):
    helper = Helper.get_instance()

    # Step 1: Prepare pipeline: each step here is guaranteed to be idempotent (make sure of it!)
    processor, tokenizer, converter = helper.setup_unseen_data_pipeline(data_loader_info=data_loader_info,
                                                                        train_data_handle=train_data_handle,
                                                                        data_handle=data_handle,
                                                                        config_helper=config_helper)

    # Step 1: Unseen data test

    test_df = data_handle.get_data()
    helper.apply_unseen_data_pipeline(test_df=test_df,
                                      config_helper=config_helper)

    # Building network
    config_helper.register_model_suffix(suffix_name='repetition', suffix_value=config_helper['repetition_prefix'])
    config_helper.register_model_suffix(suffix_name='key', suffix_value=config_helper['save_prefix'])
    network_retrieved_args = config_helper.setup_model(data_handle=data_handle)
    network = network_class(**network_retrieved_args)
    network.set_processor(processor)
    network.set_tokenizer(tokenizer)
    network.set_converter(converter)

    # Create Datasets
    test_data = helper.retrieve_unseen_dataset_iterators(config_helper=config_helper)

    network.set_data(test_data=test_data)

    helper.prepare_for_unseen_inference(network=network,
                                        data_handle=data_handle,
                                        config_helper=config_helper,
                                        test_data=test_data)

    # Custom callbacks only
    for callback in callbacks:
        if hasattr(callback, 'on_build_model_begin'):
            callback.on_build_model_begin(logs={'network': network})

    network = helper.build_model(config_helper=config_helper,
                                 network=network)

    # Custom callbacks only
    for callback in callbacks:
        if hasattr(callback, 'on_build_model_end'):
            callback.on_build_model_end(logs={'network': network})

    helper.prepare_saved_model(test_path=test_path,
                               config_helper=config_helper,
                               network=network)

    # Inference
    test_predictions = network.predict(x=iter(test_data()),
                                       steps=helper.test_steps,
                                       callbacks=callbacks)

    # Flush
    helper.clear_session()

    return test_predictions
