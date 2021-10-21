import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

import argscichat.const_define as cd
from argscichat.utility.config_helper import ConfigHelper
from argscichat.utility.framework_utils import Helper
from argscichat.utility.json_utils import load_json, save_json
from argscichat.utility.log_utils import Logger
from argscichat.utility.python_utils import flatten, merge
from argscichat.utility.tensorflow_utils import get_dataset_fn, retrieve_numpy_labels


class TFHelper(Helper):

    def __init__(self, eager_execution=False, **kwargs):
        super(TFHelper, self).__init__(**kwargs)
        self.eager_execution = eager_execution

    @classmethod
    def get_keyword(cls):
        return 'tf_scripts'

    @classmethod
    def get_factory_prefix(cls):
        return 'TF'

    def limit_gpu_usage(self, limit_gpu_visibility=False, gpu_start_index=None, gpu_end_index=None):
        gpus = tf.config.experimental.list_physical_devices('GPU')

        if limit_gpu_visibility:
            assert gpu_start_index is not None
            assert gpu_end_index is not None
            tf.config.set_visible_devices(gpus[gpu_start_index:gpu_end_index], "GPU")  # avoid other GPUs
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    def enable_eager_execution(self):
        assert tf.version.VERSION.startswith('2.'), \
            "Tensorflow version is not 2.X! This framework only supports >= 2.0 TF versions"
        tf.config.run_functions_eagerly(self.eager_execution)

    def setup(self, config_helper):
        self.limit_gpu_usage(**config_helper.extract_method_args_from_config(self.limit_gpu_usage))
        self.enable_eager_execution()

    def setup_distributed_environment(self):
        self.distributed_info = load_json(os.path.join(cd.CONFIGS_DIR, cd.JSON_DISTRIBUTED_CONFIG_NAME))

    def _get_data_config_id(self, filepath, config):
        config_path = os.path.join(filepath, cd.JSON_MODEL_DATA_CONFIGS_NAME)

        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        if os.path.isfile(config_path):
            data_config = load_json(config_path)
        else:
            data_config = {}

        if config in data_config:
            return int(data_config[config])
        else:
            max_config = list(map(lambda item: int(item), data_config.values()))

            if len(max_config) > 0:
                max_config = max(max_config)
            else:
                max_config = -1

            data_config[config] = max_config + 1

            save_json(config_path, data_config)

            return max_config + 1

    def _clear_data_config(self, filepath, config, config_id):
        config_path = os.path.join(filepath, cd.JSON_MODEL_DATA_CONFIGS_NAME)

        if os.path.isfile(config_path):
            data_config = load_json(config_path)
            del data_config[config]
            save_json(config_path, data_config)

            folder_to_remove = os.path.join(filepath, str(config_id))
            shutil.rmtree(folder_to_remove)

    def _determine_config_id(self, data_loader_info, model_config, model_type, data_handle):
        # Associates an ID to each combination for easy file naming while maintaining whole info
        config_args = {key: arg['value']
                       for key, arg in model_config.items()
                       if 'processor' in arg['flags']
                       or 'tokenizer' in arg['flags']
                       or 'converter' in arg['flags']
                       or 'data_loader' in arg['flags']}
        config_args = flatten(config_args)
        config_args = merge(config_args, data_loader_info)
        config_args_tuple = [(key, value) for key, value in config_args.items()]
        config_args_tuple = sorted(config_args_tuple, key=lambda item: item[0])

        config_name = '_'.join(['{0}-{1}'.format(name, value) for name, value in config_args_tuple])
        model_base_path = os.path.join(cd.TESTS_DATA_DIR,
                                       data_handle.data_name,
                                       model_type)
        config_id = self._get_data_config_id(filepath=model_base_path, config=config_name)
        model_path = os.path.join(model_base_path, str(config_id))

        if not os.path.isdir(model_path):
            os.makedirs(model_path)

        self.config_id = config_id
        self.model_path = model_path

    def setup_pipeline(self, data_loader_info, data_handle, config_helper):

        model_config = config_helper.model_config
        model_type = config_helper['model_type']

        self._determine_config_id(data_loader_info=data_loader_info,
                                  model_config=model_config,
                                  model_type=model_type,
                                  data_handle=data_handle)

        # Build processor
        processor_type = cd.MODEL_CONFIG[model_type]['processor']
        processor_args = {key: arg['value'] for key, arg in model_config.items() if 'processor' in arg['flags']}
        processor_args['loader_info'] = data_handle.get_additional_info()
        processor_framework_keyword = cd.PIPELINE_INFO['processor'][processor_type]
        processor_factory_name = cd.SUFFIX_MAP[processor_framework_keyword] + 'ProcessorFactory'
        processor_factory = self._get_module_factory(factory_name=processor_factory_name,
                                                     module_key='processor',
                                                     framework_keyword=processor_framework_keyword)
        processor = processor_factory.factory(processor_type, **processor_args)

        # Build tokenizer
        tokenizer_type = cd.MODEL_CONFIG[model_type]['tokenizer']
        tokenizer_framework_keyword = cd.PIPELINE_INFO['tokenizer'][tokenizer_type]
        tokenizer_factory_name = cd.SUFFIX_MAP[tokenizer_framework_keyword] + 'TokenizerFactory'
        tokenizer_factory = self._get_module_factory(factory_name=tokenizer_factory_name,
                                                     module_key='tokenizer',
                                                     framework_keyword=tokenizer_framework_keyword)
        if config_helper['preloaded'] and config_helper['load_externally']:
            tokenizer_class = tokenizer_factory.supported_tokenizers[tokenizer_type]
            tokenizer = tokenizer_class.from_pretrained(model_config['preloaded_name']['value'])
        else:
            tokenizer_args = {key: arg['value'] for key, arg in model_config.items() if 'tokenizer' in arg['flags']}
            tokenizer = tokenizer_factory.factory(tokenizer_type, **tokenizer_args)

        # Build converter
        converter_type = cd.MODEL_CONFIG[model_type]['converter']
        converter_args = {key: arg['value'] for key, arg in model_config.items() if 'converter' in arg['flags']}
        converter_framework_keyword = cd.PIPELINE_INFO['converter'][converter_type]
        converter_factory_name = cd.SUFFIX_MAP[converter_framework_keyword] + 'ConverterFactory'
        converter_factory = self._get_module_factory(factory_name=converter_factory_name,
                                                     module_key='converter',
                                                     framework_keyword=converter_framework_keyword)
        converter = converter_factory.factory(converter_type, **converter_args)

        self.processor_type = processor_type
        self.processor_factory = processor_factory
        self.processor = processor

        self.tokenizer_type = tokenizer_type
        self.tokenizer_factory = tokenizer_factory
        self.tokenizer = tokenizer

        self.converter_type = converter_type
        self.converter_factory = converter_factory
        self.converter = converter

        return self.processor, self.tokenizer, self.converter

    def setup_unseen_data_pipeline(self, data_loader_info, config_helper, train_data_handle, data_handle):
        model_config = config_helper.model_config
        model_type = config_helper['model_type']

        self._determine_config_id(data_loader_info=data_loader_info,
                                  model_config=model_config,
                                  model_type=model_type,
                                  data_handle=train_data_handle)

        # Build processor
        processor_type = cd.MODEL_CONFIG[model_type]['processor']
        processor_args = {key: arg['value'] for key, arg in model_config.items() if 'processor' in arg['flags']}
        processor_args['loader_info'] = data_handle.get_additional_info()
        processor_framework_keyword = cd.PIPELINE_INFO['processor'][processor_type]
        processor_factory_name = cd.SUFFIX_MAP[processor_framework_keyword] + 'ProcessorFactory'
        processor_factory = self._get_module_factory(factory_name=processor_factory_name,
                                                     module_key='processor',
                                                     framework_keyword=processor_framework_keyword)
        processor = processor_factory.factory(processor_type, **processor_args, retrieve_label=False)

        # Build tokenizer
        tokenizer_type = cd.MODEL_CONFIG[model_type]['tokenizer']
        tokenizer_framework_keyword = cd.PIPELINE_INFO['tokenizer'][tokenizer_type]
        tokenizer_factory_name = cd.SUFFIX_MAP[tokenizer_framework_keyword] + 'TokenizerFactory'
        tokenizer_factory = self._get_module_factory(factory_name=tokenizer_factory_name,
                                                     module_key='tokenizer',
                                                     framework_keyword=tokenizer_framework_keyword)
        if config_helper['preloaded'] and config_helper['load_externally']:
            tokenizer_class = tokenizer_factory.supported_tokenizers[tokenizer_type]
            tokenizer = tokenizer_class.from_pretrained(model_config['preloaded_name']['value'])
        else:
            tokenizer_args = {key: arg['value'] for key, arg in model_config.items() if 'tokenizer' in arg['flags']}
            tokenizer = tokenizer_factory.factory(tokenizer_type, **tokenizer_args)

        # Build converter
        converter_type = cd.MODEL_CONFIG[model_type]['converter']
        converter_args = {key: arg['value'] for key, arg in model_config.items() if 'converter' in arg['flags']}
        converter_framework_keyword = cd.PIPELINE_INFO['converter'][converter_type]
        converter_factory_name = cd.SUFFIX_MAP[converter_framework_keyword] + 'ConverterFactory'
        converter_factory = self._get_module_factory(factory_name=converter_factory_name,
                                                     module_key='converter',
                                                     framework_keyword=converter_framework_keyword)
        converter = converter_factory.factory(converter_type, **converter_args)

        self.processor_type = processor_type
        self.processor_factory = processor_factory
        self.processor = processor

        self.tokenizer_type = tokenizer_type
        self.tokenizer_factory = tokenizer_factory
        self.tokenizer = tokenizer

        self.converter_type = converter_type
        self.converter_factory = converter_factory
        self.converter = converter

        self.unseen_model_path = os.path.join(cd.TESTS_DATA_DIR,
                                              data_handle.data_name,
                                              config_helper['model_type'],
                                              str(self.config_id))

        if not os.path.isdir(self.unseen_model_path):
            os.makedirs(self.unseen_model_path)

        return self.processor, self.tokenizer, self.converter

    def apply_pipeline(self, train_df, val_df, test_df, config_helper, save_prefix=None):
        self.train_filepath = os.path.join(self.model_path, 'train_data')
        self.val_filepath = os.path.join(self.model_path, 'val_data')
        self.test_filepath = os.path.join(self.model_path, 'test_data')

        if (not os.path.isfile(self.test_filepath) and test_df is not None) \
                or (test_df is None and not os.path.isfile(self.val_filepath)):
            Logger.get_logger(__name__).info(
                'Dataset not found! Building new one from scratch....it may require some minutes')

            # Processor

            train_data = self.processor.get_train_examples(data=train_df, ids=np.arange(train_df.shape[0]))
            if val_df is not None:
                val_data = self.processor.get_dev_examples(data=val_df, ids=np.arange(val_df.shape[0]))
            if test_df is not None:
                test_data = self.processor.get_test_examples(data=test_df, ids=np.arange(test_df.shape[0]))

            # Tokenizer

            train_texts = train_data.get_data()
            self.tokenizer.build_vocab(data=train_texts, filepath=self.model_path, prefix=save_prefix)
            self.tokenizer.save_info(filepath=self.model_path, prefix=save_prefix)
            tokenizer_info = self.tokenizer.get_info()

            # Conversion

            # WARNING: suffers multi-threading (what if another processing is building the same data?)
            # This may happen only the first time an input pipeline is used. Usually calibration is on
            # model parameters
            self.converter.convert_data(examples=train_data,
                                        label_list=self.processor.get_labels(),
                                        output_file=self.train_filepath,
                                        tokenizer=self.tokenizer,
                                        suffix='train',
                                        is_training=True,
                                        additional_data={'checkpoint': None})
            self.converter.save_conversion_args(filepath=self.model_path, prefix=save_prefix)
            converter_info = self.converter.get_conversion_args()

            if val_df is not None:
                self.converter.convert_data(examples=val_data,
                                            label_list=self.processor.get_labels(),
                                            output_file=self.val_filepath,
                                            tokenizer=self.tokenizer,
                                            suffix='val',
                                            additional_data={'checkpoint': None})
            if test_df is not None:
                self.converter.convert_data(examples=test_data,
                                            label_list=self.processor.get_labels(),
                                            output_file=self.test_filepath,
                                            tokenizer=self.tokenizer,
                                            suffix='test',
                                            additional_data={'checkpoint': None})

            self.converter.save_instance_args(filepath=self.model_path, prefix=save_prefix)
        else:
            tokenizer_info = self.tokenizer_factory.get_supported_values()[self.tokenizer_type].load_info(
                filepath=self.model_path,
                prefix=save_prefix)
            converter_info = self.converter_factory.get_supported_values()[self.converter_type].load_conversion_args(
                filepath=self.model_path,
                prefix=save_prefix)
            self.converter.load_instance_args(filepath=self.model_path, prefix=save_prefix)
            self.tokenizer.initialize_with_info(tokenizer_info)
            self.converter.set_conversion_args(conversion_args=converter_info)

        self.tokenizer_info = tokenizer_info
        self.converter_info = converter_info

        self.tokenizer.show_info(tokenizer_info)
        Logger.get_logger(__name__).info('Converter info: \n{}'.format(converter_info))

    def apply_unseen_data_pipeline(self, test_df, config_helper):

        save_prefix = config_helper['save_prefix']
        test_prefix = config_helper['test_prefix']

        test_name = 'test_data' if save_prefix is None else 'test_data_{}'.format(save_prefix)
        self.test_filepath = os.path.join(self.unseen_model_path, test_name)

        info_prefix = None
        if test_prefix is not None and save_prefix is not None:
            info_prefix = '{0}_{1}'.format(test_prefix, save_prefix)
        else:
            if save_prefix is not None:
                info_prefix = save_prefix

        tokenizer_info = self.tokenizer_factory.get_supported_values()[self.tokenizer_type].load_info(
            filepath=self.model_path,
            prefix=info_prefix)
        converter_info = self.converter_factory.get_supported_values()[self.converter_type].load_conversion_args(
            filepath=self.model_path,
            prefix='{0}_{1}'.format(test_prefix, save_prefix) if test_prefix is not None else save_prefix)
        self.converter.load_instance_args(filepath=self.model_path, prefix=info_prefix)
        self.converter.load_conversion_args(filepath=self.model_path, prefix=info_prefix)

        self.tokenizer_info = tokenizer_info
        self.converter_info = converter_info

        self.tokenizer.show_info(tokenizer_info)
        Logger.get_logger(__name__).info('Converter info: \n{}'.format(converter_info))

        Logger.get_logger(__name__).info(
            'Building unseen data on the fly....it may require some minutes '
            '(this is a temporary solution and it will be improved in the near future)')

        # Processor
        test_data = self.processor.get_test_examples(data=test_df, ids=np.arange(test_df.shape[0]))

        # Tokenizer
        self.tokenizer.initialize_with_info(tokenizer_info)

        # Conversion
        for key, value in converter_info.items():
            setattr(self.converter, key, value)
        self.converter.convert_data(examples=test_data,
                                    label_list=self.processor.get_labels(),
                                    output_file=self.test_filepath,
                                    has_labels=False,
                                    tokenizer=self.tokenizer,
                                    suffix='test')

    def retrieve_dataset_iterators(self, config_helper: ConfigHelper):

        training_config = config_helper.training_config
        converter_args = self.converter.get_instance_args()

        train_data = get_dataset_fn(filepath=self.train_filepath,
                                    batch_size=training_config['batch_size'],
                                    name_to_features=self.converter.feature_class.get_mappings(self.converter_info,
                                                                                               converter_args=converter_args),
                                    selector=self.converter.feature_class.get_dataset_selector(
                                        self.processor.get_labels()),
                                    is_training=True,
                                    shuffle_amount=self.distributed_info['shuffle_amount'],
                                    reshuffle_each_iteration=self.distributed_info['reshuffle_each_iteration'],
                                    prefetch_amount=self.distributed_info['prefetch_amount'])

        fixed_train_data = get_dataset_fn(filepath=self.train_filepath,
                                          batch_size=training_config['batch_size'],
                                          name_to_features=self.converter.feature_class.get_mappings(
                                              self.converter_info,
                                              converter_args=converter_args),
                                          selector=self.converter.feature_class.get_dataset_selector(
                                              self.processor.get_labels()),
                                          is_training=False,
                                          prefetch_amount=self.distributed_info['prefetch_amount'])

        if os.path.isfile(self.val_filepath):
            val_data = get_dataset_fn(filepath=self.val_filepath,
                                      batch_size=training_config['batch_size'],
                                      name_to_features=self.converter.feature_class.get_mappings(self.converter_info,
                                                                                                 converter_args=converter_args),
                                      selector=self.converter.feature_class.get_dataset_selector(
                                          self.processor.get_labels()),
                                      is_training=False,
                                      prefetch_amount=self.distributed_info['prefetch_amount'])
        else:
            val_data = None

        if os.path.isfile(self.test_filepath):
            test_data = get_dataset_fn(filepath=self.test_filepath,
                                       batch_size=training_config['batch_size'],
                                       name_to_features=self.converter.feature_class.get_mappings(self.converter_info,
                                                                                                  converter_args=converter_args),
                                       selector=self.converter.feature_class.get_dataset_selector(
                                           self.processor.get_labels()),
                                       is_training=False,
                                       prefetch_amount=self.distributed_info['prefetch_amount'])
        else:
            test_data = None

        return train_data, fixed_train_data, val_data, test_data

    def retrieve_unseen_dataset_iterators(self, config_helper: ConfigHelper):

        training_config = config_helper.training_config
        converter_args = self.converter.get_conversion_args()

        test_data = get_dataset_fn(filepath=self.test_filepath,
                                   batch_size=training_config['batch_size'],
                                   name_to_features=self.converter.feature_class.get_mappings(self.converter_info,
                                                                                              converter_args=converter_args,
                                                                                              has_labels=False),
                                   selector=self.converter.feature_class.get_dataset_selector(
                                       self.processor.get_labels()),
                                   is_training=False,
                                   prefetch_amount=self.distributed_info['prefetch_amount'])

        return test_data

    def prepare_for_training(self, network, data_handle, config_helper,
                             train_data, fixed_train_data, val_data, test_data):
        # Useful stuff
        converter = network.converter
        training_config = config_helper.training_config

        train_steps = int(np.ceil(converter.data_sizes['train'] / training_config['batch_size']))

        if val_data is not None:
            eval_steps = int(np.ceil(converter.data_sizes['val'] / training_config['batch_size']))
        else:
            eval_steps = None

        if test_data is not None:
            test_steps = int(np.ceil(converter.data_sizes['test'] / training_config['batch_size']))
        else:
            test_steps = None

        np_train_y = retrieve_numpy_labels(data_fn=fixed_train_data, steps=train_steps)

        if val_data is not None:
            np_val_y = retrieve_numpy_labels(data_fn=val_data, steps=eval_steps)
        else:
            np_val_y = None

        if test_data is not None:
            np_test_y = retrieve_numpy_labels(data_fn=test_data, steps=test_steps)
        else:
            np_test_y = None

        Logger.get_logger(__name__).info('Total train steps: {}'.format(train_steps))
        if val_data is not None:
            Logger.get_logger(__name__).info('Total eval steps: {}'.format(eval_steps))
        if test_data is not None:
            Logger.get_logger(__name__).info('Total test steps: {}'.format(test_steps))

        self.train_steps = train_steps
        self.train_y = np_train_y

        self.val_steps = eval_steps
        self.val_y = np_val_y

        self.test_steps = test_steps
        self.test_y = np_test_y

        network.prepare_for_training(self)

    def prepare_for_unseen_inference(self, network, data_handle, config_helper, test_data):

        # Useful stuff
        converter = network.converter
        training_config = config_helper.training_config

        if test_data is not None:
            test_steps = int(np.ceil(converter.data_sizes['test'] / training_config['batch_size']))
        else:
            test_steps = None

        if test_data is not None:
            Logger.get_logger(__name__).info('Total test steps: {}'.format(test_steps))

        self.test_steps = test_steps

    def build_model(self, config_helper, network):
        text_info = merge(self.tokenizer_info, self.converter_info)
        text_info = merge(text_info, config_helper.training_config)
        network.build_model(text_info=text_info)

        return network

    def prepare_model(self, config_helper, network):
        if config_helper['preloaded']:
            network.predict(x=iter(network.val_data()), steps=1)

            initial_weights = [layer.get_weights() for layer in network.model.layers]

            if config_helper['load_externally']:
                if hasattr(network, 'preloaded_name'):
                    preloaded_name = network.preloaded_name
                else:
                    preloaded_name = config_helper['model_type']
                network.load(network.from_pretrained_weights(preloaded_name), by_name=True, is_external=True,
                             skip_mismatch=True)
            else:
                test_folder = config_helper.config_folder
                network.load(
                    os.path.join(test_folder,
                                 config_helper['preloaded_model_name'],
                                 config_helper.get_network_name()))

            # Correct loading check (inherently makes sure that restore ops are run)
            for layer, initial in zip(network.model.layers, initial_weights):
                weights = layer.get_weights()
                if weights and all(tf.nest.map_structure(np.array_equal, weights, initial)):
                    Logger.get_logger(__name__).info('Checkpoint contained no weights for layer {}!'.format(layer.name))

    def prepare_saved_model(self, test_path, config_helper, network):
        network.predict(x=iter(network.test_data()), steps=1)
        current_weight_filename = os.path.join(config_helper.path,
                                               '{}.h5'.format(config_helper.get_network_name()))
        network.load(current_weight_filename)

    def clear_session(self):
        K.clear_session()

    def save_model(self, test_path, config_helper, network):
        if config_helper['save_model']:
            filepath = os.path.join(test_path, config_helper.get_network_name())
            network.save(filepath=filepath)

    # TODO: extend to all data splits? @Frgg
    def save_ground_truth(self, test_path, config_helper):
        if config_helper['save_model']:
            filepath = os.path.join(test_path, 'y_test.json')
            if not os.path.isfile(filepath):
                save_json(filepath=filepath, data=self.test_y)
