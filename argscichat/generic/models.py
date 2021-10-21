import json
import os

from transformers.file_utils import hf_bucket_url, is_remote_url, CONFIG_NAME, cached_path

import argscichat.const_define as cd
from argscichat.utility.json_utils import load_json
from argscichat.utility.log_utils import Logger
from argscichat.utility.python_utils import merge
from argscichat.generic.factory import Factory

logger = Logger.get_logger(__name__)

pretrained_config_archive_map = {
}

local_config_archive_map = {
}

pretrained_model_archive_map = {
}

local_model_archive_map = {
}

pretrained_config_params_to_ignore = [
    "torchscript",
    "architectures",
    "pruned_heads",
    "finetuning_task",
    "num_labels",
    "model_type",
    "pad_token_id",
]


class BaseNetwork(object):

    def __init__(self, name='network', additional_data=None, is_pretrained=False):
        self.model = None
        self.optimizer = None
        self.name = name
        self.additional_data = additional_data
        self.is_pretrained = is_pretrained

        # Pipeline
        self.processor = None
        self.tokenizer = None
        self.converter = None

        # Utility
        self.label_parsing_map = {
            'generation': self._parse_generation_output,
            'classification': self._parse_classification_output,
            'regression': self._parse_regression_output
        }

    # Pretrained models

    @classmethod
    def from_pretrained_config(cls, pretrained_model_name_or_path, **kwargs):
        cache_dir = kwargs.pop('cache_dir', None)
        force_download = kwargs.pop('force_download', False)
        resume_download = kwargs.pop('resume_download', False)
        proxies = kwargs.pop('proxies', None)
        return_unused_kwargs = kwargs.pop('return_unused_kwargs', False)

        if pretrained_model_name_or_path in pretrained_config_archive_map:
            config_file = pretrained_config_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        elif pretrained_model_name_or_path in local_config_archive_map:
            config_file = local_config_archive_map[pretrained_model_name_or_path]
        else:
            config_file = hf_bucket_url(pretrained_model_name_or_path, CONFIG_NAME)

        try:
            # Load from URL or cache if already cached
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir, force_download=force_download,
                                               proxies=proxies, resume_download=resume_download)
            # Load config
            config = load_json(resolved_config_file)

        except EnvironmentError:
            if pretrained_model_name_or_path in pretrained_config_archive_map:
                msg = "Couldn't reach server at '{}' to download pretrained model configuration file.".format(
                    config_file)
            else:
                msg = "Model name '{}' was not found in model name list ({}). " \
                      "We assumed '{}' was a path or url to a configuration file named {} or " \
                      "a directory containing such a file but couldn't find any such file at this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(pretrained_config_archive_map.keys()),
                    config_file, CONFIG_NAME)
            raise EnvironmentError(msg)

        except json.JSONDecodeError:
            msg = "Couldn't reach server at '{}' to download configuration file or " \
                  "configuration file is not a valid JSON file. " \
                  "Please check network or file content here: {}.".format(config_file, resolved_config_file)
            raise EnvironmentError(msg)

        if resolved_config_file == config_file:
            logger.info("loading configuration file {}".format(config_file))
        else:
            logger.info("loading configuration file {} from cache at {}".format(
                config_file, resolved_config_file))

        if 'pruned_heads' in config:
            config['pruned_heads'] = dict((int(key), value) for key, value in config['pruned_heads'].items())

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if key in config:
                config[key] = value
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        for key in pretrained_config_params_to_ignore:
            if key in config:
                del config[key]

        config = {key: {"value": value, "flags": ["model_class"]} for key, value in config.items()}

        logger.info("Model config %s", str(config))
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def load_config(cls, model_type, preloaded=False, preloaded_model=None, load_externally=False):
        if preloaded:
            if load_externally:
                local_model_config = load_json(os.path.join(cd.CONFIGS_DIR, cd.JSON_DISTRIBUTED_MODEL_CONFIG_NAME))[
                    model_type]

                if 'preloaded_name' in local_model_config:
                    preloaded_name = local_model_config['preloaded_name']['value']
                else:
                    preloaded_name = model_type

                ext_model_config = cls.from_pretrained_config(preloaded_name)
                model_config = merge(ext_model_config, local_model_config, overwrite_conflict=True)
            else:
                assert preloaded_model is not None
                model_config = load_json(os.path.join(cd.TRAIN_AND_TEST_DIR,
                                                      preloaded_model,
                                                      cd.JSON_DISTRIBUTED_MODEL_CONFIG_NAME))
        else:
            model_config = load_json(os.path.join(cd.CONFIGS_DIR, cd.JSON_DISTRIBUTED_MODEL_CONFIG_NAME))[model_type]

        return model_config

    @classmethod
    def from_pretrained_weights(cls, pretrained_model_name_or_path, **kwargs):
        cache_dir = kwargs.pop('cache_dir', None)
        force_download = kwargs.pop('force_download', False)
        resume_download = kwargs.pop('resume_download', False)
        proxies = kwargs.pop('proxies', None)

        # Load model
        if pretrained_model_name_or_path is not None and pretrained_model_name_or_path in pretrained_model_archive_map:
            archive_file = pretrained_model_archive_map[pretrained_model_name_or_path]

            # redirect to the cache, if necessary
            try:
                resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir, force_download=force_download,
                                                    resume_download=resume_download, proxies=proxies)
            except EnvironmentError as e:
                if pretrained_model_name_or_path in pretrained_model_archive_map:
                    logger.error(
                        "Couldn't reach server at '{}' to download pretrained weights.".format(
                            archive_file))
                else:
                    logger.error(
                        "Model name '{}' was not found in model name list ({}). "
                        "We assumed '{}' was a path or url but couldn't find any file "
                        "associated to this path or url.".format(
                            pretrained_model_name_or_path,
                            ', '.join(pretrained_model_archive_map.keys()),
                            archive_file))
                raise e
            if resolved_archive_file == archive_file:
                logger.info("loading weights file {}".format(archive_file))
            else:
                logger.info("loading weights file {} from cache at {}".format(
                    archive_file, resolved_archive_file))
        else:
            if pretrained_model_name_or_path is not None and pretrained_model_name_or_path in local_model_archive_map:
                resolved_archive_file = local_model_archive_map[pretrained_model_name_or_path]
            else:
                resolved_archive_file = None

        return resolved_archive_file

    # Utility

    def _get_additional_info(self):
        return {}

    def set_processor(self, processor):
        self.processor = processor

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def set_converter(self, converter):
        self.converter = converter

    def set_data(self, train_data=None, fixed_train_data=None, val_data=None, test_data=None):
        self.train_data = train_data
        self.fixed_train_data = fixed_train_data
        self.val_data = val_data
        self.test_data = test_data

    # Saving/Weights

    def get_weights(self):
        raise NotImplementedError()

    def set_weights(self, weights):
        raise NotImplementedError()

    def save(self, filepath, overwrite=True):
        raise NotImplementedError()

    def load(self, filepath, is_external=False, **kwargs):
        raise NotImplementedError()

    def get_state(self):
        return None

    def set_state(self, network_state):
        pass

    def _update_internal_state(self, model_additional_info):
        pass

    # Training/Inference

    def predict(self, x, steps, callbacks=None, suffix='test'):
        raise NotImplementedError()

    def evaluate(self, data, steps, val_y, callbacks_None, repetition=1, parsed_metrics=None):
        raise NotImplementedError()

    def evaluate_and_predict(self, data, steps, callbacks=None, suffix='val'):
        raise NotImplementedError()

    def prepare_for_training(self, helper):
        pass

    def fit(self, train_data, fixed_train_data=None, epochs=1, verbose=1,
            callbacks=None, validation_data=None, step_checkpoint=None,
            metrics=None, additional_metrics_info=None, metrics_nicknames=None,
            label_metrics_map=None, train_steps=None, val_steps=None,
            val_y=None, train_y=None, val_inference_repetitions=1):
        raise NotImplementedError()

    def _parse_labels(self, labels):
        return labels

    def _parse_generation_output(self, output, model_additional_info=None):
        raise NotImplementedError()

    def _parse_classification_output(self, output, model_additional_info=None):
        raise NotImplementedError()

    def _parse_regression_output(self, output, model_additional_info=None):
        raise NotImplementedError()

    def _parse_predictions(self, raw_predictions, model_additional_info):
        parsed_predictions = {}
        label_dict = self.label_list.as_dict()
        for key, preds in raw_predictions.items():
            # TODO: key should always be in label_dict
            # Intermediary outputs
            if key in label_dict:
                label_type = label_dict[key].type
            else:
                label_type = 'classification'
            parsed_predictions[key] = self.label_parsing_map[label_type](output=preds,
                                                                         model_additional_info=model_additional_info)

        return parsed_predictions

    # Model definition

    def train_op(self, x, y, additional_info):
        raise NotImplementedError()

    def build_model(self, text_info):
        raise NotImplementedError()

    def loss_op(self, x, targets, training=False, state='training', additional_info=None, return_predictions=False):
        raise NotImplementedError()


class ClassificationNetwork(BaseNetwork):

    def __init__(self, weight_predictions=True, **kwargs):
        super(ClassificationNetwork, self).__init__(**kwargs)
        self.weight_predictions = weight_predictions

    def compute_output_weights(self, y_train, label_list):
        raise NotImplementedError()

    def _classification_ce(self, targets, logits, label_name):
        raise NotImplementedError()

    def _compute_losses(self, targets, logits, label_list):
        total_loss = None
        loss_info = {}
        for label_idx, label in enumerate(label_list):
            label_targets = targets[label.name]
            label_logits = logits[label.name]

            if label.type == 'classification':
                loss = self._classification_ce(targets=label_targets,
                                               logits=label_logits,
                                               label_name=label.name)
            else:
                raise RuntimeError("Invalid label type -> {}".format(label.type))

            loss_info.setdefault(label.name, loss)

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        return total_loss, loss_info

    def prepare_for_training(self, helper):
        self.compute_output_weights(y_train=helper.train_y,
                                    label_list=helper.processor.get_labels())


class GenerativeNetwork(BaseNetwork):

    def __init__(self, **kwargs):
        super(GenerativeNetwork, self).__init__(**kwargs)
        self.max_generation_length = None

    def _classification_ce(self, targets, logits):
        raise NotImplementedError()

    def _compute_losses(self, targets, logits, label_list):
        total_loss = None
        loss_info = {}
        for label_idx, label in enumerate(label_list):
            label_targets = targets[label.name]
            label_logits = logits[label.name]

            if label.type == 'generation':
                loss = self._classification_ce(targets=label_targets,
                                               logits=label_logits)
            else:
                raise RuntimeError("Invalid label type -> {}".format(label.type))

            loss_info.setdefault(label.name, loss)

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        return total_loss, loss_info

    def _parse_generation_output(self, output, model_additional_info=None):
        decoded = [self.tokenizer.decode(seq) for seq in output.tolist()]
        return decoded

    def _parse_labels(self, labels):
        decoded = {key: [self.tokenizer.decode(seq) for seq in label.tolist()] for key, label in labels.items()}
        return decoded

    def generate(self, x):
        raise NotImplementedError()


# Populate with generic models (that do not require a specific framework)
class ModelFactory(Factory):

    @classmethod
    def get_supported_values(cls):
        supported_values = super(ModelFactory, cls).get_supported_values()
        return merge(supported_values, {
        })
