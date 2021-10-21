import os
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from transformers import BertConfig

from argscichat.generic.metrics import MetricManager
from argscichat.generic.models import BaseNetwork, ModelFactory, ClassificationNetwork, GenerativeNetwork
from argscichat.tf_scripts.tf_models import M_TokensSciBert, M_TokensSciBertCRF, M_ComponentsSciBert
from argscichat.utility.evaluation_utils import build_metrics, compute_iteration_validation_error
from argscichat.utility.log_utils import Logger
from argscichat.utility.printing_utils import prettify_statistics
from argscichat.utility.python_utils import merge

logger = Logger.get_logger(__name__)


class TFNetwork(BaseNetwork):

    # Saving/Weights

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def save(self, filepath, overwrite=True):
        if filepath.endswith('.h5'):
            base_path = filepath.split('.h5')[0]
        else:
            base_path = filepath

        # Model weights
        weights_path = base_path + '.h5'
        self.model.save_weights(weights_path)

        # Model state
        network_state = self.get_state()
        if network_state is not None:
            state_path = base_path + '.pickle'
            with open(state_path, 'wb') as f:
                pickle.dump(network_state, f)

    def load(self, filepath, is_external=False, **kwargs):
        if filepath.endswith('.h5'):
            base_path = filepath.split('.h5')[0]
            weights_path = base_path + '.h5'
        else:
            base_path = filepath + '.h5'
            weights_path = base_path + '.h5'

        # Model weights
        self.model.load_weights(filepath=weights_path, **kwargs)

        # Model state
        if not is_external:
            state_path = base_path + '.pickle'
            if os.path.isfile(state_path):
                with open(state_path, 'rb') as f:
                    network_state = pickle.load(f)

                self.set_state(network_state=network_state)

    # Training/Inference

    def predict(self, x, steps, callbacks=None, suffix='test'):
        callbacks = callbacks or []

        total_preds = {}

        for callback in callbacks:
            if hasattr(callback, 'on_prediction_begin'):
                if not hasattr(callback, 'model'):
                    callback.set_model(model=self)
                callback.on_prediction_begin(logs={'suffix': suffix})

        for batch_idx in tqdm(range(steps), leave=True, position=0):

            for callback in callbacks:
                if hasattr(callback, 'on_batch_prediction_begin'):
                    callback.on_batch_prediction_begin(batch=batch_idx, logs={'suffix': suffix})

            batch = next(x)
            if type(batch) in [tuple, list]:
                batch = batch[0]
            preds, model_additional_info = self.batch_predict(x=batch)
            preds = {key: self._parse_predictions(value, model_additional_info).numpy() for key, value in preds.items()}
            for key, value in preds.items():
                total_preds.setdefault(key, []).extend(value)

            for callback in callbacks:
                if hasattr(callback, 'on_batch_prediction_end'):
                    callback.on_batch_prediction_end(batch=batch_idx, logs={'predictions': preds,
                                                                            'model_additional_info': model_additional_info,
                                                                            'suffix': suffix})

        for callback in callbacks:
            if hasattr(callback, 'on_prediction_end'):
                callback.on_prediction_end(logs={'suffix': suffix})

        return total_preds

    def evaluate_and_predict(self, data, steps, callbacks=None, suffix='val'):
        total_loss = {}

        callbacks = callbacks or []

        total_preds = {}

        for callback in callbacks:
            if hasattr(callback, 'on_prediction_begin'):
                if not hasattr(callback, 'model'):
                    callback.set_model(model=self)
                callback.on_prediction_begin(logs={'suffix': suffix})

        for batch_idx in tqdm(range(steps), leave=True, position=0):

            for callback in callbacks:
                if hasattr(callback, 'on_batch_prediction_begin'):
                    callback.on_batch_prediction_begin(batch=batch_idx, logs={'suffix': suffix})

            batch = next(data)
            batch_additional_info = self._get_additional_info()
            loss, loss_info, preds, model_additional_info = self.loss_op(x=batch[0], targets=batch[1],
                                                                         training=False,
                                                                         state='evaluation',
                                                                         return_predictions=True,
                                                                         additional_info=batch_additional_info)

            batch_info = {'val_{}'.format(key): item for key, item in loss_info.items()}
            batch_info['val_loss'] = loss

            batch_info = {key: item.numpy() for key, item in batch_info.items()}

            for key, item in batch_info.items():
                if key not in total_loss:
                    total_loss[key] = item
                else:
                    total_loss[key] += item

            preds = {key: self._parse_predictions(value.numpy(), model_additional_info) for key, value in preds.items()}
            for key, value in preds.items():
                total_preds.setdefault(key, []).extend(value)

            for callback in callbacks:
                if hasattr(callback, 'on_batch_prediction_end'):
                    callback.on_batch_prediction_end(batch=batch_idx, logs={'predictions': preds,
                                                                            'model_additional_info': model_additional_info,
                                                                            'suffix': suffix})

        for callback in callbacks:
            if hasattr(callback, 'on_prediction_end'):
                callback.on_prediction_end(logs={'suffix': suffix})

        total_loss = {key: item / steps for key, item in total_loss.items()}

        return total_loss, total_preds

    def evaluate(self, data, steps, val_y,
                 callbacks=None, repetitions=1, parsed_metrics=None):

        avg_val_info = {}
        avg_val_metrics = {}

        for rep in range(repetitions):
            val_info, val_preds = self.evaluate_and_predict(data=iter(data()), steps=steps,
                                                            callbacks=callbacks, suffix='val')
            for key, value in val_info.items():
                avg_val_info.setdefault(key, []).append(value)

            if parsed_metrics is not None:
                val_y = self._parse_labels(val_y)
                all_val_metrics = compute_iteration_validation_error(predicted_values=val_preds,
                                                                     true_values=val_y,
                                                                     prefix='val',
                                                                     parsed_metrics=parsed_metrics,
                                                                     )
                for key, value in all_val_metrics.items():
                    avg_val_metrics.setdefault(key, []).append(value)

        avg_val_info = {key: np.mean(value, axis=0) for key, value in avg_val_info.items()}
        if parsed_metrics is not None:
            avg_val_metrics = {key: np.mean(value, axis=0) for key, value in avg_val_metrics.items()}

        return avg_val_info, avg_val_metrics

    def fit(self, train_data=None, fixed_train_data=None,
            epochs=1, verbose=1,
            callbacks=None, validation_data=None,
            step_checkpoint=None,
            metrics=None, additional_metrics_info=None, metrics_nicknames=None,
            label_metrics_map=None,
            train_steps=None, val_steps=None,
            val_y=None,
            train_y=None,
            val_inference_repetitions=1):

        # self.validation_data = validation_data
        callbacks = callbacks or []

        for callback in callbacks:
            callback.set_model(model=self)
            res = callback.on_train_begin(logs={'epochs': epochs,
                                                'steps_per_epoch': train_steps})
            if res is not None and type(res) == dict and 'epochs' in res:
                epochs = res['epochs']

        if verbose:
            logger.info('Start Training!')

            if train_steps is not None:
                logger.info('Total batches: {}'.format(train_steps))

        if step_checkpoint is not None:
            if type(step_checkpoint) == float:
                step_checkpoint = int(train_steps * step_checkpoint)
                logger.info('Converting percentage step checkpoint to: {}'.format(step_checkpoint))
            else:
                if step_checkpoint > train_steps:
                    step_checkpoint = int(train_steps * 0.1)
                    logger.info('Setting step checkpoint to: {}'.format(step_checkpoint))

        if type(metrics) != MetricManager:
            parsed_metrics = build_metrics(metrics,
                                           additional_metrics_info,
                                           metrics_nicknames,
                                           label_metrics_map)
        else:
            parsed_metrics = metrics

        parsed_metrics.update_metrics_with_network_info(network=self)

        train_data = iter(train_data())

        # Training
        for epoch in range(epochs):

            if hasattr(self.model, 'stop_training') and self.model.stop_training:
                break

            for callback in callbacks:
                callback.on_epoch_begin(epoch=epoch, logs={'epochs': epochs})

            train_loss = {}
            batch_idx = 0

            # Run epoch
            pbar = tqdm(total=train_steps)
            while batch_idx < train_steps:

                for callback in callbacks:
                    callback.on_batch_begin(batch=batch_idx, logs=None)

                batch_additional_info = self._get_additional_info()
                batch_info, model_additional_info = self.batch_fit(*next(train_data), batch_additional_info)
                batch_info = {key: item.numpy() for key, item in batch_info.items()}

                for callback in callbacks:
                    callback.on_batch_end(batch=batch_idx, logs=batch_info)

                # Update any internal network state
                self._update_internal_state(model_additional_info)

                for key, item in batch_info.items():
                    if key in train_loss:
                        train_loss[key] += item
                    else:
                        train_loss[key] = item

                batch_idx += 1
                pbar.update(1)

            pbar.close()

            train_loss = {key: item / train_steps for key, item in train_loss.items()}
            train_loss_str = {key: float('{:.2f}'.format(value)) for key, value in train_loss.items()}

            val_info = None

            # Compute metrics at the end of each epoch
            callback_additional_args = {}

            if validation_data is not None:
                val_info, all_val_metrics = self.evaluate(data=validation_data,
                                                          steps=val_steps,
                                                          val_y=val_y,
                                                          callbacks=callbacks,
                                                          parsed_metrics=parsed_metrics,
                                                          repetitions=val_inference_repetitions)

                val_info_str = {key: float('{:.2f}'.format(value)) for key, value in val_info.items()}

                if metrics is not None:
                    val_metrics_str_result = {key: float('{:.2f}'.format(value)) for key, value in
                                              all_val_metrics.items()}

                    merged_statistics = merge(train_loss_str, val_info_str)
                    merged_statistics = merge(merged_statistics, val_metrics_str_result)
                    merged_statistics = merge(merged_statistics, {'epoch': epoch + 1})

                    logger.info('\n{}'.format(prettify_statistics(merged_statistics)))

                    callback_additional_args = all_val_metrics
                else:
                    if verbose:
                        merged_statistics = merge(train_loss_str, val_info_str)
                        merged_statistics = merge(merged_statistics, {'epoch': epoch + 1})
                        logger.info(prettify_statistics(merged_statistics))
            else:
                merged_statistics = merge(train_loss_str, {'epoch': epoch + 1})
                logger.info(prettify_statistics(merged_statistics))

            for callback in callbacks:
                callback_args = merge(train_loss, val_info)
                callback_args = merge(callback_args,
                                      callback_additional_args,
                                      overwrite_conflict=False)
                callback.on_epoch_end(epoch=epoch, logs=callback_args)

        for callback in callbacks:
            callback.on_train_end(logs={'name': self.name})

    def _get_input_iterator(self, input_fn, strategy):
        """Returns distributed dataset iterator."""
        # When training with TPU pods, datasets needs to be cloned across
        # workers. Since Dataset instance cannot be cloned in eager mode, we instead
        # pass callable that returns a dataset.
        if not callable(input_fn):
            raise ValueError('`input_fn` should be a closure that returns a dataset.')
        iterator = iter(
            strategy.experimental_distribute_datasets_from_function(input_fn))
        return iterator


    @tf.function
    def batch_fit(self, x, y, additional_info=None):
        loss, loss_info, model_additional_info, grads = self.train_op(x, y, additional_info=additional_info)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        train_loss_info = {'train_{}'.format(key): item for key, item in loss_info.items()}
        train_loss_info['train_loss'] = loss
        return train_loss_info, model_additional_info

    @tf.function
    def distributed_batch_fit(self, inputs, strategy):
        train_loss_info = strategy.run(self.batch_fit, args=inputs)
        train_loss_info = {key: strategy.reduce(tf.distribute.ReduceOp.MEAN, item, axis=None)
                           for key, item in train_loss_info.items()}
        return train_loss_info

    @tf.function
    def batch_predict(self, x):
        additional_info = self._get_additional_info()
        predictions, model_additional_info = self.model(x,
                                                        state='prediction',
                                                        training=False,
                                                        additional_info=additional_info)
        return predictions, model_additional_info

    @tf.function
    def distributed_batch_predict(self, inputs, strategy):
        predictions = strategy.run(self.batch_predict, args=inputs)
        return predictions

    @tf.function
    def distributed_batch_evaluate(self, inputs, strategy):
        val_loss_info = strategy.run(self.batch_evaluate, args=inputs)
        val_loss_info = {key: strategy.reduce(tf.distribute.ReduceOp.MEAN, item, axis=None)
                         for key, item in val_loss_info.items()}
        return val_loss_info

    @tf.function
    def batch_evaluate(self, x, y, additional_info=None):
        loss, loss_info = self.loss_op(x, y, training=False, state='evaluation', additional_info=additional_info)
        val_loss_info = {'val_{}'.format(key): item for key, item in loss_info.items()}
        val_loss_info['val_loss'] = loss
        return val_loss_info

    @tf.function
    def batch_evaluate_and_predict(self, x, y, additional_info=None):
        loss, loss_info, predictions, model_additional_info = self.loss_op(x, y, training=False,
                                                                           state='evaluation',
                                                                           additional_info=additional_info,
                                                                           return_predictions=True)
        val_loss_info = {'val_{}'.format(key): item for key, item in loss_info.items()}
        val_loss_info['val_loss'] = loss
        return val_loss_info, predictions, model_additional_info

    # Model definition

    def train_op(self, x, y, additional_info):
        with tf.GradientTape() as tape:
            loss, loss_info, model_additional_info = self.loss_op(x, y, training=True, additional_info=additional_info)
        grads = tape.gradient(loss, self.model.trainable_variables)
        return loss, loss_info, model_additional_info, grads


class TFClassificationNetwork(TFNetwork, ClassificationNetwork):

    def compute_output_weights(self, y_train, label_list):

        self.class_weights = {}
        for label in label_list:
            if label.type == 'classification':
                label_values = y_train[label.name]
                if len(label_values.shape) > 1:
                    label_values = label_values.ravel()
                label_classes = list(range(label.num_values))
                actual_label_classes = list(set(label_values))
                current_weights = compute_class_weight(class_weight='balanced',
                                                       classes=actual_label_classes, y=label_values)
                remaining_classes = set(label_classes).difference(set(actual_label_classes))

                seen_class_weights = {cls: weight for cls, weight in zip(actual_label_classes, current_weights)}

                for remaining in remaining_classes:
                    seen_class_weights[remaining] = 1.0

                self.class_weights.setdefault(label.name, seen_class_weights)

    def _classification_ce(self, targets, logits, label_name):
        label_class_weights = self.class_weights[label_name]

        if len(logits.shape) > 2:
            logits = tf.reshape(logits, [-1, logits.shape[-1]])
            targets = tf.reshape(targets, [-1])

        if len(logits.shape) == len(targets.shape):
            targets = tf.reshape(targets, [-1])

        # [batch_size,]
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                                       logits=logits)

        if self.weight_predictions:
            label_weights = tf.ones(shape=targets.shape[0], dtype=logits.dtype)
            if len(targets.shape) > 1:
                key_target_classes = tf.argmax(targets, axis=-1)
            else:
                key_target_classes = targets
            for cls, weight in label_class_weights.items():
                to_fill = tf.cast(tf.fill(label_weights.shape, value=weight), logits.dtype)
                label_weights = tf.where(key_target_classes == cls, to_fill, label_weights)

            cross_entropy *= label_weights

        return tf.reduce_mean(cross_entropy)

    def _regression_loss(self, targets, logits):
        difference = tf.math.squared_difference(logits, targets)
        return tf.reduce_mean(difference)

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
            elif label.type == 'regression':
                loss = self._regression_loss(targets=label_targets,
                                             logits=label_logits)
            else:
                raise RuntimeError("Invalid label type -> {}".format(label.type))

            loss_info.setdefault(label.name, loss)

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        return total_loss, loss_info

    def _parse_predictions(self, raw_predictions, model_additional_info):
        return tf.argmax(raw_predictions, axis=-1)


class TFGenerativeNetwork(TFNetwork, GenerativeNetwork):

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

    def _classification_ce(self, targets, logits):

        if len(logits.shape) > 2:
            logits = tf.reshape(logits, [-1, logits.shape[-1]])
            targets = tf.reshape(targets, [-1])

        if len(logits.shape) == len(targets.shape):
            targets = tf.reshape(targets, [-1])

        # [batch_size,]
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                                       logits=logits)

        return tf.reduce_mean(cross_entropy)

    def generate(self, x):
        generated = self.model.generate(input_ids=x['input_ids'],
                                        attention_mask=x['attention_mask'],
                                        max_length=self.max_generation_length)
        generated = {self.label_list[0].name: generated}

        return generated


# Argumentation Mining


class TFTokensSciBert(TFClassificationNetwork):

    def __init__(self, optimizer_args, name='network', additional_data=None,
                 is_pretrained=False, weight_predictions=True,
                 preloaded_name=None, **kwargs):
        super(TFTokensSciBert, self).__init__(name=name,
                                              additional_data=additional_data,
                                              is_pretrained=is_pretrained,
                                              weight_predictions=weight_predictions)
        self.optimizer_args = optimizer_args
        self.preloaded_name = preloaded_name
        self.config = BertConfig(**kwargs)

    def build_model(self, text_info):
        self.max_seq_length = text_info['max_seq_length']
        self.vocab_size = text_info['vocab_size']
        self.label_list = text_info['label_list']

        self.model = M_TokensSciBert(config=self.config,
                                     preloaded_name=self.preloaded_name,
                                     label_list=self.label_list)

        self.optimizer = tf.keras.optimizers.Adam(**self.optimizer_args)

    def loss_op(self, x, targets, training=False, state='training', additional_info=None, return_predictions=False):
        # logits -> [batch_size, classes] or {label: [batch_size, label_classes]}
        logits, model_additional_info = self.model(x, training=training)

        # Cross entropy
        label_name = self.label_list[0].name
        label_logits = logits[label_name]
        targets = targets[label_name]
        label_class_weights = self.class_weights[label_name]
        tokens_mask = model_additional_info['tokens_mask']

        tokens_mask = tf.reshape(tokens_mask, [-1])
        tokens_mask = tf.cast(tokens_mask, label_logits.dtype)
        label_logits = tf.reshape(label_logits, [-1, label_logits.shape[-1]])
        targets = tf.reshape(targets, [-1])

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                                       logits=label_logits)
        cross_entropy *= tokens_mask

        if self.weight_predictions:
            label_weights = tf.ones(shape=targets.shape[0], dtype=label_logits.dtype)
            if len(targets.shape) > 1:
                key_target_classes = tf.argmax(targets, axis=-1)
            else:
                key_target_classes = targets
            for cls, weight in label_class_weights.items():
                to_fill = tf.cast(tf.fill(label_weights.shape, value=weight), label_logits.dtype)
                label_weights = tf.where(key_target_classes == cls, to_fill, label_weights)

            cross_entropy *= label_weights

        cross_entropy = tf.reduce_sum(cross_entropy) / tf.reduce_sum(tokens_mask)

        total_loss = cross_entropy
        loss_info = {label_name: cross_entropy}

        # L2 regularization
        if self.model.losses:
            additional_losses = tf.reduce_sum(self.model.losses)
            total_loss += additional_losses
            loss_info['L2'] = additional_losses

        if return_predictions:
            return total_loss, loss_info, logits, model_additional_info

        return total_loss, loss_info, model_additional_info

    def _parse_predictions(self, raw_predictions, model_additional_info):
        label_name = self.label_list[0].name
        parsed_predictions = tf.argmax(raw_predictions[label_name], axis=-1)
        tokens_mask = model_additional_info['tokens_mask']

        parsed_predictions = tf.where(tokens_mask == 1, parsed_predictions, tf.ones_like(parsed_predictions) * -1)

        return {label_name: parsed_predictions}


class TFTokensSciBertCRF(TFTokensSciBert):

    def build_model(self, text_info):
        self.max_seq_length = text_info['max_seq_length']
        self.vocab_size = text_info['vocab_size']
        self.label_list = text_info['label_list']

        self.model = M_TokensSciBertCRF(config=self.config,
                                        label_list=self.label_list,
                                        preloaded_name=self.preloaded_name)

        self.optimizer = tf.keras.optimizers.Adam(**self.optimizer_args)

    def loss_op(self, x, targets, training=False, state='training', additional_info=None, return_predictions=False):
        # logits -> [batch_size, classes] or {label: [batch_size, label_classes]}
        logits, model_additional_info = self.model(x, training=training)

        # Loss computation
        label_name = self.label_list[0].name
        label_logits = logits[label_name]
        targets = targets[label_name]
        label_class_weights = self.class_weights[label_name]

        crf_loss, _ = tfa.text.crf_log_likelihood(inputs=model_additional_info['potentials'],
                                                  tag_indices=targets,
                                                  sequence_lengths=model_additional_info[
                                                      'sequence_lengths'],
                                                  transition_params=model_additional_info['kernel'])

        if self.weight_predictions:
            label_weights = tf.ones(shape=targets.shape[0], dtype=label_logits.dtype)
            if len(targets.shape) > 1:
                key_target_classes = tf.argmax(targets, axis=-1)
            else:
                key_target_classes = targets
            for cls, weight in label_class_weights.items():
                to_fill = tf.cast(tf.fill(label_weights.shape, value=weight), label_logits.dtype)
                label_weights = tf.where(key_target_classes == cls, to_fill, label_weights)

            crf_loss *= label_weights

        crf_loss = tf.reduce_mean(-crf_loss)
        total_loss = crf_loss
        loss_info = {label_name: crf_loss}

        # L2 regularization
        if self.model.losses:
            additional_losses = tf.reduce_sum(self.model.losses)
            total_loss += additional_losses
            loss_info['L2'] = additional_losses

        if return_predictions:
            return total_loss, loss_info, logits, model_additional_info

        return total_loss, loss_info, model_additional_info

    def _parse_predictions(self, raw_predictions, model_additional_info):
        potentials = model_additional_info['potentials']
        sequence_lengths = model_additional_info['sequence_lengths']
        kernel = model_additional_info['kernel']

        return tfa.text.crf_decode(potentials=potentials,
                                   sequence_length=sequence_lengths,
                                   transition_params=kernel)[0]


class TFComponentsSciBert(TFClassificationNetwork):

    def __init__(self, optimizer_args, name='network', additional_data=None,
                 is_pretrained=False, is_multilabel=False, weight_predictions=True,
                 preloaded_name=None, **kwargs):
        super(TFComponentsSciBert, self).__init__(name=name,
                                                  additional_data=additional_data,
                                                  is_pretrained=is_pretrained,
                                                  weight_predictions=weight_predictions)
        self.optimizer_args = optimizer_args
        self.preloaded_name = preloaded_name
        self.config = BertConfig(**kwargs)

    def build_model(self, text_info):
        self.max_seq_length = text_info['max_seq_length']
        self.vocab_size = text_info['vocab_size']
        self.label_list = text_info['label_list']

        self.model = M_ComponentsSciBert(config=self.config,
                                         preloaded_name=self.preloaded_name,
                                         label_list=self.label_list)

        self.optimizer = tf.keras.optimizers.Adam(**self.optimizer_args)

    def loss_op(self, x, targets, training=False, state='training', additional_info=None, return_predictions=False):
        # logits -> [batch_size, classes] or {label: [batch_size, label_classes]}
        logits, model_additional_info = self.model(x, training=training)

        # Cross entropy
        total_loss, loss_info = self._compute_losses(targets=targets,
                                                     logits=logits,
                                                     label_list=self.label_list)

        # L2 regularization
        if self.model.losses:
            additional_losses = tf.reduce_sum(self.model.losses)
            total_loss += additional_losses
            loss_info['L2'] = additional_losses

        if return_predictions:
            return total_loss, loss_info, logits, model_additional_info

        return total_loss, loss_info, model_additional_info


class TFModelFactory(ModelFactory):

    @classmethod
    def get_supported_values(cls):
        supported_values = super(TFModelFactory, cls).get_supported_values()
        return merge(supported_values, {
            'drinventor_tf_tokens_scibert': TFTokensSciBert,
            'drinventor_tf_tokens_scibert_crf': TFTokensSciBertCRF,
            'drinventor_tf_components_scibert': TFComponentsSciBert,
        })
