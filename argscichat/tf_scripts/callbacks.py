import os
from copy import deepcopy

import numpy as np
from argscichat.generic.callbacks import BaseCallback
from argscichat.generic.factory import Factory
from tensorflow.python.platform import tf_logging as logging
from argscichat.utility.log_utils import Logger
from argscichat.utility.python_utils import merge

logger = Logger.get_logger(__name__)


class EarlyStopping(BaseCallback):
    """Stop training when a monitored quantity has stopped improving.
    Arguments:
        monitor: Quantity to be monitored.
        min_delta: Minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: Number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: One of `{"auto", "min", "max"}`. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity.
            Training will stop if the model doesn't show improvement over the
            baseline.
        restore_best_weights: Whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    Example:
    ```python
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    # This callback will stop the training when there is no improvement in
    # the validation loss for three consecutive epochs.
    model.fit(data, labels, epochs=100, callbacks=[callback],
        validation_data=(val_data, val_labels))
    ```
    """

    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            logging.warning('EarlyStopping mode %s is unknown, '
                            'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            # logger.info('[EarlyStopping] New best value: {}'.format(self.best))
            if self.restore_best_weights:
                self.best_weights = self.model.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best epoch.')
                    self.model.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning('Early stopping conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))
        return monitor_value


class PredictionRetriever(BaseCallback):
    """
    Simple callback that allows to extract and save attention tensors during prediction phase.
    Extraction is simply implemented as attribute inspection.
    """

    def __init__(self, save_suffix=None):
        super(PredictionRetriever, self).__init__()
        self.start_monitoring = False
        self.stored_network_predictions = None
        self.save_suffix = save_suffix

    def set_model(self, model):
        self.model = model

    def on_build_model_begin(self, logs=None):
        self.network = logs['network']

    def on_prediction_begin(self, logs=None):
        self.start_monitoring = True

    def on_batch_prediction_end(self, batch, logs=None):
        if self.start_monitoring:

            # [batch_size, hops, mem_size]
            model_additional_info = logs['model_additional_info']

            if type(model_additional_info) == dict:
                network_predictions = model_additional_info['raw_predictions']
            else:
                network_predictions = model_additional_info

            if type(network_predictions) == dict:
                network_predictions = {key: value.numpy() for key, value in network_predictions.items()}
            else:
                network_predictions = network_predictions.numpy()

            if batch == 0:
                # [batch_size, hops, mem_size]
                self.stored_network_predictions = network_predictions
            else:
                # [samples, hops, mem_size]
                if type(network_predictions) == dict:
                    for key, value in network_predictions.items():
                        self.stored_network_predictions[key] = np.append(self.stored_network_predictions[key], value,
                                                                         axis=0)
                else:
                    self.stored_network_predictions = np.append(self.stored_network_predictions, network_predictions,
                                                                axis=0)

    def on_prediction_end(self, logs=None):
        if self.start_monitoring:
            # Saving
            save_name = self.network.name
            if self.save_suffix is not None:
                save_name += '_{}'.format(self.save_suffix)
            filepath = os.path.join(self.save_path,
                                    '{}_raw_predictions.npy'.format(save_name))
            np.save(filepath, self.stored_network_predictions)

            # Resetting
            self.start_monitoring = None
            self.stored_network_predictions = {}


class TrainingLogger(BaseCallback):

    def __init__(self, suffix=None):
        super(TrainingLogger, self).__init__()
        self.suffix = suffix
        self.info = {}

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            assert type(logs) == dict

            for key, item in logs.items():
                if key != 'strategy':
                    self.info.setdefault(key, []).append(deepcopy(item))

    def on_train_end(self, logs=None):
        if logs is not None and 'name' in logs:
            self.suffix = logs['name']

        if self.suffix is None:
            filename = 'info.npy'
        else:
            filename = '{}_info.npy'.format(self.suffix)
        savepath = os.path.join(self.save_path, filename)

        np.save(savepath, self.info)

        self.info = {}


class TFCallbackFactory(Factory):

    @classmethod
    def get_supported_values(cls):
        supported_values = super(TFCallbackFactory, cls).get_supported_values()
        return merge(supported_values, {
            'earlystopping': EarlyStopping,
            'traininglogger': TrainingLogger,
            'predictionretriever': PredictionRetriever
        })
