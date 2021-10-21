import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection._split import _BaseKFold

from argscichat.generic.metrics import MetricFactory, MetricManager
from argscichat.utility.json_utils import save_json, load_json
from argscichat.utility.log_utils import Logger
from argscichat.utility.python_utils import merge

logger = Logger.get_logger(__name__)


def build_metrics(error_metrics, additional_metrics_info=None,
                  metrics_nicknames=None, label_metrics_map=None):
    """
    Build validation metrics from given metrics name and problem type

    :param error_metrics: list of error metrics names (strings)
    :return: list of pointers to error metrics functions
    """

    if metrics_nicknames is None:
        metrics_nicknames = [metric_name for metric_name in error_metrics]

    additional_metrics_info = additional_metrics_info or [{} for _ in error_metrics]

    parsed_metrics = MetricManager(label_metrics_map=label_metrics_map)

    for metric, metric_info, metric_nickname in zip(error_metrics,
                                                    additional_metrics_info,
                                                    metrics_nicknames):
        parsed_metrics.add_metric(MetricFactory.factory(metric, name=metric_nickname, **metric_info))

    parsed_metrics.finalize()

    return parsed_metrics


def show_data_shapes(data, key):
    """
    Prints data shape (it could be a nested structure).
    Currently, the following nested structures are supported:

        1) List
        2) Numpy.ndarray
        3) Dict
    """

    if type(data) is np.ndarray:
        data_shape = data.shape
    elif type(data) is list:
        data_shape = [len(item) for item in data]
    else:
        data_shape = [(key, len(item)) for key, item in data.items()]

    logger.info('[{0}] Shape(s): {1}'.format(key, data_shape))


def _compute_iteration_validation_error(parsed_metrics, true_values, predicted_values,
                                        prefix=None,
                                        label_suffix=None):
    """
    Computes each given metric value, given true and predicted values.

    :param parsed_metrics: list of metric functions (typically sci-kit learn metrics)
    :param true_values: ground-truth values
    :param predicted_values: model predicted values
    :return: dict as follows:

        key: metric.__name__
        value: computed metric value
    """

    fold_error_info = {}

    if type(predicted_values) == np.ndarray and type(true_values) == np.ndarray:
        if len(true_values.shape) > 1 and true_values.shape[1] > 1 and len(np.where(true_values[0])[0]) == 1:
            true_values = np.argmax(true_values, axis=1)
            predicted_values = np.argmax(predicted_values, axis=1)

        true_values = true_values.ravel()
        predicted_values = predicted_values.ravel()

    for metric in parsed_metrics:
        signal_error = metric(y_true=true_values, y_pred=predicted_values)
        metric_name = metric.name

        if label_suffix is not None:
            metric_name = '{0}_{1}'.format(label_suffix, metric_name)
        if prefix is not None:
            metric_name = '{0}_{1}'.format(prefix, metric_name)

        fold_error_info.setdefault(metric_name, signal_error)

    return fold_error_info


def compute_iteration_validation_error(parsed_metrics, true_values, predicted_values, prefix=None):
    """
    Computes each given metric value, given true and predicted values.

    :param parsed_metrics: list of metric functions (typically sci-kit learn metrics)
    :param true_values: ground-truth values
    :param predicted_values: model predicted values
    :return: dict as follows:

        key: metric.__name__
        value: computed metric value
    """

    fold_error_info = {}

    for key, true_value_set in true_values.items():

        key_metrics = parsed_metrics.get_metrics(key)
        pred_value_set = predicted_values[key]
        if type(pred_value_set) == np.ndarray:
            pred_value_set = np.reshape(pred_value_set, true_value_set.shape)
        key_error_info = _compute_iteration_validation_error(parsed_metrics=key_metrics,
                                                             true_values=true_value_set,
                                                             predicted_values=pred_value_set,
                                                             label_suffix=key,
                                                             prefix=prefix)
        fold_error_info = merge(fold_error_info, key_error_info)

    return fold_error_info




class PrebuiltCV(_BaseKFold):
    """
    Simple CV wrapper for custom fold definition.
    """

    def __init__(self, cv_type='kfold', held_out_key='validation', **kwargs):
        super(PrebuiltCV, self).__init__(**kwargs)
        self.folds = None
        self.key_listing = None
        self.held_out_key = held_out_key

        if cv_type == 'kfold':
            self.cv = KFold(n_splits=self.n_splits, shuffle=self.shuffle)
        elif cv_type == 'stratifiedkfold':
            self.cv = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle)
        else:
            raise AttributeError('Invalid cv_type! Got: {}'.format(cv_type))

    def build_folds(self, X, y):
        self.folds = {}
        for fold, (train_indexes, held_out_indexes) in enumerate(self.cv.split(X, y)):
            self.folds['fold_{}'.format(fold)] = {
                'train': train_indexes,
                self.held_out_key: held_out_indexes
            }

    def build_all_sets_folds(self, X, y, validation_n_splits=None):
        assert self.held_out_key == 'test'

        validation_n_splits = self.n_splits if validation_n_splits is None else validation_n_splits

        self.folds = {}
        for fold, (train_indexes, held_out_indexes) in enumerate(self.cv.split(X, y)):
            sub_X = X[train_indexes]
            sub_y = y[train_indexes]

            self.cv.n_splits = validation_n_splits
            sub_train_indexes, sub_val_indexes = list(self.cv.split(sub_X, sub_y))[0]
            self.cv.n_splits = self.n_splits

            self.folds['fold_{}'.format(fold)] = {
                'train': train_indexes[sub_train_indexes],
                self.held_out_key: held_out_indexes,
                'validation': train_indexes[sub_val_indexes]
            }

    def load_dataset_list(self, load_path):
        with open(load_path, 'r') as f:
            dataset_list = [item.strip() for item in f.readlines()]

        return dataset_list

    def save_folds(self, save_path, tolist=False):

        if tolist:
            to_save = {}
            for fold_key in self.folds:
                for split_set in self.folds[fold_key]:
                    to_save.setdefault(fold_key, {}).setdefault(split_set, self.folds[fold_key][split_set].tolist())
            save_json(save_path, to_save)
        else:
            save_json(save_path, self.folds)

    def load_folds(self, load_path):
        self.folds = load_json(load_path)
        self.n_splits = len(self.folds)
        key_path = load_path.split('.json')[0] + '.txt'
        with open(key_path, 'r') as f:
            self.key_listing = list(map(lambda item: item.strip(), f.readlines()))
        self.key_listing = np.array(self.key_listing)

    def _iter_test_indices(self, X=None, y=None, groups=None):

        fold_list = sorted(list(self.folds.keys()))

        for fold in fold_list:
            yield self.folds[fold][self.held_out_key]

    def split(self, X, y=None, groups=None):

        fold_list = sorted(list(self.folds.keys()))

        for fold in fold_list:
            val_indexes = self.key_listing[self.folds[fold]['validation']] if 'validation' in self.folds[fold] else None
            test_indexes = self.key_listing[self.folds[fold]['test']] if 'test' in self.folds[fold] else None

            assert val_indexes is not None or test_indexes is not None

            yield self.key_listing[self.folds[fold]['train']], \
                  val_indexes, \
                  test_indexes
