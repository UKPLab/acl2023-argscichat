from sklearn.metrics import f1_score
import numpy as np
from argscichat.utility.metric_utils import compute_tokens_f1
from argscichat.generic.factory import Factory
from argscichat.utility.python_utils import merge


class Metric(object):

    def __init__(self, name):
        self.name = name

    def __call__(self, y_pred, y_true, **kwargs):
        raise NotImplementedError()

    def retrieve_parameters_from_network(self, network):
        pass

    def reset(self):
        raise NotImplementedError()


class F1Score(Metric):

    def __call__(self, y_pred, y_true, **kwargs):
        y_pred = np.array(y_pred) if type(y_pred) != np.ndarray else y_pred
        y_true = np.array(y_true) if type(y_true) != np.ndarray else y_true

        y_pred = y_pred.ravel()
        y_true = y_true.ravel()

        return f1_score(y_true=y_true, y_pred=y_pred, **kwargs)


class MaskF1(Metric):

    def __init__(self, average='binary', **kwargs):
        super(MaskF1, self).__init__(**kwargs)
        self.average = average

    def __call__(self, y_pred, y_true, **kwargs):
        y_pred = np.array(y_pred) if type(y_pred) != np.ndarray else y_pred
        y_true = np.array(y_true) if type(y_true) != np.ndarray else y_true

        y_pred = y_pred.ravel()
        y_true = y_true.ravel()

        return f1_score(y_true=y_true, y_pred=y_pred, average=self.average, **kwargs)


class TokensF1(Metric):

    def __init__(self, average='macro', **kwargs):
        super(TokensF1, self).__init__(**kwargs)
        self.average = average

    def __call__(self, y_pred, y_true, **kwargs):
        y_pred = np.array(y_pred) if type(y_pred) != np.ndarray else y_pred

        y_pred = y_pred.ravel()
        y_true = y_true.ravel()

        pad_indexes = np.where(y_pred == -1)[0]
        sample_weight = np.ones_like(y_pred)
        sample_weight[pad_indexes] = 0.0
        y_pred[pad_indexes] = 0

        return f1_score(y_true=y_pred, y_pred=y_true, sample_weight=sample_weight, average=self.average)


class GeneratedF1(Metric):

    def __init__(self, average='macro', **kwargs):
        super(GeneratedF1, self).__init__(**kwargs)
        self.average = average
        self.vocab = None

    def retrieve_parameters_from_network(self, network):
        assert hasattr(network, 'tokenizer')
        self.vocab = network.tokenizer.vocab

    def __call__(self, y_pred, y_true, **kwargs):
        avg_f1 = []
        for pred, gold in zip(y_pred, y_true):
            avg_f1.append(compute_tokens_f1(a_pred=pred, a_gold=gold))

        return np.mean(avg_f1)

    def reset(self):
        self.vocab = None


class MetricManager(object):

    def __init__(self, label_metrics_map):
        self.metrics = []
        self.label_metrics_map = label_metrics_map
        self.metric_map = {}

    def add_metric(self, metric):
        self.metrics.append(metric)

    def finalize(self):
        for metric in self.metrics:
            self.metric_map[metric.name] = metric

    def update_metrics_with_network_info(self, network):
        for metric in self.metrics:
            metric.retrieve_parameters_from_network(network)

    def get_metrics(self, label):
        if self.label_metrics_map is not None:
            names = self.label_metrics_map[label]
            return [self.metric_map[name] for name in names]
        else:
            return self.metrics


class MetricFactory(Factory):

    @classmethod
    def get_supported_values(cls):
        supported_values = super(MetricFactory, cls).get_supported_values()
        return merge(supported_values, {
            'f1_score': F1Score,
            'generated_f1': GeneratedF1,
            'mask_f1': MaskF1,
            'tokens_f1': TokensF1
        })
