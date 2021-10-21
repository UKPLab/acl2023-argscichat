"""

Generic label wrappers

"""

from transformers.tokenization_utils_base import BatchEncoding


class Label(object):

    def __init__(self, name, type, values=None):
        self.name = name
        self.type = type
        self.values = values
        self.weights = None

    @property
    def num_values(self):
        return len(self.values) if self.values is not None else None

    def set_values(self, values):
        self.values = values

    def convert_label_value(self, label_value, **kwargs):
        raise NotImplementedError()


class ClassificationLabel(Label):

    def __init__(self, **kwargs):
        super(ClassificationLabel, self).__init__(type='classification', **kwargs)
        assert self.values is not None

        self.label_map = {}
        for value_idx, value in enumerate(self.values):
            self.label_map.setdefault(value, value_idx)

    def convert_label_value(self, label_value, **kwargs):
        return self.label_map[label_value]


class RegressionLabel(Label):

    def __init__(self, **kwargs):
        super(RegressionLabel, self).__init__(type='regression', **kwargs)
        assert self.values is None

        self.label_map = None

    def convert_label_value(self, label_value, **kwargs):
        return label_value


class GenerativeLabel(Label):

    def __init__(self, **kwargs):
        super(GenerativeLabel, self).__init__(type='generation', **kwargs)
        assert self.values is None
        self.label_map = {}

    def set_values(self, values):
        assert type(values) == dict
        self.label_map = values

    def convert_label_value(self, label_value, **kwargs):
        assert 'tokenizer' in kwargs

        tokenized = kwargs['tokenizer'].tokenize(label_value)

        # BERT special case (ignore special tokens)
        if type(tokenized) == BatchEncoding:
            tokenized = tokenized['input_ids'][1:-1]

        return tokenized


# Labels Wrapper

class LabelList(object):

    def __init__(self, labels=None):
        self.labels = labels if labels is not None else []
        self.added_state = set()
        self.labels_dict = {}

        if labels is not None:
            for label in labels:
                self.labels_dict.setdefault(label.name, label)

    def __iter__(self):
        return self.labels.__iter__()

    def append(self, label):
        self.labels.append(label)
        if self.labels_dict is not None:
            self.labels_dict.setdefault(label.name, label)

    def __getitem__(self, item):
        return self.labels[item]

    def add_state(self, property_name, property_value):
        setattr(self, property_name, property_value)
        self.added_state.add(property_name)

    def get_state(self, property_name):
        return getattr(self, property_name, None)

    def get_added_state(self):
        return {key: self.get_state(key) for key in self.added_state}

    def as_dict(self):
        for label in self.labels:
            self.labels_dict.setdefault(label.name, label)

        return self.labels_dict

    def get_label_names(self):
        return [label.name for label in self.labels]

    def get_labels_mapping(self):
        return {label.name: label.label_map for label in self.labels}