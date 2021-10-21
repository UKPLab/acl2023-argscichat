"""

Generic data example wrappers

"""


class TextExample(object):

    def __init__(self, text, label=None):
        self.text = text
        self.label = label

    def get_data(self):
        return self.text


class PairedTextExample(object):

    def __init__(self, source, target, distance, labels=None):
        self.source = source
        self.target = target
        self.distance = distance
        self.labels = labels

    def get_data(self):
        return ' '.join([self.source, self.target])


class TokensExample(object):

    def __init__(self, tokens, tokens_labels=None):
        self.tokens = tokens
        self.tokens_labels = tokens_labels

    def get_data(self):
        return self.tokens


# List wrappers

class ExampleList(object):

    def __init__(self):
        self.content = []
        self.added_state = set()

    def __iter__(self):
        return self.content.__iter__()

    def append(self, item):
        self.content.append(item)

    def __len__(self):
        return len(self.content)

    def __getitem__(self, item):
        return self.content[item]

    def add_state(self, property_name, property_value):
        setattr(self, property_name, property_value)
        self.added_state.add(property_name)

    def get_state(self, property_name):
        return getattr(self, property_name, None)

    def get_added_state(self):
        return {key: self.get_state(key) for key in self.added_state}

    def get_data(self):
        return [item.get_data() for item in self.content]