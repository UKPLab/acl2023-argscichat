"""

This is an utility module to manage operations on json files

"""

import numpy as np
import simplejson
from sklearn.base import BaseEstimator


def load_json(file_path):
    """
    Loads JSON data from file. Special python objects are
    handled by custom serializers.

    :param file_path: path of .json file from which to load data
    :return: JSON loaded data
    """

    with open(file_path, mode='r') as f:
        data = simplejson.load(f, object_hook=_from_json)

    return data


def save_json(filepath, data):
    """
    Saves data in JSON format to file. Special python objects are
    handled by custom serializers.

    :param filepath: path of .json file in which to save data
    :param data: data to save
    """

    with open(filepath, mode='w') as f:
        simplejson.dump(data, f, default=_to_json, tuple_as_array=False, indent=4)


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = simplejson.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_path))


def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(simplejson.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data


def _to_json(python_object):
    """
    Custom JSON serializer

    :param python_object: object to serialize
    :return: custom serialized python object
    """

    if isinstance(python_object, tuple):
        python_object = {'__class__': 'tuple',
                         '__value__': list(python_object)}
    elif isinstance(python_object, BaseEstimator):
        python_object = {'__class__': 'generic_estimator',
                         '__name__': type(python_object).__name__,
                         }
    elif isinstance(python_object, np.ndarray):
        python_object = {'__class__': 'numpy_ndarray',
                         '__value__': python_object.tolist()}
    elif isinstance(python_object, np.int64):
        return int(python_object)
    elif isinstance(python_object, set):
        python_object = {'__class__': 'set',
                         '__value__': list(python_object)}
    elif isinstance(python_object, object):
        try:
            python_object = {'__class__': 'generic_object',
                             '__name__': type(python_object).__name__,
                             '__value__': python_object.__dict__.items()}
        except AttributeError:
            python_object = None
    else:
        raise TypeError(repr(python_object) + ' of type: ' + str(type(python_object)) + ' is not JSON serializable')

    return python_object


def _from_json(json_object):
    """
    Custom JSON de-serializer

    :param json_object: custom serialized python object
    :return: custom de-serialized python object
    """

    if '__class__' in json_object:
        if json_object['__class__'] == 'tuple':
            return tuple(json_object['__value__'])
        elif json_object['__class__'] == 'generic_estimator':
            return json_object['__name__']
        elif json_object['__class__'] == 'numpy_ndarray':
            return np.array(json_object['__value__'])
        elif json_object['__class__'] == 'set':
            return set(json_object['__value__'])
        elif json_object['__class__'] == 'generic_object':
            try:
                return json_object['__name__'](*json_object['__value__'])
            except TypeError:
                return None

    return json_object
