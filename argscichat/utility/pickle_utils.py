
import pickle


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    return data


def save_pickle(filepath, data):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)