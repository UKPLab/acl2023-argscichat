

class BaseCallback(object):

    def __init__(self):
        self.model = None
        self.save_path = None

    def set_model(self, model):
        self.model = model

    def set_save_path(self, save_path):
        self.save_path = save_path

    def on_build_model_begin(self, logs=None):
        pass

    def on_build_model_end(self, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_prediction_begin(self, logs=None):
        pass

    def on_prediction_end(self, logs=None):
        pass

    def on_batch_prediction_begin(self, batch, logs=None):
        pass

    def on_batch_prediction_end(self, batch, logs=None):
        pass
