import warnings
import tensorflow as tf

class CusEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=0, monitor="loss", restore_best_weights=False):
        super().__init__()
        self.patience = patience
        self.monitor = monitor
        self.epoch_info = {}
        self.wait = 0
        self.stopped_epoch = 0
        self._model = None
        self.restore_best_weights = restore_best_weights

    def on_epoch_end(self, epoch, logs=None):
        if self._model is None:
            raise Exception("No model is set!")
        if self.monitor == "loss":
            self.epoch_info[epoch] = {"loss": logs["loss"],
                                      "weights": self._model.get_weights()}

            if epoch > self.patience:
                if self.wait > self.patience:
                    if tf.round(self.epoch_info[epoch]["loss"]*100)/100 == tf.round(self.epoch_info[epoch-1]["loss"]*100)/100:
                        if self.restore_best_weights:
                            self._model.set_weights(self.epoch_info[epoch-self.patience]["weights"])
                        self.stopped_epoch = epoch
                        print(f"Model stopped by early stopping callback at epoch: {self.stopped_epoch}")
                        self.model.stop_training = True
                    else:
                        self.wait += 1
                else:
                    self.wait += 1
        else:
            warnings.warn("No value to monitor, callback hibernating")

    def set_model(self, model):
        self._model = model
