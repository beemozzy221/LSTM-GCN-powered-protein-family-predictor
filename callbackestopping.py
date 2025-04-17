import warnings
import tensorflow as tf

class CusEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=0, monitor="val_loss", restore_best_weights=False):
        super().__init__()
        self.patience = patience
        self.monitor = monitor
        self.epoch_info = {}
        self.wait = 0
        self.stopped_epoch = 0
        self._model = None
        self.restore_best_weights = restore_best_weights
        self.temp_weights = {}

    def on_epoch_end(self, epoch, logs=None):
        if self._model is None:
            raise Exception("No model is set!")
        if self.patience == 0:
            raise Exception("Patience value should be a value greater than 0")
        if self.monitor == "val_loss":
            self.epoch_info[epoch] = {"val_loss": logs["val_loss"]}

            if self.restore_best_weights:
                if epoch % self.patience == 0:
                    self.temp_weights["previous_weights"] = self._model.get_weights()

            if epoch > self.patience:
                if tf.round(self.epoch_info[epoch]["val_loss"] * 100) / 100 == tf.round(
                        self.epoch_info[epoch - 1]["val_loss"] * 100) / 100:
                    self.wait += 1

                else:
                    self.wait = 0

                if self.wait > self.patience:
                    if self.restore_best_weights:
                        self._model.set_weights(temp_weights["previous_weights"])
                    self.stopped_epoch = epoch
                    print(f"Model stopped by early stopping callback at epoch: {self.stopped_epoch}")
                    
                    raise StopIteration("Model early stopped")

        else:
            warnings.warn("No value to monitor, callback hibernating")

    def set_model(self, model):
        self._model = model
