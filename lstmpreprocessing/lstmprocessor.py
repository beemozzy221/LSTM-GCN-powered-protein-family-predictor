import os
import numpy as np
import tensorflow as tf
from lstmmodel import LSTMModel


lstm_hidden_units = [64, 64]
batch_size_dataset = 2033
batch_size_final = 64
lstm_input_sequence_length = 12
lstm_forecast_horizon = 3
feature_space = 26
lstm_learning_rate = 0.01
lstm_epochs = 5


encoded_features = np.load(os.path.abspath(f"../auxillaryfiles/protein_encoded_sequence.npy"), allow_pickle=True)
encoded_targets = np.load(os.path.abspath(f"../auxillaryfiles/protein_family_encoded_labels.npy"), allow_pickle=True)
lstm_saved_weights = r"../savedweights/lstmweights.weights.h5"

encoded_features = encoded_features.astype('int32')
encoded_targets = encoded_targets.astype('int32')

def create_tf_dataset(
    data_array: np.ndarray,
    input_sequence_length: int,
    forecast_horizon: int,
    batch_size: int = 128,
    ):

    target_offset = input_sequence_length + forecast_horizon - 1
    
    dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
    data=data_array,
    targets=data_array[target_offset:],  # Targets start from the next timestep after each input sequence
    sequence_length=input_sequence_length,
    batch_size=batch_size,
    sequence_stride = 1,
    shuffle=False)

    return dataset

#Create temporal sequences
flatten_features = encoded_features.reshape(encoded_features.shape[0] * encoded_features.shape[1], encoded_features.shape[2])
train_dataset = create_tf_dataset(flatten_features, lstm_input_sequence_length, lstm_forecast_horizon, batch_size_dataset)

#Initialize the model
lstm_model = LSTMModel(dataset = train_dataset, seq_length = lstm_input_sequence_length,
                       learning_rate = lstm_learning_rate,
                       epochs = lstm_epochs, feature_space=feature_space,
                       batch_size = batch_size_final,
                       lstm_hidden_units=lstm_hidden_units)

#Build model
lstm_layer = lstm_model.model_build()

# Save the model parameters
lstm_layer.save_weights(lstm_saved_weights)








