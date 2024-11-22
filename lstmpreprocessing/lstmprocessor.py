import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras import ops
import pandas as pd
from sklearn.utils import shuffle
from lstmmodel import lstmmodel
import os


lstm_hidden_units = [128,128]
batch_size_dataset = 1857
batch_size_final = 64
input_sequence_length = 12
forecast_horizon = 3
feature_space = 26
lstm_learning_rate = 0.01
lstm_epochs = 1
intermediate_node_output = 32
number_of_seq = 255


x_array=np.load(os.path.abspath(f"../auxillaryfiles/protein_encoded_sequence.npy"))
y_array=np.load(os.path.abspath(f"../auxillaryfiles/protein_family_encoded_labels.npy"))
node_features_lstm = np.load(os.path.abspath(f"../auxillaryfiles/nodefeatures.npy"))
flatten_xarray = x_array.reshape(x_array.shape[0]*x_array.shape[1],x_array.shape[2])

def create_tf_dataset(
    data_array: np.ndarray,
    input_sequence_length: int,
    forecast_horizon: int,
    batch_size: int = 128,
    shuffle=False,
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

train_dataset = (
    create_tf_dataset(flatten_xarray, input_sequence_length, forecast_horizon, batch_size_dataset))




lstm_model = lstmmodel(dataset = train_dataset, seq_length = input_sequence_length,
                  learning_rate = lstm_learning_rate,
                  epochs = lstm_epochs, feature_space=feature_space,
                  batch_size = batch_size_final,
                  intermediate_node_output=intermediate_node_output,
                  lstm_hidden_units=lstm_hidden_units)

lstm_layer = lstm_model.modelbuild()

lstm_out = np.zeros((node_features_lstm.shape[0],intermediate_node_output))



for i in range (0, len(node_features_lstm)-batch_size_final,batch_size_final):
    lstm_out[i:i+batch_size_final]=lstm_layer.predict(node_features_lstm[i:i+batch_size_final])




lstm_outputs = lstm_out.flatten()

padding_size = number_of_seq - (lstm_outputs.shape[0] % number_of_seq)

lstm_padded_array = np.pad(lstm_outputs, (0, padding_size), mode='constant')

lstm_padded_array = lstm_padded_array.reshape(number_of_seq,-1)


#np.save("../auxillaryfiles/lstm_padded_array.npy",lstm_padded_array)




