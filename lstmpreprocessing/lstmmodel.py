import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, LSTM
import tensorflow_datasets as tfds
from keras.models import Model


class lstmmodel():
    def __init__(self,lstm_hidden_units,intermediate_node_output,dataset,seq_length,learning_rate,feature_space,epochs,batch_size):
        self.dataset = dataset
        self.seq_length = seq_length
        self.feature_space= feature_space
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.intermediate_node_output = intermediate_node_output
        self.lstm_hidden_units = lstm_hidden_units

    def modelbuild(self):

            
        input_shape = (self.seq_length, self.feature_space)
        
        inputs = tf.keras.Input(input_shape)

        # Initialize the model
        model = Sequential()

        # Flatten the rows and columns into a linear dimensional vector
        model.add(Input(input_shape))

        for units in self.lstm_hidden_units[:-1]:
            model.add(LSTM(units, return_sequences = True))

        model.add(LSTM(self.lstm_hidden_units[-1]))

        model.add(Dense(self.intermediate_node_output, activation='relu'))

        # Add another fully connected layer to parse the output
        model.add(Dense(self.feature_space, activation='softmax'))

        model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=self.learning_rate),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=[keras.metrics.CategoricalAccuracy(),]
        )

        model.fit(self.dataset, epochs=self.epochs, batch_size=self.batch_size, verbose=2)
        
        return Model(inputs=model.inputs,outputs=model.layers[2].output)          

            
        
