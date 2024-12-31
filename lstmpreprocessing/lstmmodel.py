import keras
from keras.src.models import Sequential, Model
from keras.src.layers import Dense, Input, LSTM


class LSTMModel:
    def __init__(self,lstm_hidden_units,dataset,seq_length,learning_rate,feature_space,epochs,batch_size):
        self.dataset = dataset
        self.seq_length = seq_length
        self.feature_space= feature_space
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.lstm_hidden_units = lstm_hidden_units

    def model_build(self):
        input_shape = (self.seq_length, self.feature_space)

        # Initialize the model
        model = Sequential()

        # Flatten the rows and columns into a linear dimensional vector
        model.add(Input(input_shape))

        for units in self.lstm_hidden_units[:-1]:
            model.add(LSTM(units, return_sequences = True))
        model.add(LSTM(self.lstm_hidden_units[-1]))

        # Add another fully connected layer to parse the output
        model.add(Dense(self.feature_space, activation='softmax'))

        model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=self.learning_rate),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=[keras.metrics.CategoricalAccuracy(),]
        )

        model.fit(self.dataset, epochs=self.epochs, batch_size=self.batch_size, verbose=2)

        return Model(inputs=model.inputs,outputs=model.layers[1].output)

            
        
