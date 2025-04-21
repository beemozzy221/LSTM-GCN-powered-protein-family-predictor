import keras
import numpy as np
import tensorflow as tf
from keras.src import layers, ops
from keras.src.layers import LSTM
from graphconvo import GraphConvLayer

def create_ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []
    
    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    return keras.Sequential(fnn_layers, name=name)

def create_lstm (hidden_units, name=None):
    lstm_layers = []

    for units in hidden_units[:-1]:
        lstm_layers.append(LSTM(units, return_sequences=True, trainable=False))
    lstm_layers.append(LSTM(hidden_units[-1], trainable=False))

    return keras.Sequential(lstm_layers, name=name)

class GNNNodeClassifier(tf.keras.Model):
    def __init__(
        self,
        num_classes,
        lstm_weights,
        lstm_hidden_units,
        hidden_units,
        protein_length,
        protein_chars,
        aggregation_type="sum",
        combination_type="concat",
        dropout_rate=0.2,
        normalize=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        #Get sequence lengths
        self.protein_length = protein_length
        self.protein_chars = protein_chars

        #Unpack to give to get_config
        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.lstm_hidden_units = lstm_hidden_units
        self.lstm_weights = lstm_weights

        #Meant to be used in def call
        self.g_obj = None

        #Create preprocess layer
        #self.preprocess = create_ffn(hidden_units, dropout_rate, name = "preprocess")

        # Create the first GraphConv layer.
        self.conv1 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv1",
        )
        # Create the second GraphConv layer
        self.conv2 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv2",
        )

        # Create a postprocess layer
        self.postprocess = create_ffn(hidden_units,dropout_rate, name="post_process")
        #Create LSTM layers
        self.preprocess_lstm_layer = create_lstm(lstm_hidden_units, name="lstm_processor")
        # Create a compute logits layer
        self.compute_logits = layers.Dense(units=num_classes, name="logits")

        #Build LSTM layer and load weights
        self.preprocess_lstm_layer.build(input_shape=(None, protein_length, protein_chars))
        self.preprocess_lstm_layer.load_weights(lstm_weights)

    def call(self, inputs):
        edges, edge_weights = self.g_obj
        #Unpack graph objects
        node_features = inputs
        node_features = tf.cast(node_features, dtype="int32")
        edges = tf.cast(edges, dtype="int32")

        #For initializing the edge weights
        if edge_weights is None:
            edge_weights = tf.ones(shape=edges.shape[1])
        # Scale edge_weights to sum to 1.
        edge_weights = edge_weights / tf.math.reduce_sum(edge_weights)

        #LSTM preprocessing layer
        x1 = self.preprocess_lstm_layer(node_features)

        # Apply the first graph conv layer
        x2 = self.conv1((x1, edges, edge_weights))

        # Apply the second graph conv layer
        x3 = self.conv2((x2, edges, edge_weights))

        # Postprocess node embedding
        node_embeddings = self.postprocess(x3)

        # Compute logits
        return self.compute_logits(node_embeddings)

    def get_config(self):
        config = super(self).get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "hidden_units": self.hidden_units,
                "graph_info": self.graph_info,
                "num_classes": self.num_classes,
                "lstm_hidden_units": self.lstm_hidden_units,
                "lstm_weights": self.lstm_weights,
                "protein_chars": self.protein_chars,
                "protein_length": self.protein_length
            }
        )
        return config
    
    @classmethod
    def from_config(cls, config):
        config["graph_info"][1] = np.array(config["graph_info"][1])

        return cls(**config)

    def build(self, input_shape):
        # Call build() on layers that need to know their input shapes
        super(GNNNodeClassifier, self).build(input_shape)


