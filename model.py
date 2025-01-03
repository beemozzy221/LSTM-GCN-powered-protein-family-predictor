import keras
import numpy as np
import tensorflow as tf
from keras.src import layers
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
        graph_info,
        aggregation_type="sum",
        combination_type="concat",
        dropout_rate=0.2,
        normalize=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        #Unpack to give to get_config
        self.hidden_units = hidden_units
        self.graph_info = graph_info
        self.num_classes = num_classes
        self.lstm_hidden_units = lstm_hidden_units
        self.lstm_weights = lstm_weights

        #Unpack graph_info to three elements: node_features, edges, and edge_weight.
        node_features, edges, edge_weights = graph_info
        self.node_features = node_features
        self.edges = edges
        self.edge_weights = edge_weights

        # Set edge_weights to ones if not provided.
        if self.edge_weights is None:
            self.edge_weights = tf.ones(shape=edges.shape[1])
        # Scale edge_weights to sum to 1.
        self.edge_weights = self.edge_weights / tf.math.reduce_sum(self.edge_weights)
        #Create preprocess layer
        self.preprocess = create_ffn(hidden_units, dropout_rate, name = "preprocess")

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
        self.preprocess_lstm_layer.build(input_shape=(None, *node_features.shape[1:]))
        self.preprocess_lstm_layer.load_weights(lstm_weights)

    def call(self, input_node_indices):

        #LSTM preprocessing layer
        x1 = self.preprocess_lstm_layer(self.node_features)

        # Apply the first graph conv layer
        x2 = self.conv1((x1, self.edges, self.edge_weights))

        # Apply the second graph conv layer
        x3 = self.conv2((x2, self.edges, self.edge_weights))

        # Postprocess node embedding
        x4 = self.postprocess(x3)
        
        # Fetch node embeddings for the input node_indices
        node_embeddings = tf.gather(x4, input_node_indices)

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
                "lstm_weights": self.lstm_weights
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


