import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
from graphinfo import GraphInfo
from model import GNNNodeClassifier

print(tf.version.VERSION)

hidden_units = [64, 64]
lstm_hidden_units = [64, 64]
dropout_rate = 0.5
batch_size = 200
test_size = 0.7
num_classes = 106

#Load all files
gnn_weights = r"savedweights/gnnweights.weights.h5"
lstm_weights = r"savedweights/lstmweights.weights.h5"
predict_folder_path = r"predict/predict.npy"
encoded_predict_features = np.load(predict_folder_path)
protein_features = np.load("auxillaryfiles/protein_encoded_sequence.npy")

#Construct all the necessary graph items
adjacency_matrix = np.load(f"auxillaryfiles/adjmatrix.npy")
node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
graph = GraphInfo(
    edges=(node_indices.tolist(), neighbor_indices.tolist()),
    num_nodes=adjacency_matrix.shape[0],
)
graph_info = (protein_features, np.array(graph.edges), None)

#Construct the model
model = GNNNodeClassifier(num_classes=num_classes, hidden_units=hidden_units, graph_info=graph_info,
                          dropout_rate=dropout_rate, lstm_weights=lstm_weights,
                          lstm_hidden_units=lstm_hidden_units,
                          name="gnn_model")
model.build(input_shape=(None, ))
model.load_weights(gnn_weights)
model.summary()

#Predict





  





