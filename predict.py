import os


os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
from Bio import SeqIO
from graphinfo import GraphInfo
from model import GNNNodeClassifier
from matplotlib import pyplot as plt
from predict_processor import PredictProcess

print(tf.version.VERSION)

#Softmax conversion function
def softmax(logits_model):
    exp_logits = np.exp(logits_model - np.max(logits_model))  # Numerical stability
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

hidden_units = [64, 64]
lstm_hidden_units = [64, 64]
dropout_rate = 0.5
batch_size = 200
num_classes = 106

#Load all files
gnn_weights = r"savedweights/gnnweights.weights.h5"
lstm_weights = r"savedweights/lstmweights.weights.h5"
predict_folder_path = r"predict/predict.fasta"
source_protein_seq_filepath = r"proteinfastaseq/alignedproseq.fasta"
protein_family_folder = r"unaligned_protein_families"
concatenate_protein_features = np.load("auxillaryfiles/protein_encoded_sequence.npy", allow_pickle=True)
adjacency_matrix = np.load(r"auxillaryfiles/newadjmatrix.npy")

#Encode the to-be-predicted FASTA file
fasta_sequences = SeqIO.parse(open(predict_folder_path), 'fasta')
protein_features = [sequence_char for sequence in fasta_sequences for sequence_char in sequence.seq]
protein_features = [seq_char[0] for seq_char in protein_features]

#Encoding and appending to the trained protein matrix
pred_obj = PredictProcess(protein_features, concatenate_protein_features, adjacency_matrix,
                          predict_folder_path, source_protein_seq_filepath, protein_family_folder)
encoded_protein_features = pred_obj.feature_encoder()
concatenated_protein_features = pred_obj.append_predict_protein()
mod_adjacency_matrix = pred_obj.adj_matrix()

#Construct all the necessary graph items
node_indices, neighbor_indices = np.where(mod_adjacency_matrix == 1)
graph = GraphInfo(
    edges=(node_indices.tolist(), neighbor_indices.tolist()),
    num_nodes=mod_adjacency_matrix.shape[0],
)
graph_info = (concatenated_protein_features,
              np.array(graph.edges, dtype="int32"), None)
print(f"number of nodes: {graph.num_nodes}, number of edges: {len(graph.edges[0])}")

#Construct the model
model = GNNNodeClassifier(num_classes=num_classes, hidden_units=hidden_units, graph_info=graph_info,
                          dropout_rate=dropout_rate, lstm_weights=lstm_weights,
                          lstm_hidden_units=lstm_hidden_units,
                          name="gnn_model")
model.build(input_shape=(None, ))
model.load_weights(gnn_weights)
model.summary()

#Predict
predict_indices = np.array([255], dtype="int32")
logits = model.predict(predict_indices)

#Visualize the results
probabilities = softmax(logits)

# Indices of the top 5 probabilities
top_indices = np.argsort(probabilities[0])[::-1][:10]
top_probabilities = probabilities[0][top_indices]
top_classes = [str(class_) for class_ in top_indices]
print(f"Most probable class: {top_indices[0]}")

plt.figure(figsize=(8, 5))
plt.bar(top_classes, top_probabilities, color='skyblue', tick_label=top_classes)
plt.xlabel("Class")
plt.ylabel("Probability")
plt.title("Top 5 Probable Classes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()






  





