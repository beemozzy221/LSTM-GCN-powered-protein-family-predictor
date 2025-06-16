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

def entropy_results(probs):
    log_probs = tf.math.log(probs + 1e-12)
    entropy = -tf.reduce_sum(probs * log_probs, axis=1)

    num_class = tf.cast(tf.shape(probs)[1], tf.float32)
    max_entropy = tf.math.log(num_class)

    normalized_entropy = entropy / max_entropy
    return normalized_entropy

def append_prediction_to_txt(output_folder, output_text, seq_id, class_probs, entropy):
    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, output_text)

    with open(filepath, 'a') as f:
        f.write(f"Sequence ID: {seq_id}\n")
        f.write("Class Probabilities:\n")
        for i, prob in enumerate(class_probs):
            f.write(f"  Class {i}: {prob:.4f}\n")
        f.write(f"Entropy: {entropy:.4f}\n")
        f.write("-" * 40 + "\n")

def predict_seq(seq_loc, seq_id):
    hidden_units = [64, 64]
    lstm_hidden_units = [64, 64]
    dropout_rate = 0.5
    num_classes = 11
    predict_index = 7938
    protein_length = 2287
    protein_chars = 26

    #Load all files
    gnn_weights = r"savedweights/gnnweights.weights.h5"
    lstm_weights = r"savedweights/lstmweights.weights.h5"
    predict_folder_path = seq_loc
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
    concatenated_protein_features = pred_obj.append_predict_protein()
    mod_adjacency_matrix = pred_obj.adj_matrix()

#Isolating the neighbour nodes and features of the target node
predict_neighbour_indices = np.array(np.where(mod_adjacency_matrix[predict_index] == 1))[0]
neighbour_protein_features = concatenate_protein_features[predict_neighbour_indices]
predict_concatenated_protein_features = np.concatenate((neighbour_protein_features,
                                                       np.expand_dims(concatenated_protein_features[-1], axis=0)), axis=0)

    #adjacency matrix generation
    predict_neighbour_indices = np.array(np.where(mod_adjacency_matrix[predict_index] == 1))[0]
    predict_neighbour_indices = np.append(predict_neighbour_indices, np.array([predict_index]), axis=0)
    mod_new_adj = mod_adjacency_matrix[predict_neighbour_indices, :][:, predict_neighbour_indices]

    #Construct all the necessary graph items
    node_indices, neighbor_indices = np.where(mod_new_adj == 1)
    graph = GraphInfo(
        edges=(node_indices.tolist(), neighbor_indices.tolist()),
        num_nodes=mod_new_adj.shape[0],
    )
    graph_info = [np.array(graph.edges, dtype="int32"), None]
    print(f"number of nodes: {graph.num_nodes}, number of edges: {len(graph.edges[0])}")

    #Construct the model
    model = GNNNodeClassifier(num_classes=num_classes, hidden_units=hidden_units,
                              dropout_rate=dropout_rate, lstm_weights=lstm_weights,
                              lstm_hidden_units=lstm_hidden_units,
                              protein_length = protein_length,
                              protein_chars = protein_chars,
                              name="gnn_model")
    model.build(input_shape=(None, ))
    model.load_weights(gnn_weights)
    model.summary()

    #Predict
    predict_concatenated_protein_features = np.array(predict_concatenated_protein_features,
                                                     dtype="float32")
    model.g_obj = graph_info
    logits = model.predict(predict_concatenated_protein_features, batch_size=128)

    #Visualize the results
    probabilities = softmax(logits)

    # Indices of the top 5 probabilities
    top_indices = np.argsort(probabilities[-1])[::-1][:10]
    top_probabilities = probabilities[-1][top_indices]
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


for i, results in entropy_results(probabilities):
    if results > 0.5:
        print(f"Sample {i} results significant")
    else:
        print(f"Sample {i} results insignificant")






  





