import os
import time

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from callbackestopping import CusEarlyStopping
from model import GNNNodeClassifier
from graphinfo import GraphInfo
from sklearn.metrics import roc_curve, auc

def sample_subgraph(node_features, adj_matrix, node_labels, batch_splits):
    if node_features.shape[0] % batch_splits != 0:
        split = node_features.shape[0] // (batch_splits - 1)
        batches = []

        #Create batches_seq
        for multiplication in range(batch_splits - 1):
            batches.append([node_features[split * multiplication:split * (multiplication+1)],
            adj_matrix[split*multiplication:split * (multiplication + 1), :][:, split*multiplication:split * (multiplication + 1)],
            node_labels[split*multiplication:split * (multiplication + 1)]])

        rem = node_features.shape[0] - (node_features.shape[0] % batch_splits)
        batches.append([node_features[rem:], adj_matrix[rem:, :][:, rem:],
        node_labels[rem:]])

    else:
        split = node_features.shape[0] // batch_splits
        batches = []

        # Create batches_seq
        for multiplication in range(batch_splits):
            batches.append([node_features[split * multiplication:split * (multiplication + 1)],
                            adj_matrix[split * multiplication:split * (multiplication + 1), :][:,
                            split * multiplication:split * (multiplication + 1)],
                            node_labels[split * multiplication:split * (multiplication + 1)]])

    return batches, split

def softmax(logits_model):
    exp_logits = np.exp(logits_model - np.max(logits_model))  # Numerical stability
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

def roc_calculation(labels_, targets_, num_classes_, name):
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes_):
        fpr[i], tpr[i], _ = roc_curve(labels_[:, i], targets_[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(labels_.ravel(), targets_.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    sorted_classes = sorted(roc_auc.items(), key=lambda x: x[1], reverse=True)
    top_n = 5  # Number of top classes to plot
    top_classes = [k for k, _ in sorted_classes[:top_n]]

    # Plot only the top N classes
    plt.figure(figsize=(10, 8))
    for i in top_classes:
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    # Add micro-average
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(f"AUCROC - {name}")
    plt.show()

#Parameters for training
hidden_units = [64, 64]
lstm_hidden_units = [64, 64]
learning_rate = 0.01
dropout_rate = 0.5
num_epochs = 11
batch_size = 10

#SRC files
protein_features = np.load("auxillaryfiles/protein_encoded_sequence.npy", allow_pickle=True)
protein_family_targets = np.load("auxillaryfiles/protein_family_encoded_labels.npy")
gnn_saved_weights = r"savedweights/gnnweights.weights.h5"
lstm_saved_weights = r"savedweights/lstmweights.weights.h5"

protein_features = protein_features[:600]
protein_family_targets = protein_family_targets[:600]
protein_family_targets = protein_family_targets[:, :11]

num_classes = len(np.unique(protein_family_targets, axis=0))

#Create a graph info list
adjacency_matrix = np.load(f"auxillaryfiles/newadjmatrix.npy")
node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
graph = GraphInfo(
    edges=(node_indices.tolist(), neighbor_indices.tolist()),
    num_nodes=adjacency_matrix.shape[0],
)
print(f"number of nodes: {graph.num_nodes}, number of edges: {len(graph.edges[0])}")

#Graph info
graph_info = (protein_features, np.array(graph.edges), None)

#Start defining the model parameters
early_stopping = CusEarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
)

model = GNNNodeClassifier(num_classes=num_classes, lstm_hidden_units=lstm_hidden_units,
                          hidden_units=hidden_units,
                          protein_length=2287,
                          protein_chars=26,
                          dropout_rate=dropout_rate,
                          lstm_weights=lstm_saved_weights,
                          name="gnn_model")


#Train the model
batches_seq, size = sample_subgraph(protein_features, adjacency_matrix,
                                    protein_family_targets, batch_size)

#Plotting
acc_history = []
loss_history = []
final_label = []
final_epoch = []
final_epoch_labels = []

#Model fit parameters
loss_func = keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate)
callback_list_ = [early_stopping]
for cb_ in callback_list_:
    cb_.set_model(model)

batches_seq_perm = len(batches_seq) - 1
for epoch in range(0, num_epochs):
    print(f"Epoch {epoch + 1}:")
    mean_loss = []
    mean_acc = []
    val_loss = 0
    val_acc = 0
    for batch_no, batch in enumerate(batches_seq):
        print(f"{batch_no + 1}/{len(batches_seq)}:")

        # Leave one out validation
        if (epoch < len(batches_seq)) & (batch_no == batches_seq_perm - epoch):
            print(f"Skipping batch {batch_no + 1}; designated as validation fold:")

            continue

        if (epoch >= len(batches_seq)) & (batch_no == batches_seq_perm - (epoch % len(batches_seq))):
            print(f"Skipping batch {batch_no + 1}; designated as validation fold:")

            continue

        step_start_time = time.time()

        feat, adj, labe = batch

        feat = np.array(feat, dtype="float32")
        adj = np.array(adj, dtype="float32")
        labe = np.array(labe, dtype="float32")

        sub_node_indices, sub_neighbor_indices = np.where(adj == 1)
        sub_graph = GraphInfo(
            edges=(sub_node_indices.tolist(), sub_neighbor_indices.tolist()),
            num_nodes=adj.shape[0],
        )
        # Graph info
        model.g_obj = [np.array(sub_graph.edges), None]

        with tf.GradientTape() as tape:
            logits = model(feat)
            loss = loss_func(labe, logits)

            if epoch + 1 == num_epochs:
                final_epoch.append(logits)
                final_epoch_labels.append(labe)

        # Compute Gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        # Apply Gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        #Compute mean loss
        mean_loss.append(loss)
        #Accuracy calculations
        acc_conversion = tf.argmax(softmax(logits), axis=-1, output_type=tf.int32)
        labe_conversion = tf.argmax(labe, axis=-1, output_type=tf.int32)
        acc_one_hot = tf.one_hot(acc_conversion, depth=num_classes)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(acc_conversion, labe_conversion), dtype=tf.float32))
        mean_acc.append(accuracy)

        end_time = time.time()  # End timer
        step_time = end_time - step_start_time

        # Print for each batch
        print(f"Loss: {loss} Accuracy: {accuracy} Step Time: {step_time}")

    print("Now testing the model on validation data")

    if epoch < len(batches_seq):
        print(f"Now validating for batch no: {(batches_seq_perm - epoch) + 1}")
        feat, adj, labe = batches_seq[batches_seq_perm - epoch]

        feat = np.array(feat, dtype="float32")
        adj = np.array(adj, dtype="float32")
        labe = np.array(labe, dtype="float32")

        sub_node_indices, sub_neighbor_indices = np.where(adj == 1)
        sub_graph = GraphInfo(
            edges=(sub_node_indices.tolist(), sub_neighbor_indices.tolist()),
            num_nodes=adj.shape[0],
        )
        # Graph info
        model.g_obj = [np.array(sub_graph.edges), None]

        with tf.GradientTape() as tape:
            val_logits = model(feat)
            val_loss = loss_func(labe, val_logits)

    if epoch >= len(batches_seq):
        print(f"Now validating for batch no: {(batches_seq_perm - (epoch % len(batches_seq))) + 1}")
        feat, adj, labe = batches_seq[batches_seq_perm - (epoch % len(batches_seq))]

        feat = np.array(feat, dtype="float32")
        adj = np.array(adj, dtype="float32")
        labe = np.array(labe, dtype="float32")

        sub_node_indices, sub_neighbor_indices = np.where(adj == 1)
        sub_graph = GraphInfo(
            edges=(sub_node_indices.tolist(), sub_neighbor_indices.tolist()),
            num_nodes=adj.shape[0],
        )
        # Graph info
        model.g_obj = [np.array(sub_graph.edges), None]

        with tf.GradientTape() as tape:
            val_logits = model(feat)
            val_loss = loss_func(labe, val_logits)

    normalized_logits = tf.nn.softmax(val_logits)
    predicted_classes = tf.argmax(normalized_logits, axis=-1)
    labe_classes = tf.argmax(labe, axis=-1)

    val_conversion = tf.one_hot(predicted_classes, depth=val_logits.shape[-1], dtype=tf.int32)
    labels = tf.cast(labe, dtype=tf.int32)
    val_acc = tf.reduce_mean(tf.cast(tf.equal(predicted_classes, labe_classes), dtype=tf.float32))

    final_label = [labels, val_conversion]

    mean_loss = sum(mean_loss) / len(mean_loss)
    mean_acc = sum(mean_acc) / len(mean_acc)
    print(f"Epoch loss: {mean_loss}")
    print(f"Epoch accuracy: {mean_acc}")
    print(f"Epoch validation loss: {val_loss} validation accuracy: {val_acc}")

    # Manually Trigger Callbacks
    try:
        logs = {"loss": mean_loss,
                "acc": mean_acc,
                "val_acc": val_acc,
                "val_loss": val_loss}
        for cb in callback_list_:
            cb.on_epoch_end(epoch, logs)
        loss_history.append(mean_loss)
    except StopIteration as e:
        print(str(e))
        break

#Save the model
#model.save_weights(gnn_saved_weights)

#Training loss
epochs = range(1, len(loss_history) + 1)

# Plotting Accuracy
plt.figure(figsize=(10, 8))

# Plotting Loss
plt.plot(epochs, loss_history, 'bo-', label='Training Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

#Plotting the AUC curves
predictions = np.array(final_label[1])
predictions = softmax(predictions)
prediction_labels = np.array(final_label[0])

final_epoch = np.concatenate(final_epoch, axis=0)
final_epoch = softmax(final_epoch)
final_epoch_labels = np.concatenate(final_epoch_labels, axis=0)

# Compute ROC curve and AUC for each class
roc_calculation(prediction_labels, predictions, num_classes, "Training")

# Compute ROC curve and AUC for each class for final epoch
roc_calculation(final_epoch_labels, final_epoch, num_classes, "Validation")

