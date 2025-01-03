import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
import matplotlib.pyplot as plt
from model import GNNNodeClassifier
from graphinfo import GraphInfo
from sklearn.metrics import roc_curve, auc
from traintestvalidatesplit import TrainTestVaiCreate

#Parameters for training
hidden_units = [64, 64]
lstm_hidden_units = [64, 64]
learning_rate = 0.01
dropout_rate = 0.5
num_epochs = 2
batch_size = 30
test_size = 0.3
num_classes = 106

#SRC files
protein_features = np.load("auxillaryfiles/protein_encoded_sequence.npy")
protein_family_targets = np.load("auxillaryfiles/protein_family_encoded_labels.npy")
gnn_saved_weights = r"savedweights/gnnweights.weights.h5"
lstm_saved_weights = r"savedweights/lstmweights.weights.h5"

#Create train,test,validate object
train_test_split = TrainTestVaiCreate(protein_features, protein_family_targets)
xml_data, yml_data = train_test_split.split(test_size)
x_train, x_test, train_indices, test_indices = xml_data
y_train, y_test, _, _ = yml_data

#Create a graph info list
adjacency_matrix = np.load(f"auxillaryfiles/adjmatrix.npy")
node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
graph = GraphInfo(
    edges=(node_indices.tolist(), neighbor_indices.tolist()),
    num_nodes=adjacency_matrix.shape[0],
)
print(f"number of nodes: {graph.num_nodes}, number of edges: {len(graph.edges[0])}")

#Graph info
graph_info = (protein_features, np.array(graph.edges), None)

#Start defining the model parameters
early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=50, restore_best_weights=True
)

model = GNNNodeClassifier(num_classes=num_classes, lstm_hidden_units=lstm_hidden_units, hidden_units=hidden_units,
                          graph_info=graph_info, dropout_rate=dropout_rate, lstm_weights=lstm_saved_weights, name="gnn_model")

model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.CategoricalAccuracy(name="acc")]
)

#Train the model
model.fit(
        shuffle=True,
        x=train_indices,
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(test_indices, y_test),
        callbacks=[early_stopping],    
)

#Save the model
#model.save_weights(gnn_saved_weights)

#Plotting
y_pred_prob = model.predict(train_indices)
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_train[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_train.ravel(), y_pred_prob.ravel())
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
plt.show()












