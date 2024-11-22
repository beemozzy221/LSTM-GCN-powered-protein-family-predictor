import os
os.environ["KERAS_BACKEND"] = "tensorflow"


from traintestvalidatesplit import ttvcreate
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras import ops
import pandas as pd
from model import GNNNodeClassifier
from graphconvo import GraphConvLayer
from graphinfo import GraphInfo
from sklearn.utils import shuffle

#Parameters for training

hidden_units = [64,64]
learning_rate = 0.001
dropout_rate = 0.5
num_epochs = 100
batch_size = 200
test_size=0.7
num_classes=106

#For creating timeseries dataset
batch_size_dataset = 1857
input_sequence_length = 12
forecast_horizon = 3



x_array=np.load("auxillaryfiles/protein_encoded_sequence.npy")
y_array=np.load("auxillaryfiles/protein_family_encoded_labels.npy")
lstm_padded_array = np.load("auxillaryfiles/lstm_padded_array.npy")

#Create train,test,validate object

ttvobj=ttvcreate(x_array,y_array)
xmldata, ymldata = ttvobj.preprocess(test_size)

x_train, x_test, train_indices, test_indices = xmldata

y_train, y_test, train_indices, test_indices = ymldata

#Create a graph info list

adjacency_matrix = np.load(f"auxillaryfiles/adjmatrix.npy")
node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
graph = GraphInfo(
    edges=(node_indices.tolist(), neighbor_indices.tolist()),
    num_nodes=adjacency_matrix.shape[0],
)
print(f"number of nodes: {graph.num_nodes}, number of edges: {len(graph.edges[0])}")

#graph_info = (x_array,np.asarray(graph.edges),None)

graph_info = (lstm_padded_array,np.asarray(graph.edges),None)


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
    targets=data_array[target_offset:],  
    sequence_length=input_sequence_length,
    batch_size=batch_size,
    sequence_stride = 1,
    shuffle=False)

    return dataset

train_dataset = (
    create_tf_dataset(flatten_xarray, input_sequence_length, forecast_horizon, batch_size_dataset))





early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=50, restore_best_weights=True
)


model = GNNNodeClassifier(
    graph_info=graph_info,
    num_classes=num_classes,
    hidden_units=hidden_units,
    dropout_rate=dropout_rate,
    name="gnn_model",
)


model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
)

#Train the model

model.fit(
        shuffle = True,
        x=train_indices,
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early_stopping],    
)


#model.save("model1.h5")




















