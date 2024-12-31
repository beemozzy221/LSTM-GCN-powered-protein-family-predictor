# LSTM-GCN-powered-protein-family-predictor

The model performs a protein family prediction (about to be) trained on DNA metabolism proteins in the protein class of the Panther database (pantherdb.org). The DNA metabolism protein class of the panther database has around 21000 genes with 160 protein families (sub-families not included lest having lack of representation for each class). However, for demonstration purposes, the current model only train on 255 of those said sequences targeting 106 of the said labels.

The data preprocessing is the natural first step as the sequences have been aligned with a aligned length of 1857 amino acids, each with amino acid represented with a 26 dimension vector. The preprocessing step reduces the axes of the dataset to just 2 (the batch and the processed information for each element in the batch). Since LSTM layers require dataset in a 3D vector form of the structure (batch size, number_of_time_steps, features) (with the relevant targets being (batch size, targets). The dataset was transformed into the preferred structure using create_timeseries function which outputs a tensor with (inputs, targets) for the LSTM layer to train on.

The processed data are now the node features for each element in the batch. The node features are then used to train the GCN layers. Since GCN layers require a adjacency matrix (ADJM) for node edge informations, a seperate python file was used to create the ADJM of dimenstions (num_nodes, num_nodes) with a "1" for sequences with similarity score > 1000 and "0" otherwise. The processed dataset are then reshaped and is ready to be inputted into the GCN layer along with the ADJM for training.

The processed GCN outputs are then given to Dense layers for final processing and are used then for the final prediction. Loss functions and optimizers are changeable to train the model and make predictions. 

Here each protein sequence represent a node and the edges form between the families (but could change with the choice of the similarity score threshold).

Note that this model is not ready to accept inputs and make predictions. While it could be done quickly, the model training part seems to be of first priority for now.

Note: Testing has been done for training and validation only (metric: accuracy).

IMPORTANT: NOTICE THAT THE MODEL DOES NOT HAVE THE ENCODED NODE FEATURES THAT ARE TO BE GIVEN TO THE LSTM MODEL due to its large size. Instead, the main file can run on processed node features (lstm_padded_array) that are already processed and uploaded. So the main file (mainfileanddataprep) should work as expected.

!!!!!MODEL IS NOT OPTIMIZED!!!!! (epochs, batch size, choice of optimizers, loss etc could affect the model performance).

EDIT 1: This is a model built by a 3rd year undergraduate of University of Colombo, Faculty of Science doing Bioinformatics. Expect the coding to be shabby and substandard to the conventions followed by the wider coding community.

EDIT 2: Graphconvo.py and model.py codes are taken from keras documentation with minor modificaitons.

EDIT 3: Subset of protein sequences are aligned using MUSCLE.

MAJOR UPDATE TO CODES: 31/12/2024:

The codes have been changed to match with the more rigorous style of coding. Large amounts of edits have been done to make the codes more robust. The general model architecture has also been changed to simplify the training process and experiment with different architectures in the future. PREDICT FUNCTIONALITY IS STILL UNDER CONSTRUCTION.

Auxillary files doesn't have the encoded protein features. It could be found in this drive link: https://drive.google.com/file/d/1GozAiYDZjVksLqAIYs_nLGq2ka9Nbpjy/view



