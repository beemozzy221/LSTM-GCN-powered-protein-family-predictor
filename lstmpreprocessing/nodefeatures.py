import numpy as np
#Input sequences to be put into the processed LSTM model, as numpy inputs
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(0,len(data) - sequence_length+1,sequence_length):
        sequence = data[i:i + sequence_length]
        sequences.append(sequence)
    return np.array(sequences)


input_sequence_length = 12

x_array=np.load("../auxillaryfiles/protein_encoded_sequence.npy")

flatten_xarray = x_array.reshape(x_array.shape[0]*x_array.shape[1],x_array.shape[2])

input_sequences = create_sequences(flatten_xarray, input_sequence_length)

print(input_sequences.shape)

np.save("../auxillaryfiles/nodefeatures.npy",input_sequences)
