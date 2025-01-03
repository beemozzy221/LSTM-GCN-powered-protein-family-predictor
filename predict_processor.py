import numpy as np
import adjscoregenerator
from profeatandlabelsencoder import seq2onehot

class PredictProcess:

    def __init__(self, predict_protein_features, concatenate_protein_array, source_score_matrix,
                 predict_protein_seq_filepath, source_protein_seq_filepath):
        self.protein_features = predict_protein_features
        self.concatenate_protein_array = concatenate_protein_array
        self.source_score_matrix = source_score_matrix
        self.predict_filepath = predict_protein_seq_filepath
        self.source_filepath = source_protein_seq_filepath

        print("Processor initialized")

    def feature_encoder(self):
        return np.array(seq2onehot(self.protein_features))

    def append_predict_protein(self):
        print(f"Now processing for protein features of length {len(self.protein_features)}")
        encoded_protein_array = PredictProcess.feature_encoder(self)

        try:
            assert encoded_protein_array.shape[0] <= self.concatenate_protein_array.shape[1]

            if encoded_protein_array.shape[0] == self.concatenate_protein_array.shape[1]:
                return np.concatenate((self.concatenate_protein_array,
                                       encoded_protein_array.reshape(1, *encoded_protein_array.shape)), axis=0)

            padding_length = self.concatenate_protein_array.shape[1] - encoded_protein_array.shape[0]
            padding_array = np.zeros(shape=(padding_length, encoded_protein_array.shape[1]))
            padded_features = np.concatenate((encoded_protein_array, padding_array), axis=0)
            concatenated_protein_array = np.concatenate((self.concatenate_protein_array,
                                                         padded_features.reshape(1, *padded_features.shape)), axis=0)

            return concatenated_protein_array

        except AssertionError:
            print(f"Predict protein length exceeds trained proteins' length of {self.concatenate_protein_array.shape[1]}")

    def adj_matrix (self):
        predict_score_matrix = np.pad(self.source_score_matrix, ((0, 1), (0, 1)), mode='constant', constant_values=0)
        predict_row = adjscoregenerator.create_score_matrix_unaligned(self.predict_filepath, self.source_filepath)
        predict_row = np.append(predict_row, np.array([0]), axis=0)
        predict_score_matrix[-1, :] = predict_row
        predict_score_matrix[:, -1] = predict_row.T

        return adjscoregenerator.compute_adjacency_matrix(predict_score_matrix, 1400)












