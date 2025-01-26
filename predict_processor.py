import os

import numpy as np
import adjscoregenerator
from profeatandlabelsencoder import seq2onehot

class PredictProcess:

    def __init__(self, predict_protein_features, concatenate_protein_array, adjacency_matrix,
                 predict_protein_seq_filepath, source_protein_seq_filepath, protein_family_folder):
        self.protein_features = predict_protein_features
        self.concatenate_protein_array = concatenate_protein_array
        self.predict_filepath = predict_protein_seq_filepath
        self.source_filepath = source_protein_seq_filepath
        self.adjacency_matrix = adjacency_matrix
        self.protein_fam_folder = protein_family_folder

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
        predict_adj_matrix = np.pad(self.adjacency_matrix, ((0, 1), (0, 1)), mode='constant', constant_values=0)

        predict_row = [adjscoregenerator.create_score_matrix_unaligned(self.predict_filepath, f"{self.protein_fam_folder}"
                                                                                              f"/{protein_fam}/unalignedproseq.fasta")
                       for protein_fam in os.listdir(self.protein_fam_folder)]
        predict_row = [x for x in predict_row if x is not None]
        homogenized_predict_row = [ent for pro_fam in predict_row for ent in pro_fam]
        homogenized_predict_row = np.append(homogenized_predict_row, np.array([0]))

        w_mask = np.ones(self.adjacency_matrix.shape[0]+1)
        predict_row = (homogenized_predict_row >= 800) * w_mask

        predict_adj_matrix[-1, :] = predict_row
        predict_adj_matrix[:, -1] = predict_row.T

        return predict_adj_matrix

















