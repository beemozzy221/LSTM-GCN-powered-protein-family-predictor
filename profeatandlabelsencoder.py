import numpy as np
from Bio import SeqIO
from prelimlabelsencoder import LabelsEncode

inputs = 255
max_len_of_seq = 2033
amino_char = 26
input_file = r"proteinfastaseq/alignedproseq.fasta"
protein_panther_ids_file = r"proteinfastaseq/uniaccwpan.txt"


def seq2onehot(seq):
    """Create 26-dim embedding"""
    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
             'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_size = len(chars)
    vocab_embed = dict(zip(chars, range(vocab_size)))

    # Convert vocab to one-hot
    vocab_one_hot = np.zeros((vocab_size, vocab_size), int)
    for _, val in vocab_embed.items():
        vocab_one_hot[val, val] = 1

    embed_x = [vocab_embed[v] for v in seq]
    seqs_x = np.array([vocab_one_hot[j, :] for j in embed_x])

    return seqs_x

def ready_encoding(no_seq, protein_length, uniandpanid, aminochars):

    #Prepare the arrays
    unencoded_features_array = np.full(protein_length, "-", dtype=object)
    protein_features = np.full((no_seq, protein_length, aminochars), "-", dtype=object)
    protein_labels = np.zeros(shape=no_seq, dtype=object)
    i = 0

    #Parse the FASTA files
    fasta_sequences = SeqIO.parse(open(input_file), 'fasta')
    with open(uniandpanid) as uni_pan_ids:
        for fasta in fasta_sequences:
            name, sequence = fasta.id, str(fasta.seq)
            uniprot_index = name.find("|")
            uniprot_index += 1
            protein_id_chars = []

            #Seek until "|", writes the protein ID
            while name[uniprot_index] != "|":
                protein_id_chars.append(name[uniprot_index])
                uniprot_index+=1

            #Convert to string
            uniprot_id = "".join(protein_id_chars)
  
            for ids in uni_pan_ids:

                #Seek until Uniprot ID
                if ids.rfind(uniprot_id) != -1:
                    index = ids.rindex("|")
                    index += 1
                    panther_id_chars = []

                    #Read until the last character
                    while ids[index]> " ":
                        panther_id_chars.append(ids[index])
                        index += 1

                    #Convert the Panther ID to string and write to the protein list
                    panther_id = ''.join(panther_id_chars)
                    protein_labels[i] = panther_id

                    j = 0
                    for protein_chars in sequence:
                        unencoded_features_array[j] = protein_chars
                        j += 1
                    protein_features[i] = seq2onehot(unencoded_features_array)
                    i += 1

                    #Seek to the start of the file
                    uni_pan_ids.seek(0)
                    break

    return protein_features, protein_labels

if __name__ == "__main__":
    features, unencoded_labels = ready_encoding(inputs, max_len_of_seq, protein_panther_ids_file, amino_char)
    encoder = LabelsEncode(unencoded_labels)
    encoded_labels = encoder.labels2onehot()

    #Save the files
    np.save('auxillaryfiles/protein_encoded_sequence.npy', features)
    np.save('auxillaryfiles/protein_family_encoded_labels.npy', encoded_labels)
                
            
