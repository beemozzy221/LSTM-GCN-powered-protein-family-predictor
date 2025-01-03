import numpy as np
from Bio import SeqIO
from Bio import Align
from Bio.pairwise2 import align

aligned_protein_seq_filepath = r"proteinfastaseq/aligned_protein_seq_filepath.fasta"
entries = 255

def compute_adjacency_matrix (sequence_distances:np.ndarray, similarity_threshold:int):
    """Computes the adjacency matrix from distances matrix.

    It uses the formula in https://github.com/VeritasYin/STGCN_IJCAI-18#data-preprocessing to
    compute an adjacency matrix from the distance matrix.
    The implementation follows that paper.

    Args:
        sequence_distances: np.ndarray of shape `(num_routes, num_routes)`. Entry `i,j` of this array is the
            distance between roads `i,j`.
        similarity_threshold: A threshold specifying if there is an edge between two nodes. Specifically, `A[i,j]=1`
            if `w_mask >= sigma` and `A[i,j]=0` otherwise, where `A` is the adjacency
            matrix and `w2=route_distances * route_distances`

    Returns:
        A boolean graph adjacency matrix.
    """
    num_routes = sequence_distances.shape[0]
    w_mask =  np.ones([num_routes, num_routes]) - np.identity(num_routes)
    return (sequence_distances >= similarity_threshold) * w_mask

def create_score_matrix (aligned_protein_seq:str, no_of_ent:int):
    i,j = 0, 0

    #Start converting the FASTA files
    score_matrix = np.zeros(shape=(no_of_ent,no_of_ent),dtype=int)
    fasta_sequences = SeqIO.parse(open(aligned_protein_seq), 'fasta')
    aligner = Align.PairwiseAligner(match_score=1.0)

    for queryf in fasta_sequences:
        for targetf in SeqIO.parse(aligned_protein_seq, "fasta"):
            query = str(queryf.seq)
            target = str(targetf.seq)
            score = aligner.score(target, query)
            score_matrix[i,j] = score
            j += 1
            if j == no_of_ent:
                j = 0
                break
        i += 1

    return score_matrix

def create_score_unaligned (target_sequence, comparison_sequence):
    match_score = 2
    mismatch_penalty = -1
    gap_open = -0.5
    gap_extend = -0.1

    alignments = align.globalms(target_sequence,
                                comparison_sequence, match_score, mismatch_penalty,
                                gap_open, gap_extend)

    return alignments[0][2]

def create_score_matrix_unaligned(target_fasta, source_fasta):

    target_sequences = list(SeqIO.parse(open(target_fasta), 'fasta'))
    source_sequences = list(SeqIO.parse(open(source_fasta), 'fasta'))
    assert len(target_sequences) == 1, "Only one sequence can be processed at a time!"

    return np.array([create_score_unaligned(target_seq.seq, source_seq.seq) for target_seq
                              in target_sequences for source_seq in source_sequences])

if __name__ == "__main__":
    #Generate and store the score matrix
    #score_matrix_ = create_score_matrix(aligned_protein_seq_filepath, entries)
    #np.save('auxillaryfiles/score_matrix_.npy', score_matrix_)

    #Generate and store the adjacency matrix
    #adj_matrix = compute_adjacency_matrix(np.load("auxillaryfiles/score_matrix_.npy"), sigma)
    #np.save('auxillaryfiles/adj_matrix.npy', adj_matrix)

    print("")

