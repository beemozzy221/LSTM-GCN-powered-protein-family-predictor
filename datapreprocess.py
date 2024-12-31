import numpy as np
from Bio import SeqIO
from Bio import Align

alignedproseq=r"proteinfastaseq/alignedproseq.fasta"
no_of_ent=255
sigma=1400
              
def compute_adjacency_matrix(sequence_distances:np.ndarray, sigma:int):
    """Computes the adjacency matrix from distances matrix.

    It uses the formula in https://github.com/VeritasYin/STGCN_IJCAI-18#data-preprocessing to
    compute an adjacency matrix from the distance matrix.
    The implementation follows that paper.

    Args:
        sigma: numpy n - dimensional array
        sequence_distances: integer threshold
        route_distances: np.ndarray of shape `(num_routes, num_routes)`. Entry `i,j` of this array is the
            distance between roads `i,j`.
        sigma2: Determines the width of the Gaussian kernel applied to the square distances matrix.
        epsilon: A threshold specifying if there is an edge between two nodes. Specifically, `A[i,j]=1`
            if `w_mask >= sigma` and `A[i,j]=0` otherwise, where `A` is the adjacency
            matrix and `w2=route_distances * route_distances`

    Returns:
        A boolean graph adjacency matrix.
    """
    num_routes = sequence_distances.shape[0]
    w_mask =  np.ones([num_routes, num_routes]) - np.identity(num_routes)
    return (sequence_distances >= sigma) * w_mask

def createScoreMatrix(alignedproseq:str,no_of_ent:int):
    i,j = 0, 0

    #Start converting the FASTA files
    score_matrix=np.zeros(shape=(no_of_ent,no_of_ent),dtype=int)
    fasta_sequences = SeqIO.parse(open(alignedproseq),'fasta')
    aligner = Align.PairwiseAligner(match_score=1.0)

    for queryf in fasta_sequences:
        for targetf in SeqIO.parse(alignedproseq, "fasta"):
            query = str(queryf.seq)
            target = str(targetf.seq)
            score = aligner.score(target, query)
            score_matrix[i,j] = score
            j += 1
            if (j == no_of_ent):
                j = 0
                break
        i += 1

    return score_matrix

if __name__ == "__main__":
    #Generate and store the score matrix
    scorematrix= createScoreMatrix(alignedproseq, no_of_ent)
    #np.save('auxillaryfiles/scorematrix.npy', scorematrix)

    #Generate and store the adjacency matrix
    adjmatrix=compute_adjacency_matrix(np.load("auxillaryfiles/scorematrix.npy"),sigma)
    np.save('auxillaryfiles/adjmatrix.npy', adjmatrix)
