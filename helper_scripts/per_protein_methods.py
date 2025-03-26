import ripserplusplus as rpp
import numpy as np

import multiprocessing

from scipy.sparse.csgraph import minimum_spanning_tree

import warnings
warnings.filterwarnings("ignore")


def create_bipartite_graph(matrix):
    n = matrix.shape[0]
    bipartite_adj = np.zeros((2 * n, 2 * n)) 

    for i in range(n):
        for j in range(n):
            bipartite_adj[i, j + n] = matrix[i, j]
            bipartite_adj[i + n, j] = matrix[i, j]

    return bipartite_adj
    
def get_per_protein_features_bipartite_h0(matrices):
    batch_size, n_nodes, _ = matrices.shape
    barcodes_list = []

    for b in range(batch_size):
        dist_matrix = 1 - np.maximum(matrices[b], matrices[b].T)
        bipartite_adj = create_bipartite_graph(dist_matrix)
        mst = np.array(minimum_spanning_tree(bipartite_adj).toarray())
        rows, cols = np.where(mst > 0)

        weights = mst[rows, cols]

        barcodes = [[0, w, 0] for w in sorted(weights)]
        barcodes_list.append(barcodes)
    return barcodes_list


def get_per_protein_features_from_attention_matrices_h1(matrices):
    batch_size, n_nodes, _ = matrices.shape
    barcodes_list = []

    for b in range(batch_size):
        dist_matrix = 1 - np.maximum(matrices[b], matrices[b].T)
        np.fill_diagonal(dist_matrix, 0)

        diagrams = rpp.run(f"--format distance --dim {1}", data=dist_matrix)
        H0_barcodes = diagrams[0]
        H1_barcodes = diagrams[1]

        barcodes_list.append([[x[0], x[1], 0] for x in H0_barcodes.tolist()] + [[x[0], x[1], 1] for x in H1_barcodes.tolist()])

    return barcodes_list


def get_per_protein_features_from_attention_matrices_nonsymmetric(matrices):
    batch_size, n_nodes, _ = matrices.shape
    barcodes_list = []

    for b in range(batch_size):
        dist_matrix = 1 - matrices[b]  
        np.fill_diagonal(dist_matrix, 0)

        diagrams = rpp.run(f"--format distance --dim {1}", data=dist_matrix)

        H0_barcodes = diagrams[0]
        H1_barcodes = diagrams[1]

        barcodes_list.append([[x[0], x[1], 0] for x in H0_barcodes.tolist()] + [[x[0], x[1], 1] for x in H1_barcodes.tolist()])

    return barcodes_list

def get_barcodes_batch(matrices, func):
    with multiprocessing.Pool() as pool:
        barcodes = pool.map(func, matrices)
        return barcodes