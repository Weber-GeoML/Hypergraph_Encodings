"""Measure smoothing of a vector field on a graph."""

import numpy as np
from numba import jit


@jit(nopython=True)
def dirichlet_energy(X, edge_index):
    # computes Dirichlet energy of a vector field X with respect to a graph with a given edge index
    n = X.shape[0]
    m = len(edge_index[0])
    l = X.shape[1]
    degrees = np.zeros(n)
    for counter in range(m):
        u = edge_index[0][counter]
        degrees[u] += 1
    y = np.linalg.norm(X.flatten()) ** 2
    for counter in range(m):
        for i in range(l):
            u = edge_index[0][counter]
            v = edge_index[1][counter]
            y -= X[u][i] * X[v][i] / (degrees[u] * degrees[v]) ** 0.5
    return y


def dirichlet_normalized(X, edge_index):
    energy = dirichlet_energy(X, edge_index)
    norm_squared = sum(sum(X**2))
    return energy / norm_squared
