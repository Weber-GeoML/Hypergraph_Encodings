import numpy as np
from scipy.sparse import csr_matrix


class DisperseUnweightedClique:
    pass


class DisperseWeightedClique:
    pass


class DisperseUnweightedStar:
    pass


def disperse(cls, node, alpha, neighbors, rc=None, cr=None, rw=None, *_):
    total = len(neighbors)
    N = neighbors[node]

    if not N or len(N) == 1:
        return csr_matrix(([1.0], ([node], [0])), shape=(total, 1))

    if cls == DisperseUnweightedClique:
        x = np.zeros(total)
        x[N] = (1 - alpha) / (len(N) - 1)
        x[node] = alpha
        return csr_matrix(x)

    if cls == DisperseWeightedClique:
        W = rw[node]
        factor = (1 - alpha) / (np.sum(W) - W[np.where(N == node)[0][0]])
        x = np.zeros(total)
        x[N] = W * factor
        x[node] = alpha
        return csr_matrix(x)

    if cls == DisperseUnweightedStar:
        dispersion = np.zeros(total)
        k = 0
        for e in rc[node]:
            if len(cr[e]) > 1:
                k += 1
                for x in cr[e]:
                    dispersion[x] += (1 - alpha) / (len(cr[e]) - 1)
        if k > 0:
            dispersion /= k
        dispersion[node] = alpha
        return csr_matrix(dispersion)


def prepare_cost_matrix(cls, neighbors):
    K = len(neighbors)
    C = np.full((K, K), 3, dtype=np.int8)

    for m in range(K):
        N = neighbors[m]
        for n in N:
            C[n, m] = C[m, n] = 1

        for i in range(len(N)):
            for j in range(i, len(N)):
                s, t = N[i], N[j]
                if s != t and s != m and C[s, t] == 3:
                    C[t, s] = C[s, t] = 2

        C[m, m] = 0

    return C


def prepare_weights(rc, cr, neighbors):
    return [np.sum([x in e for e in cr[n]]) for n in neighbors]


def any_bits(f, s, t):
    a1, b1 = s.bits, s.offset
    a2, b2 = t.bits, t.offset
    l1, l2 = len(a1), len(a2)
    bdiff = b2 - b1

    for i in range(max(1, 1 + bdiff), min(l1, l2 + bdiff)):
        if f(a1[i], a2[i - bdiff]):
            return True

    return False


def intersects(u, v):
    return any_bits(lambda a, b: (a & b) != 0, u, v)


def truncated_cost(m, n, neighbors):
    if n == m:
        return 0
    if n in neighbors[m]:
        return 1
    if intersects(neighbors[n], neighbors[m]):
        return 2
    return 3


def wasserstein(u, v, C, dispersions):
    U, X = dispersions[u].nonzero()
    V, Y = dispersions[v].nonzero()
    C = C[U[:, None], V]
    return OptimalTransport.sinkhorn2(X, Y, C, 1e-1, maxiter=500, atol=1e-2)


def aggregate(cls, S, W):
    s, n = 0.0, len(S)
    if n <= 1:
        return 0.0
    for i in range(n):
        for j in range(i + 1, n):
            s += W[S[i], S[j]]
    return s * 2 / (n * (n - 1)) if cls == AggregateMean else np.max(s)


def node_curvature_neighborhood(i, W, neighbors):
    """ Compute the ORC node curvature using the edges"""
    N = neighbors[i]
    if len(N) <= 1:
        return 1.0
    return np.sum([1.0 - W[i, j] for j in N if j != i]) / (len(N) - 1)


def node_curvature_edges(node, dist, rc):
    """ Compute the ORC node curvature using the edges"""
    degree = len(rc[node])
    if degree == 0:
        return 1.0
    return np.sum([dist[edge] for edge in rc[node]]) / degree


def hypergraph_curvatures(dispersion, aggregations, rc, cr, alpha, cost):
    neighbors = [set(edges) for edges in rc]
    C = prepare_cost_matrix(cost, neighbors)

    if dispersion == DisperseWeightedClique:
        rw = prepare_weights(rc, cr, neighbors)
    else:
        rw = None

    D = [disperse(dispersion, n, alpha, neighbors, rc, cr, rw) for n in range(len(rc))]

    w = np.zeros((len(rc), len(rc)), dtype=np.float32)
    for i in range(len(rc)):
        for j in range(i + 1, len(rc)):
            w[j, i] = wasserstein(i, j, C, D)

    nc = [node_curvature_neighborhood(n, w, neighbors) for n in range(len(rc))]

    ac = [
        {
            "aggregation": aggregation,
            "edge_curvature": [
                1 - aggregate(aggregation, cr[e], w) for e in range(len(cr))
            ],
            "node_curvature_edges": [
                node_curvature_edges(n, ec, rc) for n in range(len(rc))
            ],
        }
        for aggregation in aggregations
    ]

    return {
        "dispersions": D,
        "directional_curvature": 1.0 - w,
        "node_curvature_neighborhood": nc,
        "aggregations": ac,
    }


def hypergraph_curvatures_dispatch(dispersion, aggregation, incidence, alpha, cost):
    if isinstance(incidence, csr_matrix):
        rc = [
            incidence.indices[incidence.indptr[i] : incidence.indptr[i + 1]].tolist()
            for i in range(incidence.shape[0])
        ]
        cr = [
            incidence.indptr[incidence.indices == i].tolist()
            for i in range(incidence.shape[1])
        ]
    else:
        rc = incidence
        cr = transpose_edgelist(incidence)

    if isinstance(aggregation, tuple):
        aggregations = list(aggregation)
    else:
        aggregations = [aggregation]

    return hypergraph_curvatures(dispersion, aggregations, rc, cr, alpha, cost)


def edgelist_format(I, J, n) -> list:
    x: list = [[] for _ in range(n)]
    for i, j in zip(I, J):
        x[i].append(j)
    return x


def transpose_edgelist(cr) -> list:
    rc: list = [[] for _ in range(max(len(e) if e else 0 for e in cr))]
    for j, e in enumerate(cr):
        for i in e:
            rc[i].append(j)
    return list(filter(None, rc))
