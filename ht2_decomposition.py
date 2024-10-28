import torch
import tensorly as tl
from tensorly.decomposition import partial_tucker


def left_svd_qr(X, rank):
    _, R = tl.qr(X.T)
    R = R[0 : R.shape[1], :]
    U, S, _ = torch.linalg.svd(R.T, False)
    U = U[:, 0:rank]
    S = S[0:rank]
    return U, S


def get_singular_values(X):
    _, R = tl.qr(X.T)
    R = R[0 : R.shape[1], :]
    _, S, _ = torch.linalg.svd(R.T, False)
    return S


def ht2_decomposition(X, ranks):
    X_ = X
    R1, R3, R13 = ranks
    dims = X.shape

    unfolded_0 = tl.unfold(X, 0)
    U_1, _ = left_svd_qr(unfolded_0, R1)
    X_ = torch.tensordot(X_, U_1, dims=([0], [0]))

    unfolded_2 = tl.unfold(X, 2)
    U_3, _ = left_svd_qr(unfolded_2, R3)
    X_ = torch.tensordot(X_, U_3, dims=([1], [0]))
    X_ = X_.permute([2, 0, 3, 1])

    unfolded_0_1 = X_.reshape(R1 * dims[1], -1)
    G, _ = left_svd_qr(unfolded_0_1, R13)
    G_1 = G.reshape(R1, dims[1], R13)
    X_ = G.T @ unfolded_0_1
    G_2 = X_.reshape(R13, R3, dims[3])

    rec = torch.tensordot(G_1, G_2, dims=([2], [0]))
    rec = torch.tensordot(rec, U_1, dims=([0], [1]))
    rec = torch.tensordot(rec, U_3, dims=([1], [1]))
    rec = rec.permute([2, 0, -1, 1])

    print(f"Res error: {tl.norm(rec - X, 2)/tl.norm(X, 2)}")

    return U_1, U_3, G_1, G_2
