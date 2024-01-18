import torch as pt

__all__ = [
    "sparse_eye",
    "sparse_identity",
]


def sparse_identity(n: int, layout: str = "csr", requires_grad: bool = True) -> pt.sparse.Tensor:
    """Construct a sparse identity matrix"""
    assert layout in ("csr", "csc"), "Matrix sparsity layout must be one of 'csr' or 'csc'."
    o = pt.ones(n)
    seq = pt.arange(0, n + 1, dtype=pt.int)
    if layout == 'csr':
        return pt.sparse_csr_tensor(seq, seq[:-1], o, (n, n), requires_grad=requires_grad)
    if layout == 'csc':
        return pt.sparse_csc_tensor(seq, seq[:-1], o, (n, n), requires_grad=requires_grad)


sparse_eye = sparse_identity
