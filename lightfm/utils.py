import typing as t

import torch as pt

__all__ = [
    "SparseTensorDataset",
    "SparseCOOTensorT",
    "SparseCSRTensorT",
    "sparse_eye",
    "sparse_identity",
]

SparseCOOTensorT: "t.TypeAlias" = pt.Tensor  # pt.layout = pt.sparse_coo
SparseCSRTensorT: "t.TypeAlias" = pt.Tensor  # pt.layout = pt.sparse_csr


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


class SparseTensorDataset(pt.utils.data.Dataset):
    def __init__(self, inputs: pt.Tensor, targets: t.Optional[pt.Tensor] = None):
        if targets is not None:
            assert inputs.values().numel() == targets.numel(), \
                'Inputs and targets must align in size.'
        self._data = inputs.to_sparse_coo()
        self._indices: pt.Tensor = self._data.indices()
        self._values: pt.Tensor = self._data.values()
        self._targets = targets

    def __len__(self) -> int:
        return self._values.shape[0]

    def __getitem__(self, idx: int) -> t.Tuple:
        # Let out of bounds error be implicit when idx >= len(self).
        row, col = self._indices[:, idx].view(-1)
        value = self._values[idx]
        if self._targets is None:
            return row, col, value
        target = self._targets[idx]
        return row, col, value, target
