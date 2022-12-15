import torch

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class SVDFeatureReduction(BaseTransform):
    r"""Dimensionality reduction of node features via Singular Value
    Decomposition (SVD).

    Args:
        out_channels (int): The dimensionlity of node features after
            reduction.
    """
    def __init__(self, out_channels):
        self.out_channels = out_channels

    def __call__(self, data: Data) -> Data:
        if data.x.size(-1) > self.out_channels:
            U, S, _ = torch.linalg.svd(data.x)
            data.x = torch.mm(U[:, :self.out_channels],
                              torch.diag(S[:self.out_channels]))
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.out_channels})'
