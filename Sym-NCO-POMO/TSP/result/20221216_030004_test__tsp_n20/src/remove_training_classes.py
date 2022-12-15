from typing import List

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class RemoveTrainingClasses(BaseTransform):
    r"""Removes classes from the node-level training set as given by
    :obj:`data.train_mask`, *e.g.*, in order to get a zero-shot label scenario.

    Args:
        classes (List[int]): The classes to remove from the training set.
    """
    def __init__(self, classes: List[int]):
        self.classes = classes

    def __call__(self, data: Data) -> Data:
        data.train_mask = data.train_mask.clone()
        for i in self.classes:
            data.train_mask[data.y == i] = False
        return data

    def __repr__(self) -> str:
        return 'f{self.__class__.__name__}({self.classes})'
