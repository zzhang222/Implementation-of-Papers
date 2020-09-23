"""
@author: jpzxshi
"""
from .module import Module
from .module import StructureNN
from .module import LossNN
from .fnn import FNN
from .seq2seq import S2S
from .msnn import MSNN

__all__ = [
    'Module',
    'StructureNN',
    'LossNN',
    'FNN',
    'S2S',
    'MSNN',
]


