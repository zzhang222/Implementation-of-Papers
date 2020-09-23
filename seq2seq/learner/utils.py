"""
@author: jpzxshi
"""
from functools import wraps
import time

import numpy as np
import torch

#
# Useful tools.
#
def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t = time.time()
        result = func(*args, **kwargs)
        print('\'' + func.__name__ + '\'' + ' took {} s'.format(time.time() - t))
        return result
    return wrapper

class lazy_property:
    def __init__(self, func): 
        self.func = func 
        
    def __get__(self, instance, cls): 
        val = self.func(instance)
        setattr(instance, self.func.__name__, val)
        return val
    
#
# Numpy tools.
#
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

#
# Torch tools.
#
def cross_entropy_loss(y_pred, y_label):
    if y_pred.size() == y_label.size():
        return torch.mean(-torch.sum(torch.log_softmax(y_pred, dim=-1) * y_label, dim=-1))
    else:
        return torch.nn.CrossEntropyLoss()(y_pred, y_label.long())