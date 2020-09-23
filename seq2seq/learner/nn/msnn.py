#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 22:49:19 2020

@author: zen
"""

import nodepy.linear_multistep_method as lm
import numpy as np
import torch
from scipy.integrate import odeint
from .module import LossNN
from .fnn import FNN

class MSNN(LossNN):
    '''Multistep neural networks.
    '''
    def __init__(self, dim, layers=3, width=30, activation='tanh', initializer='orthogonal', M = 6, scheme = 'AM'):
        super(MSNN, self).__init__()
        self.dim = dim
        self.layers = layers
        self.width = width
        self.activation = activation
        self.initializer = initializer
        self.M = M
        switch = {'AM': lm.Adams_Moulton,
                  'AB': lm.Adams_Bashforth,
                  'BDF': lm.backward_difference_formula}
        method = switch[scheme](M)
        self.alpha = np.float32(-method.alpha[::-1])
        self.beta = np.float32(method.beta[::-1])  
        self.modus = self.__init_modules()
    
    def criterion(self, x0, h):
        return self.__integrator_loss(x0, h)
    
    def predict(self, x0, h, steps=1):
        prediction = odeint(self.dF, x0, np.arange(steps)*h)
        return prediction
    
    def dF(self, x, t):
        x = torch.tensor(x, dtype = self.Dtype, device = self.Device)
        return self.modus['F'](x).cpu().detach().numpy()
    
    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules['F'] = FNN(self.dim, self.dim, self.layers, self.width, self.activation, self.initializer)
        return modules 
    
    def __integrator_loss(self, x0, h):
        M = self.M
        Y = self.alpha[0]*x0[:,M:,:] + h*self.beta[0]*self.modus['F'](x0[:,M:,:])
        for m in range(1, M+1):
            Y = Y + self.alpha[m]*x0[:,M-m:-m,:] + h*self.beta[m]*self.modus['F'](x0[:,M-m:-m,:])
        return torch.nn.MSELoss()(Y, torch.zeros_like(Y))*self.dim