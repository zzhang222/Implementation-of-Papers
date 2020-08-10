#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 18:42:36 2020

@author: zen
"""

import torch
import torch.nn as nn
from .module import Module, StructureNN
        
class ASCell(Module):
    '''Antisymmetric cell
    '''
    def __init__(self, ind, width, activation, gamma, eps):
        super(ASCell, self).__init__()
        self.ind = ind
        self.width = width
        self.activation = activation
        self.gamma = gamma
        self.eps = eps
        
        self.W = nn.Parameter((torch.empty([self.width, self.width])).requires_grad_(True))
        self.V = nn.Parameter((torch.empty([self.ind, self.width])).requires_grad_(True))
        self.b = nn.Parameter(torch.zeros([self.width]).requires_grad_(True))
        torch.nn.init.xavier_uniform(self.W)
        torch.nn.init.xavier_uniform(self.V)
    
    def forward(self, inputs, state):
        return state + self.eps * self.activation(state @ (self.W - self.W.t()) - self.gamma * state + inputs @ self.V + self.b)

class ASNN(Module):
    def __init__(self, ind, width, gamma = 0.01, eps = 0.01):
        super(ASNN, self).__init__()
        self.ind = ind
        self.width = width
        self.gamma = gamma
        self.eps = eps
        self.cell = ASCell(self.ind, self.width, torch.tanh, self.gamma, self.eps)
       
    def forward(self, x, init_state):
        state = init_state[0]
        inputs = x.unbind(1)
        outputs = []
        for i in range(len(inputs)):
            state = self.cell(inputs[i], state)
            outputs.append(state)
        return torch.stack(outputs, dim = 1), state

class RNN(StructureNN):
    def __init__(self, ind, outd, width, cell, return_all):
        super(RNN, self).__init__()
        self.ind = ind
        self.outd = outd
        self.width = width
        self.cell = cell
        self.modus = self.__init_modules()
        self.return_all = return_all   
 
    def forward(self, x):
        to_squeeze = True if len(x.size()) == 2 else False
        if to_squeeze:
            x = x.view(1, self.len_in, self.dim_in)
        zeros = torch.zeros([1, x.size(0), self.width], dtype=x.dtype, device=x.device)
        init_state = (zeros, zeros) if self.cell == 'LSTM' else zeros
        x, _ = self.modus['RNN'](x, init_state)
        if self.return_all:
            y = self.modus['LinMOut'](x)
        else:
            y = self.modus['LinMOut'](x[:,-1])
        return y
    
    def __init_modules(self):
        modules = nn.ModuleDict()
        if self.cell == 'RNN':
            modules['RNN'] = nn.RNN(self.ind, self.width, batch_first=True)
        elif self.cell == 'LSTM':
            modules['RNN'] =  nn.LSTM(self.ind, self.width, batch_first=True)
        elif self.cell == 'GRU':
            modules['RNN'] = nn.GRU(self.ind, self.width, batch_first=True)
        elif self.cell == 'ASNN':
            modules['RNN'] = ASNN(self.ind, self.width)
        else:
            raise NotImplementedError
            
        modules['LinMOut'] = nn.Linear(self.width, self.outd)
        return modules
