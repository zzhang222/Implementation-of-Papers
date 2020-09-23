#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 10:50:09 2020

@author: zen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import learner as ln
import torch

class LorenzData(ln.Data):
    '''Data for learning the lorenz system.
    '''
    def __init__(self, x0, h, train_num, test_num, s2s = False, S = 10, T = 4):
        super(LorenzData, self).__init__()
        self.x0 = x0
        self.h = h
        self.train_num = train_num
        self.test_num = test_num
        self.S = S
        self.T = T
        self.__init_data()
        # need to use a sliding window if you are using seq2seq prediction
        if s2s:
            self.__trunk_data()
        
    @property
    def dim(self):
        return 3
    
    def deriv(self, u, t):
        x, y, z = u
        xdot = 10 * (y - x)
        ydot = x * (28 - z) - y
        zdot = x * y - 8/3 * z
        return xdot, ydot, zdot

    def __generate_flow(self, x0, h, num):
        t = np.arange(0, h*num, h)
        X = odeint(self.deriv, x0, t, rtol = 1e-12)
        x = X.reshape([1, X.shape[0], X.shape[1]])
        return x
    
    def __normalize(self, x):
        for i in range(self.dim):
            x[:,:,i] = (x[:,:,i] - self.mini[i])/(self.maxi[i] - self.mini[i])*2-1

    def __init_data(self):
        X_train = self.__generate_flow(self.x0, self.h, self.train_num)
        self.X_train = self.__generate_flow(X_train[0,-1], self.h, self.train_num)
        self.y_train = self.__generate_flow(X_train[0,-1], self.h, self.train_num)
        self.X_test = self.__generate_flow(self.X_train[0,-1], self.h, self.test_num)
        self.y_test = self.__generate_flow(self.X_train[0,-1], self.h, self.test_num)
        self.maxi = np.max(self.X_train, axis = 1)[0]
        self.mini = np.min(self.X_train, axis = 1)[0]
        self.__normalize(self.X_train)
        self.__normalize(self.X_test)
        
    def __trunk_data(self):
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        self.X_train2 = self.X_train
        self.X_test2 = self.X_test
        for i in range(self.train_num - self.S - self.T + 1):
            X_train.append(self.X_train[0,i:i+self.S,:])
            y_train.append(self.X_train[0,i+self.S:i+self.S+self.T,:])
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        for i in range(self.test_num - self.S - self.T + 1):
            X_test.append(self.X_test[0,i:i+self.S,:])
            y_test.append(self.X_test[0,i+self.S:i+self.S+self.T,:])
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        
def plot(data, net, net_type):
    if net_type == 'MSNN':
        X_true = data.X_test[0].cpu().detach().numpy()
        X_predict = net.predict(X_true[0], data.h, X_true.shape[0])
    elif net_type == 'LSTM':
        y_predict, state = net(data.X_train[:,:-1])
        X_true = data.X_test[0].cpu().detach().numpy()
        X_predict = net.predict(data.X_test[:,:1], state, X_true.shape[0], return_np = True)[0]
    elif net_type == 'S2S':
        X_true = data.X_test2[0]
        X0 = torch.cat([data.X_train[-2:-1,-(data.S-data.T):], data.y_train[-2:-1]], dim = 1)
        X_predict = net.predict(X0, data.test_num // data.T, return_np = True)
    plt.plot(np.arange(data.test_num) * data.h, X_true[:,0], label = 'Ground Truth')
    plt.plot(np.arange(data.test_num) * data.h, X_predict[:,0], label = net_type)
    plt.xlabel('t')
    plt.ylabel('X')
    plt.legend()
    plt.title('Lorenz system, X axis')
    
def main():
    device = 'gpu' # 'cpu' or 'gpu'
    net_type = 'S2S' # 'lstm' or 'msnn' or 's2s' or 's2s_p'
    
    # data
    x0 = [2, 1, 1]
    h = 0.05
    train_num = 2000
    test_num = 500
    
    # training
    S = 10
    T = 4
    lr = 0.001
    iterations = 100000
    print_every = 1000
    
    s2s = (net_type == 'S2S')
    data = LorenzData(x0, h, train_num, test_num, s2s, S, T)
    
    if net_type == 'LSTM':
        layers = 1
        cell = net_type
        return_all = True
        width = 30
        net = ln.nn.RNN(data.dim, data.dim, layers, width, cell, return_all)
        criterion = 'MSE'
    elif net_type == 'MSNN':      
        layers = 4
        width = 30
        M = 3
        scheme = 'AM'
        net = ln.nn.MSNN(data.dim, layers, width, M = M, scheme = scheme)
        criterion = None
    elif net_type == 'S2S':
        cell = 'LSTM'
        hidden_size = 30
        attention = False
        net = ln.nn.S2S(data.dim, S, data.dim, T, hidden_size, cell, attention)
        criterion = 'MSE'

    args = {
        'data': data,
        'net': net,
        'criterion': criterion,
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        'batch_size': None,
        'print_every': print_every,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device
    }
    
    ln.Brain.Init(**args)
    if net_type == 'LSTM':
        ln.Brain.Run_rnn()
    elif net_type == 'MSNN':
        ln.Brain.Run_msnn()
    elif net_type == 'S2S':
        ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()
    model = ln.Brain.Best_model()
    plot(data, model, net_type)
    
if __name__ == '__main__':
    main()