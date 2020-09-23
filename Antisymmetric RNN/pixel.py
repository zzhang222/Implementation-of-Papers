#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:18:47 2020

@author: zen
"""

import numpy as np
import learner as ln
import mnist
import torch

class MNISTData(ln.Data):
    def __init__(self):
        super(MNISTData, self).__init__()
        #mnist.init()
        self.X_train, self.y_train, self.X_test, self.y_test = mnist.load()
        self.__prep()
    
    def __prep(self):
        self.X_train = (self.X_train / 255.0 - 0.1307) / 0.3081
        self.X_test = (self.X_test / 255.0 - 0.1307) / 0.3081
        self.X_train = self.X_train.reshape([self.X_train.shape[0], self.X_train.shape[1], 1])
        self.X_test = self.X_test.reshape([self.X_test.shape[0], self.X_test.shape[1], 1])
        
def accuracy(x,y):
    max_x, argmax_x = torch.max(x, dim = 1)
    argmax_x = argmax_x.float()
    return torch.sum(argmax_x == y)/float(x.shape[0])

def callback(data, net):
    batch_size = 500
    num_batch = data.X_test.size(0) // batch_size
    acc = 0
    with torch.no_grad():
        for j in range(num_batch):
            y_test = net(data.X_test[j * batch_size : (j+1) * batch_size])
            acc += accuracy(y_test, data.y_test[j * batch_size : (j+1) * batch_size])
        acc = acc / num_batch
        print('{:<9}Accuracy: {:<25}'.format('', acc), flush = True)

def unison_shuffled_copies(a, b):
    assert a.shape[1] == b.shape[1]
    p = np.random.permutation(a.shape[1])
    return a[:,p], b[:,p]
    
def main():
    device = 'cpu' # 'cpu' or 'gpu'
    net_type = 'ASNN'
    width = 200
    ind = 1
    outd = 10
    # training
    lr = 0.01
    iterations = 100000
    print_every = 100
    batch_size = 500
    multi_gpu = False
    permute = True
    return_all = False
    
    criterion = 'CrossEntropy'
    data = MNISTData()
    if permute:
        data.X_train, data.X_test = unison_shuffled_copies(data.X_train, data.X_test)
    net = ln.nn.RNN(ind, outd, width, net_type, return_all = return_all)
    args = {
        'data': data,
        'net': net,
        'criterion': criterion,
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        'batch_size': batch_size,
        'print_every': print_every,
        'save': True,
        'callback': callback,
        'dtype': 'float',
        'device': device,
        'multi_gpu': multi_gpu
    }
    
    ln.Brain.Init(**args)
    ln.Brain.Run_rnn()
    ln.Brain.Restore()
    ln.Brain.Output()

if __name__ == '__main__':
    main()
