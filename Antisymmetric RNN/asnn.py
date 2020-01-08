import torch
import torch.jit as jit
from torch import nn
import time

class ASNNCell(jit.ScriptModule):
    __constants__ = ['gamma', 'epsilon', 'hidden_size']

    def __init__(self, input_size, hidden_size, sigma, gamma, epsilon):
        super(ASNNCell, self).__init__()
        self.weight_ih = nn.Parameter(torch.randn(hidden_size, input_size)/input_size)
        self.weight_hh = nn.Parameter(torch.randn(hidden_size, hidden_size)*sigma*sigma/hidden_size)
        self.hidden_size = hidden_size
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.gamma = gamma
        self.epsilon = epsilon

    @jit.script_method
    def forward(self, inputs, hx):
        hy = hx + self.epsilon * torch.tanh(torch.mm(inputs, self.weight_ih.t()) +
                 torch.mm(hx, (self.weight_hh.t()-self.weight_hh - self.gamma*torch.eye(self.hidden_size).cuda())) + self.bias)

        return hy

class ASNNLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(ASNNLayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, inputs, state):
        inputs = inputs.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            state = self.cell(inputs[i], state)
            outputs += [state]
        return torch.stack(outputs),[state]


class Net1(jit.ScriptModule):
    def __init__(self, layer, rnn_size, output_size):
        super(Net1, self).__init__()
        self.layer = layer
        self.weight = nn.Parameter(torch.randn(rnn_size, output_size)*0.1)
        self.bias = nn.Parameter(torch.zeros(output_size))
    
    @jit.script_method
    def forward(self, inputs, initial_state):
        outputs, state = self.layer(inputs, initial_state)
        return torch.mm(state[0],self.weight)+self.bias

def train(trainloader, valloader, optimizer, net, initial_state, niter=150):
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    test_losses = []
    for epoch in range(niter):  # loop over the dataset multiple times

        running_loss = 0.0
        start = time.time()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            inputs = inputs.reshape(inputs.shape[0],784).t().reshape(784,
                                inputs.shape[0],1)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs, initial_state).reshape(-1,10)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                test_loss = 0
                test_acc = 0
                cnt = 0
                with torch.no_grad():
                    for j, test_data in enumerate(valloader, 0):
                        test_inputs, test_labels = data
                        test_inputs = test_inputs.cuda()
                        test_labels = test_labels.cuda()
                        test_inputs = test_inputs.reshape(test_inputs.shape[0],
                                      784).t().reshape(784,test_inputs.shape[0],1)
                        test_outputs = net(test_inputs, initial_state)
                        _, predicted = torch.max(test_outputs, 1)
                        test_acc = torch.mean((predicted == test_labels).double())\
                        + test_acc
                        test_loss = criterion(test_outputs, test_labels)+ test_loss
                        cnt = cnt + 1
                    test_loss = test_loss / cnt
                    test_acc = test_acc / cnt
                end = time.time()
                print('Time elapse: %.5f' % (end - start))
                start = time.time()
                print('[%d, %5d] loss: %.5f test_loss: %.5f test_acc: %.5f' %
                      (epoch + 1, i + 1, running_loss / 200, test_loss, test_acc))
                train_losses.append(running_loss / 200)
                test_losses.append(test_loss)
                running_loss = 0.0

    print('Finished Training')
    return test_losses