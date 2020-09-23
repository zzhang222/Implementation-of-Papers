"""
@author: zen
"""
import torch

from .module import StructureNN, Module
    
class Normal_Cell(Module):
    def __init__(self, dim_in, dim_out, hidden_size, cell):
        super(Normal_Cell, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden_size = hidden_size
        self.cell = cell
        self.rnn_cell = self.__init_rnn()
        self.dense = self.__init_W()
    
    def forward(self, hs, state, y):
        # hs is not used if we don't use attention
        state = self.rnn_cell(y, state)
        if self.cell == 'LSTM':
            r = state[0]
        else:
            r = state
        return self.dense['Wn'](r), state 
    
    def __init_rnn(self):
        if self.cell == 'RNN':
            return torch.nn.RNNCell(self.dim_in, self.hidden_size)
        elif self.cell == 'LSTM':
            return torch.nn.LSTMCell(self.dim_in, self.hidden_size)
        elif self.cell == 'GRU':
            return torch.nn.GRUCell(self.dim_in, self.hidden_size)
        else:
            raise NotImplementedError
            
    def __init_W(self):
        dense = torch.nn.ModuleDict()
        dense['Wn'] = torch.nn.Linear(self.hidden_size, self.dim_out, bias = False)
        return dense
    
class Attention_Cell(Module):
    def __init__(self, dim_in, dim_out, hidden_size, cell):
        super(Attention_Cell, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden_size = hidden_size
        self.cell = cell
        self.rnn_cell = self.__init_rnn()
        self.dense = self.__init_W()
    
    def forward(self, hs, state, y):
        state = self.rnn_cell(y, state)
        if self.cell == 'LSTM':
            r = state[0]
        else:
            r = state
        a = torch.softmax(hs @ r.reshape([r.shape[0], self.hidden_size, 1]), dim = 1)
        c = torch.sum(a * hs, dim = 1)
        cr = torch.cat([c, r], dim = 1)
        return self.dense['Wy'](torch.tanh(self.dense['Wn'](cr))), state 
    
    def __init_rnn(self):
        if self.cell == 'RNN':
            return torch.nn.RNNCell(self.dim_in, self.hidden_size)
        elif self.cell == 'LSTM':
            return torch.nn.LSTMCell(self.dim_in, self.hidden_size)
        elif self.cell == 'GRU':
            return torch.nn.GRUCell(self.dim_in, self.hidden_size)
        else:
            raise NotImplementedError
            
    def __init_W(self):
        dense = torch.nn.ModuleDict()
        dense['Wn'] = torch.nn.Linear(2*self.hidden_size, self.hidden_size, bias = False)
        dense['Wy'] = torch.nn.Linear(self.hidden_size, self.dim_out, bias = False)
        return dense
    
class S2S(StructureNN):
    '''Seq2seq model
    Input: [batch size, len_in, dim_in]
    Output: [batch size, len_out, dim_out]
    '''
    def __init__(self, dim_in, len_in, dim_out, len_out, hidden_size=10, cell='LSTM', attention = True):
        super(S2S, self).__init__()
        self.dim_in = dim_in
        self.len_in = len_in
        self.dim_out = dim_out
        self.len_out = len_out
        self.l = (dim_in - dim_out) // 2
        self.hidden_size = hidden_size
        self.cell = cell
        self.attention = attention
        self.encoder = self.__init_encoder()
        self.decoder = self.__init_decoder()
        
    def forward(self, x):
        to_squeeze = True if len(x.size()) == 2 else False
        if to_squeeze:
            x = x.view(1, self.len_in, self.dim_in)
        zeros = torch.zeros([1, x.size(0), self.hidden_size], dtype=x.dtype, device=x.device)
        y = x[:,-1,self.l:self.len_in-self.l].clone()
        pred = []
        init_state = (zeros, zeros) if self.cell == 'LSTM' else zeros
        hs, state = self.encoder(x, init_state)
        state = [torch.squeeze(state[0], dim = 0), torch.squeeze(state[1], dim = 0)] if self.cell == 'LSTM' else torch.squeeze(state, dim = 0)
        for i in range(self.len_out):
            y, state = self.decoder(hs, state, y)
            pred.append(y)
        pred = torch.stack(pred, dim = 1)
        return pred.squeeze(0) if to_squeeze else pred
        
    def __init_encoder(self):
        if self.cell == 'RNN':
            return torch.nn.RNN(self.dim_in, self.hidden_size, batch_first=True)
        elif self.cell == 'LSTM':
            return torch.nn.LSTM(self.dim_in, self.hidden_size, batch_first=True)
        elif self.cell == 'GRU':
            return torch.nn.GRU(self.dim_in, self.hidden_size, batch_first=True)
        else:
            raise NotImplementedError
    
    def __init_decoder(self):
        if self.attention:
            return Attention_Cell(self.dim_out, self.dim_out,self.hidden_size, self.cell)
        else:
            return Normal_Cell(self.dim_out, self.dim_out,self.hidden_size, self.cell)
    
    def predict(self, x0, step, return_np = True):
        preds = []
        for i in range(step):
            preds.append(self(x0))
            x0 = torch.cat([x0[:, self.len_out:, :], self(x0)], dim = 1)
        preds = torch.stack(preds).reshape([-1, self.dim_out])
        if return_np:
            preds = preds.cpu().detach().numpy()
        return preds
