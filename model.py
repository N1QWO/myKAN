import torch
import torch.nn as nn
    
class KANLayer(nn.Module):
    def __init__(self,in_,slice_,lr=1e-3,alfa=0.9,beta=0.99,labd=0.99,b=5):
        super(KANLayer, self).__init__()
        self.W = {
            'Wbb': nn.Parameter(torch.rand((1, b, slice_))/torch.sqrt(torch.tensor(slice_))),
            'Wb': nn.Parameter(torch.rand((1, in_, slice_))/torch.sqrt(torch.tensor(slice_))),
            'Wc': nn.Parameter(torch.rand((in_, 1)))
        }
        self.q = nn.Parameter(torch.ones((1, in_, b)))
        self.b = b
        self.m = nn.Parameter(((torch.arange(self.b) - self.b // 2)*torch.sqrt(torch.tensor(self.b))).unsqueeze(0).unsqueeze(0))
        self.slice = slice_
        #self.u = {k: torch.zeros_like(v) for k, v in self.W.items()}
        #self.g = {k: torch.zeros_like(v) for k, v in self.W.items()}
        self.eps = 1e-8
        self.lr = lr
        self.alfa = alfa
        self.beta = beta
        self.labd = labd
        self.backprop = {}

    def sigmoid(self,x):
        return 1/(1+torch.exp(-x))
    def Silu(self,x):
        return x*self.sigmoid(x)


    def phsi(self,x):
        kn = torch.sqrt(torch.tensor(2 * torch.pi))
        return torch.exp(-((x - self.m) / self.q) ** 2 / 2) / (kn*self.q)
    
    # def der_phsi(self,x,m):
    #     q = 1
    #     return  ((m-x)/q) * self.backprop['phsi'] 
    # def der_sigmoid(self,x):
    #     return self.sigmoid(x)*(1-self.sigmoid(x))
    # def der_SiLU(self,x):
    #     return self.sigmoid(x) + x*self.der_sigmoid(x)


    # def backward(self, grad_output):
    #     '''
    #     grad_output: gradient coming from the next layer (batch_size, n, 1)
    #     '''
    #     grad_output = grad_output.unsqueeze(-1)  # match dimensions
    #     self.grad = {}
    #     self.grad['Wb'] = torch.mean(self.backprop['B'] * grad_output, axis=0)  # (1, in_, slice)
    #     self.grad['Wbb'] = torch.mean(torch.matmul(self.backprop['phsi'].transpose(1, 2), self.W['Wb']) * grad_output, axis=0)  # (b, slice)
    #     self.grad['Wc'] = torch.mean(self.backprop['Silu'] * grad_output, axis=0)  # (in_, 1)

    #     for sr in ['Wb', 'Wbb', 'Wc']:
    #         tm = self.grad[sr]
    #         self.u[sr] = self.beta * self.u[sr] + (1 - self.beta) * tm
    #         self.g[sr] = self.labd * self.g[sr] + (1 - self.labd) * (tm ** 2)
    #         self.W[sr] = self.W[sr] - self.lr * self.u[sr] / (torch.sqrt(self.g[sr] + self.eps))

    #     #self.grad['x'] = self.backprop['x']*
    #     #return self.grad['x']

    def forward(self, x):
        '''
            x: Tensor of shape (batch_size, n)
            return: Tensor of shape (batch_size, n*slice)
        '''
        batch_size, n = x.shape
        x = x.unsqueeze(-1)  # (batch_size, n, 1)

       # half = self.b // 2
        #m_values = torch.arange(self.b) - half
        #m_values = m_values.unsqueeze(0).unsqueeze(0)  # (1, 1, b)

        #self.backprop['der_SiLU'] = self.der_SiLU(x)  # (batch_size, n, 1)
        self.backprop['phsi'] = self.phsi(x)  # (batch_size, n, b)
        self.backprop['B'] = torch.matmul(self.backprop['phsi'], self.W['Wbb'])  # (batch_size, n, slice)
        self.backprop['Silu'] = self.Silu(x)  # (batch_size, n, 1)
        self.backprop['S'] = self.backprop['Silu'] * self.W['Wc']  # (batch_size, n, 1)
        self.backprop['Bw'] = self.backprop['B'] * self.W['Wb']  # (batch_size, n, slice)
        self.backprop['result'] = self.backprop['S'] + self.backprop['Bw']  # (batch_size, n, slice)

        # Вычисление градиентов
        #self.backprop['x'] = self.der_SiLU(x) * self.W['Wc'] + torch.matmul(self.der_phsi(x, m_values), self.W['Wbb']) * self.W['Wb']

        return self.backprop['result'].reshape(batch_size, -1)




