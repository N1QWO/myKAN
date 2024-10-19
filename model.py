import numpy as np 

    
class KANlayer():
    def __init__(self,in_,out_,lr=1e-3,alfa=0.9,beta=0.99,labd=0.99,slice = 5,b=10):
        self.W = {
            'Wbb': np.ones((1,b,slice)),
            'Wb':  np.ones((1,in_,slice)),
            'Wc': np.ones((in_,1)),
        }
        self.b = b
        self.slice = slice
        self.grad = {
                'Wbb':np.array(),
                'Wb': np.array(),
                'Wc': np.array()
                }
        self.u = {
            'Wbb': np.zeros((1,b,slice)),
            'Wb':  np.zeros((1,in_,slice)),
            'Wc': np.zeros((in_,1)),
        }
        self.g = {
            'Wbb': np.zeros((1,b,slice)),
            'Wb':  np.zeros((1,in_,slice)),
            'Wc': np.zeros((in_,1)),
        }
        self.eps = 1e-8
        self.lr = lr
        self.alfa = alfa
        self.beta = beta
        self.labd = labd
        self.backprop = {}

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def Silu(self,x):
        return x*self.sigmoid(x)


    def phsi(self,x,m):
        q = 1
        return np.exp(((x - m) / q) ** 2 / 2) / (np.sqrt(2 * np.pi))
    
    def der_phsi(self,x,m):
        q = 1
        return  ((m-x)/q) * self.backprop['phsi'] 
    def der_sigmoid(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    def der_SiLU(self,x):
        return self.sigmoid(x) + x*self.der_sigmoid(x)

    def bacward(self,grad):
        '''
            grad shape (samples , in_, 1)
        '''
        self.grad['Wb'] = np.mean(self.backprop['B'] * grad,axis=1)  # (1, in_, slice)  = mean (samples, in_, slice)  
        self.grad['Wbb'] = np.mean(np.matmul(self.backprop['phsi'].transpose(0, 2, 1), self.W['Wb']) * grad ,axis=1)  # (b, slice) = mean ((samples, in_, b).T @ (1, in_, slice))

        self.grad['Wc'] = np.mean(self.backprop['Silu'] * grad ,axis=1)   # (in_, 1) = mean ((samples , in_, 1)) * (samples , in_,)
        self.grad['der_Silu'] = self.backprop['der_SiLU'] # (samples, in_, 1) = (samples , in_, 1)
        self.grad['x'] = self.backprop['x']
        for sr in ['Wb','Wbb','Wc']:
            tm = self.grad[sr]
            self.u[sr] = self.beta*self.u[sr] + (1-self.beta)*tm
            self.g[sr] = self.labd*self.g[sr] + (1-self.labd)*(tm)**2
            self.W[sr] = self.W[sr] - self.lr * self.u[sr] / (np.sqrt(self.g[sr]+1e-10))


    def forward(self,x):
        '''
            x: vector
            [n] -> [n*sum(b,slice)] --> [n*slice]
            return [n] -> [n*slice]        

        '''

        x = x[:, :, np.newaxis]  # (samples , n, 1)

        half = int(self.b/2)
        m_values = np.arange(self.b) - half
        m_values = m_values[np.newaxis, np.newaxis, :]  # (1, 1, b)

        self.backprop['der_SiLU'] = self.der_SiLU(x) # (samples, n, 1)

        self.backprop['phsi'] = self.phsi(x,m_values) # (samples, n, b)
        self.backprop['B'] = np.matmul(self.backprop['phsi'],self.W['Wbb']) # (samples, n, slice) = (samples, n, b) @ (b, slice) =

        self.backprop['Silu'] = self.Silu(x) # (samples , n, 1)
        self.backprop['S'] = self.backprop['Silu'] * self.W['Wc'] # (samples , n, 1) = (samples , n, 1) * (n, 1) 

        self.backprop['Bw'] = self.backprop['B']  * self.W['Wb'] # (samples, n, slice) = (samples, n, slice) * (1, n, slice) 

        self.backprop['result'] = self.backprop['S']  + self.backprop['Bw'] # (samples, n, slice) = (samples , n, 1) + (samples, n, slice) 
        self.backprop['x'] = self.der_SiLU(x) * self.W['Wc'] +  np.matmul(self.der_phsi(x,m_values),self.W['Wbb']) * self.W['Wb'] # (samples, n, slice) = (samples , n, 1) + (samples, n, slice) 

        return self.backprop['result'].reshape(self.backprop['result'][0], -1)

