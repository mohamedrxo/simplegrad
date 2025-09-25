import numpy as np
from simplegrad import Tensor

class Linear():
    def __init__(self,in_dim,out_dim,bias=True):
        self.w = Tensor(np.random.randn(in_dim,out_dim),requires_grad=True)
        self.b = Tensor(np.random.randn(out_dim),requires_grad=True) if bias else None
        self.bias=bias
        
    def __call__(self, x):
        out = x @ self.w 
        if self.bias:
            out  =  out + self.b
        return out
    def parameters(self):
        return [self.w,self.b]
        