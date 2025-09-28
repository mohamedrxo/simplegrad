import numpy as np
from simplegrad import Tensor

# working version
class Linear:
    def __init__(self, in_dim, out_dim, bias=True):
        # Better weight initialization - Xavier/Glorot initialization
        self.w = Tensor(np.random.randn(in_dim, out_dim) / np.sqrt(in_dim), requires_grad=True)
        self.b = Tensor(np.zeros(out_dim), requires_grad=True) if bias else None
        self.bias = bias
        
    def __call__(self, x):
        out = x @ self.w 
        if self.bias:
            out = out + self.b
        return out
    
    def parameters(self):
        params = [self.w]
        if self.b is not None:
            params.append(self.b)
        return params
    
# non working version 

# 1 deviding by sqrt(in_dim) let to better model initialization
# 2 the bias should ne initualized to zero instead of random
# 3 [self.w,self.b] can produce [self.w,None]

# class Linear():
#     def __init__(self,in_dim,out_dim,bias=True):
#         self.w = Tensor(np.random.randn(in_dim,out_dim),requires_grad=True)
#         self.b = Tensor(np.random.randn(out_dim),requires_grad=True) if bias else None
#         self.bias=bias
        
#     def __call__(self, x):
#         out = x @ self.w 
#         if self.bias:
#             out  =  out + self.b
#         return out
#     def parameters(self):
#         return [self.w,self.b]