import numpy as np

class Tensor:
    def __init__(self, data, children=(), _op="", label="", grad=None,requires_grad=True):
        # ensure numpy array
        self.data = np.array(data, dtype=float)
        self.grad = None if not requires_grad else (np.zeros_like(self.data) if grad is None else grad)
        self.children = children
        self._op = _op
        self.label = label
        self._backward = lambda: None
        self.shape=self.data.shape
        self.requires_grad=requires_grad

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, op={self._op}, label={self.label})"

    # --- elementwise ops ---
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), _op="+")
        if self.requires_grad or other.requires_grad:
            out.requires_grad=True
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # reduce grad shape if broadcasting happened
                grad_self = out.grad
                while grad_self.ndim > self.data.ndim:
                    grad_self = grad_self.sum(axis=0)
                for i, dim in enumerate(self.data.shape):
                    if dim == 1:
                        grad_self = grad_self.sum(axis=i, keepdims=True)
                self.grad += grad_self

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                grad_other = out.grad
                while grad_other.ndim > other.data.ndim:
                    grad_other = grad_other.sum(axis=0)

                for i, dim in enumerate(other.data.shape):
                    if dim == 1:
                        grad_other = grad_other.sum(axis=i, keepdims=True)
                other.grad += grad_other

        out._backward = _backward
        return out
    
    def __radd__(self, other):
        # Just reverse the order: addition is commutative
        return self + other
    
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, (self, other), _op="-")
        if self.requires_grad or other.requires_grad:
            out.requires_grad=True
        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += -out.grad
        out._backward = _backward
        return out
    
    def __rsub__(self, other):
        # Just reverse the order: subtraction is commutative
        return self - other
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), _op="*")
        if self.requires_grad or other.requires_grad:
            out.requires_grad=True

        def _backward():
            # gradient wrt self
            grad_self = other.data * out.grad
            while grad_self.ndim > self.data.ndim:
                grad_self = grad_self.sum(axis=0)
            for i, dim in enumerate(self.data.shape):
                if dim == 1:
                    grad_self = grad_self.sum(axis=i, keepdims=True)
            if self.requires_grad:
                self.grad += grad_self

            # gradient wrt other
            grad_other = self.data * out.grad
            while grad_other.ndim > other.data.ndim:
                grad_other = grad_other.sum(axis=0)
            for i, dim in enumerate(other.data.shape):
                if dim == 1:
                    grad_other = grad_other.sum(axis=i, keepdims=True)
            if other.requires_grad:
                other.grad += grad_other

        out._backward = _backward
        return out

    
    def __rmul__(self, other):
        # Just reverse the order: multiplication is commutative
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, (self, other), _op="/")
        if self.requires_grad or other.requires_grad:
            out.requires_grad=True

        def _backward():
            # dX = 1/y * dZ
            if self.requires_grad:
                self.grad += (1 / other.data) * out.grad
            # dY = -x / y^2 * dZ
            if other.requires_grad:
                other.grad += (-self.data / (other.data ** 2)) * out.grad

        out._backward = _backward
        return out
    
    def __rtruediv__(self, other):
        # Just reverse the order: division is commutative
        return self / other


    # --- reductions ---
    def sum(self):
        out = Tensor(self.data.sum(), (self,), _op="sum")
        if self.requires_grad :
            out.requires_grad=True
        def _backward():
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    def mean(self):
        out = Tensor(self.data.mean(), (self,), _op="mean")
        if self.requires_grad :
            out.requires_grad=True
        def _backward():
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * out.grad / self.data.size
        out._backward = _backward
        return out

    # --- matrix multiplication ---
    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data.dot(other.data), (self, other), _op="matmul")
        if self.requires_grad or other.requires_grad:
            out.requires_grad=True
        def _backward():
            if self.requires_grad:
                self.grad += out.grad.dot(other.data.T)
            if out.requires_grad:
                other.grad += self.data.T.dot(out.grad)
        out._backward = _backward
        return out
    def __matmul__(self, other):
        return self.matmul(other)

    def __rmatmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other.matmul(self)
    
    def pow(self,other):
        if not isinstance(other, (int, float)):
            raise Exception("other must be int or float")
        if self.requires_grad :
            out.requires_grad=True

        out = Tensor(np.power(self.data,other), (self,), _op="pow")
        def _backward():
            if self.requires_grad:
                self.grad += (other*self.data**(other-1))*out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self):
        out = Tensor(1/(1+np.exp(-self.data)), (self,), _op="sigmoid")
        if self.requires_grad :
            out.requires_grad=True
        def _backward():
            if self.requires_grad:
                self.grad += out.data*(1-out.data)*out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out = Tensor(np.maximum(0,self.data), (self,), _op="relu")
        if self.requires_grad :
            out.requires_grad=True
        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0).astype(float)*out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        out = Tensor(np.tanh(self.data), (self,), _op="tanh")
        if self.requires_grad :
            out.requires_grad=True
        def _backward():
            if self.requires_grad:
                self.grad += (1-out.data**2)*out.grad
        out._backward = _backward
        return out
    

    def exp(self):
        out  = Tensor(np.exp(self.data), (self,), _op="exp")
        if self.requires_grad :
            out.requires_grad=True
        def _backward():
            if self.requires_grad:
                self.grad += out.data*out.grad
        out._backward = _backward
        return out
    
    def cos(self):
        out  = Tensor(np.cos(self.data), (self,), _op="cos")
        if self.requires_grad :
            out.requires_grad=True
        def _backward():
            if self.requires_grad:
                self.grad += -np.sin(self.data)*out.grad
        out._backward = _backward
        return out
    
    def sin(self):
        out  = Tensor(np.sin(self.data), (self,), _op="sin")
        if self.requires_grad :
            out.requires_grad=True
        def _backward():
            if self.requires_grad:
                self.grad += np.cos(self.data)*out.grad
        out._backward = _backward
        return out
    
    def ln(self):
        out  = Tensor(np.log(self.data), (self,), _op="log")
        if self.requires_grad :
            out.requires_grad=True
        def _backward():
            if self.requires_grad:
                self.grad += 1/self.data*out.grad
        out._backward = _backward
        return out
    def log(self,base):
        if base is not isinstance(base,(int,float)):
            raise Exception("base must be int or float")
        out = Tensor(np.log(self.data) / np.log(base), (self,), _op=f"log_{base}")
        if self.requires_grad :
            out.requires_grad=True

        def _backward():
            if self.requires_grad:
                self.grad += (1 / (self.data * np.log(base))) * out.grad

        out._backward = _backward
        return out
    @staticmethod
    def _match_shape(grad, shape):
        """
        Reduce grad to match the given shape by summing over broadcasted axes.
        """
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)

        for i, dim in enumerate(shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad



    def backward(self):
        topo = []
        visited = set()
        if not self.requires_grad:
                return print("no grad please set requires_grad=True")
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()
