from .tensor import Tensor   # makes Tensor accessible at top level
from .losses import MSELoss
from .optim import SGD
from .linear import Linear

__all__ = ["Tensor","MSELoss","SGD","Linear"]         # controls what gets imported with *
__version__ = "0.1.0"
