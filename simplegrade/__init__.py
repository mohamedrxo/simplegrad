from .tensor import Tensor   # makes Tensor accessible at top level
from .losses import MSELoss
from .optim import SGD
from .linear import Linear
from .dataloader  import DataLoader

__all__ = ["Tensor","MSELoss","SGD","Linear","DataLoader"]         # controls what gets imported with *
__version__ = "0.1.0"
