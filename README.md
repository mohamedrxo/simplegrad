# SimpleGrad

**SimpleGrad** is a lightweight Python library for automatic differentiation and neural networks.  
It is inspired by **PyTorch** and **Tinygrad**, offering a balance between simplicity, flexibility, and educational value.

---

## Features

- **Tensor operations**: `+`, `-`, `*`, `/`, `@` (matmul), `pow`, `exp`, `log`, `ln`, `sin`, `cos`, `tanh`, `relu`, `sigmoid`.  
- **Automatic differentiation**: `.backward()` computes gradients for all operations in the computation graph.  
- **Neural networks**: Build networks using `Linear` layers and activation functions.  
- **Loss functions**: `MSE`, `MAE`, etc. (can be extended).  
- **Gradient descent**: Manual update or custom optimizers can be implemented.  
- **Minimal and educational**: Perfect for learning how autograd works under the hood.

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/mohamedrxo/simplegrad.git
cd simplegrad
pip install -e .
