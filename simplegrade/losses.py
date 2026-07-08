import numpy as np

class MSELoss:
    def __init__(self):
        self.pred = None
        self.target = None

    def __call__(self, pred, target):
        self.pred = pred
        self.target = target
        return ((pred - target) * (pred - target)).mean()
