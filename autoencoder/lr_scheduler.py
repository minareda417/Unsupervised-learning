import numpy as np
from abc import ABC, abstractmethod

class LRScheduler(ABC):
    def __init__(self, lr):
        # initial learning rate
        self.lr = lr

    @abstractmethod
    def get_lr(self, epoch):
        pass

class StepLRScheduler(LRScheduler):
    def __init__(self, lr, step_size, gamma):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(lr)

    def get_lr(self, epoch):
        return self.lr * (self.gamma ** (epoch // self.step_size))
    

class ExponentialLRScheduler(LRScheduler):
    def __init__(self, lr, gamma):
        self.gamma = gamma
        super().__init__(lr)

    def get_lr(self, epoch):
        return self.lr * (self.gamma ** epoch)
    

class CosineAnnealingLRScheduler(LRScheduler):
    def __init__(self, lr, t_max, lr_min):
        # t_max: max number of epochs
        self.t_max = t_max
        self.lr_min = lr_min
        super().__init__(lr)

    def get_lr(self, epoch):
        return self.lr_min + 0.5 * (self.lr - self.lr_min) * (1 + np.cos(np.pi*epoch/self.t_max))