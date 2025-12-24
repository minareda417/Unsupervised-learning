import numpy as np
from abc import ABC, abstractmethod

class LRScheduler(ABC):
    """
    Abstract base class for learning rate schedulers.
    
    Provides interface for dynamically adjusting learning rate during training.
    """
    
    def __init__(self, lr):
        """
        Initialize the learning rate scheduler.
        
        Args:
            lr: Initial learning rate
        """
        # Store initial learning rate
        self.lr = lr

    @abstractmethod
    def get_lr(self, epoch):
        """
        Calculate learning rate for given epoch.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Learning rate for the current epoch
        """
        pass

class StepLRScheduler(LRScheduler):
    """
    Step decay learning rate scheduler.
    
    Reduces learning rate by a multiplicative factor at fixed intervals.
    Formula: lr * (gamma ^ (epoch // step_size))
    """
    
    def __init__(self, lr, step_size, gamma):
        """
        Initialize step decay scheduler.
        
        Args:
            lr: Initial learning rate
            step_size: Number of epochs between learning rate decay steps
            gamma: Multiplicative factor for learning rate decay (e.g., 0.1)
        """
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(lr)

    def get_lr(self, epoch):
        """
        Calculate learning rate using step decay.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Decayed learning rate
        """
        return self.lr * (self.gamma ** (epoch // self.step_size))
    

class ExponentialLRScheduler(LRScheduler):
    """
    Exponential decay learning rate scheduler.
    
    Reduces learning rate exponentially each epoch.
    Formula: lr * (gamma ^ epoch)
    """
    
    def __init__(self, lr, gamma):
        """
        Initialize exponential decay scheduler.
        
        Args:
            lr: Initial learning rate
            gamma: Exponential decay rate (e.g., 0.95 for 5% decay per epoch)
        """
        self.gamma = gamma
        super().__init__(lr)

    def get_lr(self, epoch):
        """
        Calculate learning rate using exponential decay.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Exponentially decayed learning rate
        """
        return self.lr * (self.gamma ** epoch)
    

class CosineAnnealingLRScheduler(LRScheduler):
    """
    Cosine annealing learning rate scheduler.
    
    Gradually decreases learning rate following a cosine curve from initial
    value to minimum value over t_max epochs. Provides smooth decay and
    often improves convergence.
    """
    
    def __init__(self, lr, t_max, lr_min):
        """
        Initialize cosine annealing scheduler.
        
        Args:
            lr: Initial (maximum) learning rate
            t_max: Total number of epochs for one annealing cycle
            lr_min: Minimum learning rate at the end of the cycle
        """
        # Maximum number of epochs for annealing schedule
        self.t_max = t_max
        # Minimum learning rate to reach
        self.lr_min = lr_min
        super().__init__(lr)

    def get_lr(self, epoch):
        """
        Calculate learning rate using cosine annealing.
        
        Formula: lr_min + 0.5 * (lr - lr_min) * (1 + cos(π * epoch / t_max))
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Learning rate following cosine annealing schedule
        """
        return self.lr_min + 0.5 * (self.lr - self.lr_min) * (1 + np.cos(np.pi*epoch/self.t_max))