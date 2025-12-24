from abc import ABC, abstractmethod
import numpy as np

class ActivationFunction(ABC):
    """Abstract base class for activation functions."""
    
    @abstractmethod
    def activate(self, x):
        """Apply activation function."""
        pass

    @abstractmethod
    def derivative(self, x):
        """Compute derivative for backpropagation."""
        pass

class SigmoidActivation(ActivationFunction):
    """Sigmoid activation: σ(x) = 1 / (1 + e^(-x)), range: (0, 1)"""
    
    def activate(self, x):
        # Clip values to prevent overflow in exp(-x)
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    def derivative(self, x):
        # σ'(x) = σ(x) * (1 - σ(x))
        s = self.activate(x)
        return s * (1 - s)
    
class ReLUActivation(ActivationFunction):
    """ReLU activation: f(x) = max(0, x), range: [0, ∞)"""
    
    def activate(self, x):
        return np.maximum(0, x)
    
    def derivative(self, x):
        # f'(x) = 1 if x > 0, else 0
        return np.where(x > 0, 1, 0)
    
class TanhActivation(ActivationFunction):
    """Tanh activation: tanh(x), range: (-1, 1)"""
    
    def activate(self, x):
        return np.tanh(x)
    
    def derivative(self, x):
        # f'(x) = 1 - tanh²(x)
        return 1 - np.tanh(x)**2