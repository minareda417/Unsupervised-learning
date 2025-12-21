from abc import ABC, abstractmethod
import numpy as np

class ActivationFunction(ABC):
    @abstractmethod
    def activate(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass

class SigmoidActivation(ActivationFunction):
    def activate(self, x):
        # Clip values to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    def derivative(self, x):
        s = self.activate(x)
        return s * (1 - s)
    
class ReLUActivation(ActivationFunction):
    def activate(self, x):
        return np.maximum(0, x)
    
    def derivative(self, x):
        return np.where(x > 0, 1, 0)
    
class TanhActivation(ActivationFunction):
    def activate(self, x):
        return np.tanh(x)
    
    def derivative(self, x):
        return 1 - np.tanh(x)**2