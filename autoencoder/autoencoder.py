from .neuralnet import NeuralNet
from .activations import SigmoidActivation

class Autoencoder(NeuralNet):
    def __init__(self, *sizes, activation = SigmoidActivation(), seed=0):
        sizes = sizes + sizes[-2::-1]
        super().__init__(*sizes, activation=activation, seed=seed)

    def encode(self, a):
        for i in range(len(self.weights)//2):
            w = self.weights[i]
            b = self.biases[i]

            a = self.activation.activate(w@a +b)
        return a

    def decode(self, a):
        for i in range(len(self.weights)//2, len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            a = self.activation.activate(w@a +b)
        return a


