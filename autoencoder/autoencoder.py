from .neuralnet import NeuralNet
from .activations import SigmoidActivation

class Autoencoder(NeuralNet):
    """
    Autoencoder neural network for unsupervised learning and dimensionality reduction.
    
    Inherits from NeuralNet and creates a symmetric architecture where the decoder
    mirrors the encoder layers in reverse order.
    """
    
    def __init__(self, *sizes, activation = SigmoidActivation(), seed=0):
        """
        Initialize the autoencoder with a symmetric architecture.
        
        Args:
            *sizes: Variable number of layer sizes for the encoder half.
                   The decoder structure is automatically created as the mirror.
                   Example: (784, 128, 64) creates encoder: 784->128->64
                   and decoder: 64->128->784
            activation: Activation function to use (default: SigmoidActivation)
            seed: Random seed for weight initialization
        """
        # Create symmetric architecture by mirroring encoder layers (excluding bottleneck)
        sizes = sizes + sizes[-2::-1]
        super().__init__(*sizes, activation=activation, seed=seed)

    def encode(self, a):
        """
        Encode input data to compressed latent representation.
        
        Applies forward propagation through the first half of the network
        (encoder layers) to produce the bottleneck/latent representation.
        
        Args:
            a: Input data to encode
            
        Returns:
            Compressed latent representation
        """
        # Forward pass through encoder layers (first half of network)
        for i in range(len(self.weights)//2):
            w = self.weights[i]
            b = self.biases[i]
            # Apply linear transformation followed by activation
            a = self.activation.activate(w@a +b)
        return a

    def decode(self, a):
        """
        Decode latent representation back to original data space.
        
        Applies forward propagation through the second half of the network
        (decoder layers) to reconstruct the input from latent representation.
        
        Args:
            a: Latent representation to decode
            
        Returns:
            Reconstructed data in original space
        """
        # Forward pass through decoder layers (second half of network)
        for i in range(len(self.weights)//2, len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            # Apply linear transformation followed by activation
            a = self.activation.activate(w@a +b)
        return a


