import numpy as np
import random
from sklearn.utils import shuffle
from .activations import ActivationFunction, SigmoidActivation
from .lr_scheduler import LRScheduler

class NeuralNet:
    def __init__(self, *sizes,
                 activation: ActivationFunction = SigmoidActivation(),
                 seed = 0):
        self.nlayers = len(sizes)
        self.sizes = sizes
        self.activation = activation
        np.random.seed(seed)
        # Xavier/Glorot initialization for better gradient flow
        self.weights = [np.random.randn(out_sz, in_sz) * np.sqrt(2.0 / (in_sz + out_sz)) 
                       for in_sz, out_sz in zip(sizes, sizes[1:])] 
        # each array represent weights between each layer 
        # each array's shape is (no. of neurons in current layer, no. of neurons in previous layer)

        self.biases = [np.random.randn(j, 1) for j in sizes[1:]]
        # each array represent the baises the will be added to each neuron in same layer

    def forward(self, a):
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            a = self.activation.activate(w@a + b)
        return a
    
    def train(self, 
              X_train, 
              Y_train, 
              epochs, 
              batch_size=128, 
              eta=0.1, 
              l2_param = 0,
              lr_scheduler: LRScheduler = None):
        """
        X : input data  N*ip_sz*1 (column vectors)
        Y : true labels N*out_sz*1 (column vectors)
        """
        n = len(X_train)
        for nepoch in range(epochs):
            X_train, Y_train = shuffle(X_train, Y_train)
            for i in range(0, len(X_train), batch_size):
                batch_x = np.hstack([x for x in X_train[i:i+batch_size]])
                batch_y = np.hstack([y for y in Y_train[i:i+batch_size]])
                if lr_scheduler is not None:
                    eta = lr_scheduler.get_lr(nepoch)
                self._update_mini_batch(batch_x, batch_y, eta, n, l2_param)
            
            print(f"epoch {nepoch+1:>3}/{epochs} : [{self.evaluate(X_train, Y_train):.6f}]")

    def _update_mini_batch(self, batch_x, batch_y, eta, n, l2_param):
        """
        batch_x : batch of input images  ip_sz*N
        batch_y : batch of true labels  out_sz*N
        """
        gradient_w, gradient_b = self._backprop(batch_x, batch_y)
        m = batch_x.shape[1]
        if l2_param != 0:
            self.weights = [(1 - eta*l2_param/n)*w - (eta/m)*nw 
                            for w, nw in zip(self.weights, gradient_w)]
        else:
            self.weights = [w - (eta/m)*nw for w, nw in zip(self.weights, gradient_w)]
        self.biases = [b - (eta/m)*nb for b, nb in zip(self.biases, gradient_b)]
    
    def evaluate(self, X, Y_true):
        """
        X : batch of input images  N*ip_sz*1
        Y_true : batch of true labels  N*out_sz*1
        """
        Y_pred = self.forward(np.hstack(X))
        Y_true_stacked = np.hstack(Y_true)
        mse = np.mean(np.square(Y_pred - Y_true_stacked))
        return mse

    def _backprop(self, X, Y):
        """
        X : batch of input images  ip_sz*N
        Y : batch of true labels  out_sz*N
        """
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        gradient_b = [np.zeros(b.shape) for b in self.biases]    

        a = X
        activations = [X]  # store all the activations layer by layer
        z_values = [] # store all the z vectors layer by layer
        z = None

        for w, b in zip(self.weights, self.biases):
            z = w@a + b; z_values.append(z) 
            a = self.activation.activate(z); activations.append(a)
        
        #             out_sz*N                    out_sz*N
        delta = self._cost_func_deriv(activations[-1], Y) * self.activation.derivative(z_values[-1])  # dL/dz = dL/da * da/dz
        #                out_sz*N        (input_layer_size*N).T
        gradient_w[-1] = delta @ activations[-2].T # dL/dw = dL/dz * dz/dw
        #                          out_sz*N -> out_sz*1
        gradient_b[-1] = np.sum(delta, axis=1, keepdims=True) # dL/db = dL/dz * dz/db

        for l in range(2, self.nlayers):
            # assume current layer weights have size (curr_out * curr_ip)
            # assume next layer weights have size (next_out * next_ip)
            # then (next_ip = curr_out) 
            # {((next_out * next_ip).T @ next_out*N -> next_ip*N)  .  curr_out*N} ->  curr_out*N
            delta = (self.weights[-l+1].T @ delta) * self.activation.derivative(z_values[-l]) # dL/dz for layer l
            #           curr_out*N  @  (curr_ip * N).T        
            gradient_w[-l] = delta @ activations[-l-1].T # dL/dw for layer l
            #            curr_out*N -> curr_out*1
            gradient_b[-l] =  np.sum(delta, axis=1, keepdims=True) # dL/db for layer l

        return gradient_w, gradient_b
            

    def _cost_func_deriv(self, y, y_true):
        return (y-y_true)

