import numpy as np
import random

class NeuralNet:
    def __init__(self,  *sizes):
        self.nlayers = len(sizes)
        self.sizes = sizes
        np.random.seed(0)
        self.weights = [np.random.randn(out_sz, in_sz) for in_sz , out_sz in zip(sizes, sizes[1:])] 
        # each array represent weights between each layer 
        # each array's shape is (no. of neurons in current layer, no. of neurons in previous layer)

        self.biases = [np.random.randn(j, 1) for j in sizes[1:]]
        # each array represent the baises the will be added to each neuron in same layer

    def feed_forward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(w@a + b)
        return a
    
    def SGD(self, training_data, epochs, batch_size, eta, test_data = None):
        for nepoch in range(epochs):
            np.random.shuffle(training_data)
            batches = [training_data[i:i+batch_size] for i in range (0, len(training_data), batch_size)]
            for mini_batch in batches:
                self.update_mini_batch(mini_batch, eta)
            print(f"epoch {nepoch} : {self.evaluate(test_data)}/{len(test_data)}")

    def update_mini_batch(self, mini_batch, eta):
        X = np.hstack([x for x, _ in mini_batch])
        Y = np.hstack([y for _, y in mini_batch])

        gradient_w, gradient_b = self.backprop(X, Y)
        m = len(mini_batch)

        self.weights = [w - (eta/m)*nw for w, nw in zip(self.weights, gradient_w)]
        self.biases = [b - (eta/m)*nb for b, nb in zip(self.biases, gradient_b)]
    
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def backprop(self, X, Y):
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
            a = self.sigmoid(z); activations.append(a)
        
        #             out_sz*N                    out_sz*N
        delta = self.cost_func_deriv(a, Y) * self.sigmoid_deriv(z) # dL/dz = dL/da * da/dz
        #                out_sz*N        (input_layer_size*N).T
        gradient_w[-1] = delta @ activations[-2].T # dL/dw = dL/dz * dz/dw
        #                          out_sz*N -> out_sz*1
        gradient_b[-1] = np.sum(delta, axis=1, keepdims=True) # dL/db = dL/dz * dz/db

        for l in range(2, self.nlayers):
            # assume current layer weights have size (curr_out * curr_ip)
            # assume next layer weights have size (next_out * next_ip)
            # then (next_ip = curr_out) 
            # {((next_out * next_ip).T @ next_out*N -> next_ip*N)  .  curr_out*N} ->  curr_out*N
            delta = (self.weights[-l+1].T @ delta) * self.sigmoid_deriv(z_values[-l]) # dL/dz for layer l
            #           curr_out*N  @  (curr_ip * N).T        
            gradient_w[-l] = delta @ activations[-l-1].T # dL/dw for layer l
            #            curr_out*N -> curr_out*1
            gradient_b[-l] =  np.sum(delta, axis=1, keepdims=True) # dL/db for layer l

        return gradient_w, gradient_b
            

    def cost_func_deriv(self, y, y_true):
        return (y-y_true)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_deriv(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    

if __name__ == "__main__":
    pass
