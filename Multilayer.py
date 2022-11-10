import numpy as np 
import Layers

class Multilayerperceptron:

    def __init__(self, input, n_layers, n_units ):
        """
        Constructs a multilayerperceptron 

            Args: 
            input(array): the perceptrons input
            n_layers(int): number of layers
            n_units(array): number of units per layer
        """
        self.n_layers = n_layers                                            
        self.input = input
        self.multilayer = np.empty(n_layers)
        self.multilayer[0] = Layers.Layer(n_units[0], len(self.input))
        for i in range(1, n_layers):
            self.multilayer[i] = Layers.Layer(n_units[i], n_units[i-1])

    def forward_step(self):
        """
        Propagates the input through the layers
        """
        self.output = self.multilayer[0].forward_step(self.input)
        for i in range(self.n_layers):
            self.output = self.multilayer[i].forward_step(self.output)
        return self.output 

    def backpropagation(self):
        """
        Updates the weights and biases of the network
        """
        for i in range(self.n_layers, 0, -1):
            self.multilayer[i].backward_step()

        return self.multilayer


