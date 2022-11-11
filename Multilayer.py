import numpy as np

import Layers


class Multilayerperceptron:

    def __init__(self, in_units, n_layers, n_units):
        """
        Constructs a multilayerperceptron 

            Args: 
            input(array): the perceptrons input
            n_layers(int): number of layers
            n_units(array): number of units per layer
        """
        self.n_layers = n_layers                                            
        self.multilayer = np.empty(n_layers)
        self.multilayer[0] = Layers.Layer(n_units[0], in_units)
        for i in range(1, n_layers):
            self.multilayer[i] = Layers.Layer(n_units[i], n_units[i-1])

    def forward_step(self, input):
        """
        Propagates the input through the layers

        Returns: 
                The predicted output of the MLP given the input 
        """
        self.input = input
        self.output = self.multilayer[0].forward_step(self.input)
        for i in range(self.n_layers):
            self.output = self.multilayer[i].forward_step(self.output)
        return self.output 

    def backpropagation(self, loss):
        """
        Updates the weights and biases of the network
        """
        for count, layer in enumerate(self.multilayer.reverse()):
            if count == 0:
                self.backward[count] = layer.backward_step(loss)
            else: 
                self.backward[count] = layer.backward_step(self.backward[count - 1] * layer.weights)

        return self.multilayer



