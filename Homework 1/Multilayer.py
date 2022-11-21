import numpy as np

import Layers


class Multilayerperceptron:

    def __init__(self, in_units, n_layers, n_units):
        """
        Constructs a multilayerperceptron 

            Args: 
            in_units(int): number of input units
            n_layers(int): number of layers
            n_units(array): number of units per layer
        """
        self.n_layers = n_layers
        self.backward = []                                            
        self.multilayer = []
        self.multilayer.append(Layers.Layer(n_units[0], in_units))
        for i in range(1, n_layers):
            self.multilayer.append(Layers.Layer(n_units[i], n_units[i-1]))

    def forward_step(self, input):
        """
        Propagates the input through the layers

        Returns: 
                The predicted output of the MLP given the input 
        """
        self.input = input
        self.output = self.multilayer[0].forward_step(self.input)
        for i in range(1, self.n_layers):
            self.output = self.multilayer[i].forward_step(self.output)
        return self.output 

    def backpropagation(self, error, n):
        """
        Updates the weights and biases of the network

        Args:
            error: derivative of loss function

        Returns the updated MLP
        """
        for count, layer in enumerate(reversed(self.multilayer)):
            # The last layer takes the derivative of the loss function as the error signal
            if count == 0:
                self.backward.append(layer.backward_step(error, n))
            #all other layers get their error signal from the layer before 
            else: 
                self.backward.append(layer.backward_step(self.backward[count - 1], n))

        return self.multilayer



