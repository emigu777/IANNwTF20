import numpy as np

class layer:

    def __init__(self, n_units, input_units):       #constructor
        self.n_units = int(n_units)                 #number of units in layer
        self.input_units = int(input_units)         #number of units in preceding layer
        self.bias = np.zeros(self.n_units)          #array filled with zeros as biases
        self.weights = np.random.uniform(-10, 10, size=( self.input_units, self.n_units))   #matrix filled with random floats between -10 and 10 as weights
        self.input = np.empty(self.input_units)     #empty array for inputs in the layer
        self.preactivation = np.empty(self.n_units) #empty array for preactivation values
        self.activation = np.empty(self.n_units)    #empty array for activation values

    def forward_step(self):
        self.preactivation = np.dot(self.input, self.weights)  #multiply weights matrix with input array and safe results to preactivation array
        for i in range(self.n_units):
            self.preactivation[i] += self.bias[i]              #add biases to each preactivation to get final preactivation
            self.activation[i] = np.maximum(0, self.preactivation[i])   #ReLu (preactivation) = activation
        return self.activation

    #def backward_step(self):




l1 = layer(4, 5)
print(l1.bias)
print(l1.weights)
l1.input = (1, 1, 0, 0, 0)
print(l1.input)
l1.forward_step()
print(l1.preactivation)
print(l1.activation)