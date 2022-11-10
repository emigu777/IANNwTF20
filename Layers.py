import numpy as np

class layer:

    def __init__(self, n_units, input_units):       #constructor
        self.n_units = int(n_units)                 #
        self.input_units = int(input_units)
        self.bias = np.zeros(self.n_units)
        self.weights = np.random.uniform(-10, 10, size=( self.input_units, self.n_units))
        self.input = np.empty(self.input_units)
        self.preactivation = np.empty(self.n_units)
        self.activation = np.empty(self.n_units)

    def forward_step(self):
        self.preactivation = np.dot(self.input, self.weights)  # multiply matrix with array
        for i in range(self.n_units):
            self.preactivation[i] += self.bias[i]
            self.activation[i] = np.maximum(0, self.preactivation[i])
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