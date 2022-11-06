import numpy as np

class Layer:

    def __init__(self, n_units, input_units):
        self.n_units = int(n_units)
        self.input_units = int(input_units)
        self.bias = np.zeros(self.n_units)
        self.weights = np.random.rand([self.input_units, self.n_units])
        self.input = np.empty(self.input_units)
        self.preactivation = np.empty(self.n_units)
        self.activation = np.empty(self.n_units)

    def forward_step(self):
        for i in range(self.n_units):
            for j in range(self.input_units):
                self.preactivation[i] = self.input[j] * self.weights[j][i]
            self.preactivation[i] += self.bias[i]
            self.activation[i] = np.maximum(0, self.preactivation[i])
        return self.activation

    def backward_step(self): 
        
