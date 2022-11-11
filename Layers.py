import numpy as np

class Layer:

    def __init__(self, n_units, input_units):       #constructor
        self.n_units = int(n_units)                 #number of units in layer
        self.input_units = int(input_units)         #number of units in preceding layer
        self.bias = np.zeros(self.n_units)          #array filled with zeros as biases
        self.weights = np.random.uniform(-10, 10, size=( self.input_units, self.n_units))   #matrix filled with random floats between -10 and 10 as weights
        self.input = np.empty(self.input_units)     #empty array for inputs in the layer
        self.preactivation = np.empty(self.n_units) #empty array for preactivation values
        self.activation = np.empty(self.n_units)    #empty array for activation values

        self.error_signal_n = np.empty(self.input_units)  #array to safe dLbyda(n) the signal which the layer gets from the layer n+1
        self.error_signal_for_n_minus_one = np.empty(self.input_units) #array to safe dLbyda(n-1) the signal will be given to layer n-1
        self.dLbydb = np.empty(self.n_units)            #array to safe dL/db(n)
        self.dLbydW = np.empty((self.input_units, self.n_units)) #dLbydW(n) matrix for the gradients of the weights

    def forward_step(self, input):
        self.input = input
        self.preactivation = np.dot(self.input, self.weights) #multiply weights matrix with input array and safe results to preactivation array
        #self.preactivation = self.preactivation.ravel()
        print(self.preactivation)
        
        for i in range(self.n_units):
            self.preactivation[i] += self.bias[i]              #add biases to each preactivation to get final preactivation
            self.activation[i] = np.maximum(0, self.preactivation[i])   #ReLu (preactivation) = activation
        return self.activation

    def backward_step(self, error_signal, n):          #n is the learning rate
        self.error_signal = error_signal               #the error signal is the gradient dL/da which the layer gets from the layer n+1
        for i in range(self.n_units):
            if self.preactivation[i] <= 0:
                self.dLbydb[i] = 0                     #ReLu' becomes zero when the preactivation is <= 0
            else:
                self.dLbydb[i] = self.error_signal[i]

        # dd(n)/da(n-1) = weights are multiplied with dL/dd(n) to get dL/da(n-1) which is the new error signal given to layer n-1
        self.error_signal_for_n_minus_one = np.dot(self.weights, self.dLbydb)

        #dL/dW(n) is calculated by dL/dd * input
        self.dLbydW = np.atleast_2d(self.input).T * self.dLbydb

        #updating the parameters with θnew = θold − η∇θL
        for i in range(self.n_units):
            self.bias[i] -= n * self.dLbydb[i]
            for j in range(self.input_units):
                self.weights[j][i] -= n * self.dLbydW[j][i]

        return(self.error_signal_for_n_minus_one)


