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

        self.a1 = np.empty(self.n_units)            #a1 array to safe σ(preactivation) ◦dL/dactivation
        self.dLbydW = np.empty(size = (self.input_units, self.n_units)) #dLbydW matrix for the gradients of the weights
        self.lastlayer = False                      #our layer is not the last one unless we say self.lastlayer = True

    def forward_step(self, input):
        self.input = input
        self.preactivation = np.dot(self.input, self.weights)  #multiply weights matrix with input array and safe results to preactivation array
        for i in range(self.n_units):
            self.preactivation[i] += self.bias[i]              #add biases to each preactivation to get final preactivation
            self.activation[i] = np.maximum(0, self.preactivation[i])   #ReLu (preactivation) = activation
        return self.activation

    def backward_step(self, target, ):
        #we call "σ(preactivation) ◦dL/dactivation" "a1" and calculate it:
        for i in range(self.n_units):
            if self.preactivation[i] <= 0:      #ReLu'(preactivation) becomes zero for smaller than zero and one if else
                self.a1[i] = 0
            else:
                self.a1[i] = self.activation[i] - target[i] #activation - target is dL/dactivation, the derivative of the loss function with regard to the activation
        #return self.a1
        #calculate dL/dW
        for i in range(self.input_units)





l1 = layer(4, 5)
print(l1.bias)
print(l1.weights)
l1.forward_step([1, 1, 0, 0, 0])
print(l1.input)
print(l1.preactivation)
print(l1.activation)
l1.backward_step([1, 1, 1, 1, 2])
print(l1.a1)