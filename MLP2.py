import numpy as np
import matplotlib.pyplot as plt 
from Multilayer import Multilayerperceptron

#creating random input values
x = np.random.rand(100)
#array for target output 
t  = np.empty(100)
for i in range(len(x)):
    t[i] = x[i]**3-x[i]**2

loss = []

plt.plot(x, t, 'bo')
plt.show()


mlp = Multilayerperceptron(1, 2, [10, 1])

for i in range(1000):
    for input in x:
        predicted = mlp.forward_step[input]
        mlp.backpropagation((predicted - t[input]))
        loss.append(1/2 * np.square((predicted - t[input])))

lossarray = np.array(loss)
average = np.average(lossarray.reshape(-1, 100), axis = 1)
plt.plot(range(1000), average, color='green', marker='o')
plt.title("Training Progress")
plt.xlabel('epochs')
plt.ylabel('average loss')

