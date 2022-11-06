import numpy as np
import matplotlib.pyplot as plt 
import Layers

#creating random input values
x = np.random.rand(100)
#array for target output 
t  = np.empty(100)
for i in range(len(x)):
    t[i] = x[i]**3-x[i]**2

plt.plot(x, t, 'bo')
plt.show()


