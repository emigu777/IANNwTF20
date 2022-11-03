
#   2.1     Task 01 - Building your data set
#           You are going to need some data to train your network on. Use NumPy1 to:

#   1. Randomly generate 100 numbers between 0 and 1 and save them to an
#           array ’x’. These are your input values.

#   2. Create an array ’t’. For each entry x[i] in x, calculate x[i]**3-x[i]**2
#           and save the results to t[i]. These are your targets.

#   -----2.1.1-----
#   create array x
x = []

#   fill array x with 100 random numbers
import random
for i in range (0,100):
    x.append(random.random())

#   see if it works by printing array x
print(x)
#   seems to work :)

#   -----2.1.2-----
#   create array t
t = []

#   fill array by saving the results of x[i]**3-x[i]**2
for i in range (0,100):
    t.append(x[i]**3-x[i]**2)

#   print t to see if it works
print(t)
#   the results in t seem plausible to me, I guess it's correct
