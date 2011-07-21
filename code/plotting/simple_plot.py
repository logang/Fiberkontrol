import matplotlib.pyplot as plt
import numpy as np
import os
import sys


file = sys.argv[1]

filename = '/Users/kellyz/Documents/Data/Fiberkontrol/' + file

print filename

a = np.load(filename)['x']

dataA = a.tolist()[0][2]
if dataA.count([]) > 0:
    dataA.remove([])

length = int(len(dataA))

x = np.linspace(0, length, length)
y = dataA[:length]
plt.plot(x, y)
plt.show()
