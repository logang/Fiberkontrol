import matplotlib.pyplot as plt
import numpy as np
import os
import sys

file = sys.argv[1]
start = int(sys.argv[2])
end = int(sys.argv[3])



#filename = '/Users/kellyz/Documents/Code/python/Fiberkontrol/code/' + file
filename = '/Users/kellyz/Documents/Code/python/Fiberkontrol/code/higher-sensitivity-detector-test.npz'


print filename

a = np.load(filename)['x']

dataA = a.tolist()[0][2]
if dataA.count([]) > 0:
    dataA.remove([])

length = int(len(dataA))

#just a test to see if histogram works

trial = dataA[start:end]
x = range(start, end)


fig1 = plt.figure(1)
lineTrace = plt.subplot(211)
plt.plot(x, trial)

axes = plt.axis()
plt.show()

newAxes = fig1.ginput(n=1, timeout=20, show_clicks=True)

print newAxes

plt.subplot(212)
histogram = plt.hist(trial, bins=1000, range=None, normed=False, cumulative=False, 
                     bottom=None, histtype='bar', align='mid',
                     orientation='vertical', rwidth=None, log=False)
plt.show()

