import tkFileDialog
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


if __name__ == '__main__':

#    directory ='/Users/kellyz/Documents/Data/Fiberkontrol/'
#    file = sys.argv[1]
#    dataFilename = directory + file
    dataFilename = tkFileDialog.askopenfilename()

    d = np.load(dataFilename)
    print d.files
    data = d['data']
    fluor = data[:,0]
    t = d['time_stamps']
    licks = data[:,3]
    len = min(np.size(fluor), np.size(t))
    t = t[0:len]
    fluor = fluor[0:len]
    licks = licks[0:len]
    print np.shape(fluor)
    print np.shape(t)
    plt.plot(t, 1.7*licks, color='r', alpha=0.5)
    plt.plot(t, (fluor-np.mean(fluor))/np.mean(fluor), color='b')
    plt.show()

