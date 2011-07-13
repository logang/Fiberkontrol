from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


class PlotLine():
    
    def __init__(self, filename):
        self.filename = filename

    def get_coords(self, numClicks):
        print 'hello'
        a = self.plot.ginput(n=1, timeout=10, show_clicks=True,
               mouse_add=1)
        print a
        return a


    def make_plot(self):
        print "make_plot"
        a = np.load(self.filename)['x']

        dataA = a.tolist()[0][2]
        if dataA.count([]) > 0:
            dataA.remove([])

        length = int(len(dataA))

        self.x = np.linspace(0, length, length)
        self.y = dataA[:length]


        self.plot = plt.plot(self.x, self.y)

    def make_new_plot(self, coords):
        x1 = int(coords[1][0])
        y1 = int(coords[1][1])

        x2 = int(coords[2][0])
        y2 = int(coords[2][1])

        
        self.xnew = self.x[0:max(x1, x2)-min(x1, x2)]
        self.ynew = self.y[min(x1,x2):max(x1, x2)]

        self.plot = plt.cla()
        self.plot = plt.plot(self.xnew, self.ynew)

    def normalize(self):
        self.yinit = int(self.ynew[0])
       
        self.ynew = array(self.ynew)/self.yinit
       

        self.plot = plt.cla()
        self.plot = plt.plot(self.xnew, self.ynew)

if __name__=="__main__":
    directory = raw_input('directory?: ')

    counter = 1
    while(True):
        
        file = raw_input('filename?: ')
        if file == '#':
            break
        filename = '/Users/kellyz/Documents/Code/python/Fiberkontrol/code/' + directory +'/' + directory + '-' + file + '.01.npz'

        plot = PlotLine(filename)
        plot.make_plot()
        counter +=1

    coords = ginput(3, timeout=0)

    plot.make_new_plot(coords)
    plot.normalize()
    print "clicked",coords
    show()

