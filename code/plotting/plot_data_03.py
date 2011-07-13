from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


class PlotLine():
    
    def __init__(self, directory, file):
        self.directory = directory
        self.file = file
        self.filename = directory + file
        # self.filename = filename
        
        self.mouseNumber = (file[9:])[:-4]
        print self.mouseNumber

        self.initialLine = 1
        self.normalizedLine = 2

    def get_coords(self, numClicks):
        print 'hello'
        a = self.plot.ginput(n=1, timeout=10, show_clicks=True,
               mouse_add=1)
        
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

        plt.figure(1)
#        plt.figure(self.initialLine)
        
        self.plot = plt.plot(self.x, self.y)
        plt.figlegend(self.plot, self.filename, 'upper right')
        


#To pick the points yourself
#        coords = ginput(3, timeout=0)
        coords = [[0,0],[1000,0],[15000,0]]

        plt.figure(2)
        plt.hold(True)
#        plt.figure(self.normalizedLine)

        self.newPlot = PlotLine(self.directory, self.file)
        self.newPlot.x = self.x
        self.newPlot.y = self.y
        self.newPlot.make_new_plot(coords)
       
        plt.figure(3)
        plt.hold(True)
        self.newPlot.normalize()

        plt.draw()
        plt.hold(False)

        plt.figure(2)
        plt.hold(False)



    def make_new_plot(self, coords):
        x1 = int(coords[1][0])
#        y1 = int(coords[1][1])
        print 'x1',x1

        x2 = int(coords[2][0])
#       y2 = int(coords[2][1])
        print 'x2', x2
        
        self.xnew = self.x[0:max(x1, x2)-min(x1, x2)]
        self.ynew = self.y[min(x1,x2):max(x1, x2)]

        print size(self.xnew)
        print size(self.ynew)

#        self.plot = plt.cla()
        self.plot = plt.plot(self.xnew, self.ynew)

    def normalize(self):
        self.yinit = int(self.ynew[0])
       
        print 'yinit',self.yinit

        self.ynew = array(self.ynew)/self.yinit
        print 'ynew[0]: ',self.ynew[0]

#        self.plot = plt.cla()
        self.plot = plt.plot(self.xnew, self.ynew)

if __name__=="__main__":
    directory = raw_input('directory?: ')

    counter = 1

    for fileList in os.walk('/Users/kellyz/Documents/Code/python/Fiberkontrol/code/' + directory):


        files = fileList[2]

        initialPlots = []
        plotLabels = []

    for file in files: 
        print file
        if file[-3:] == 'npz':

            if file == '#':
                break
            fullDirectory = '/Users/kellyz/Documents/Code/python/Fiberkontrol/code/' + directory + '/'

            filename = '/Users/kellyz/Documents/Code/python/Fiberkontrol/code/' + directory + '/' + file
            
            plot = PlotLine(fullDirectory, file)
            plot.make_plot()
           
            counter +=1
            initialPlots.append(plot.plot)
            plotLabels.append(plot.mouseNumber)
            
            
     
    print initialPlots
    print plotLabels
    plt.figure(1)
    plt.figlegend(initialPlots, plotLabels, 'upper right')
    plt.figure(2)
    plt.figlegend(initialPlots, plotLabels, 'upper right')
    
#    coords = ginput(3, timeout=0)
            
#    plot.make_new_plot(coords)
#    plot.normalize()
#    print "clicked",coords
    plt.figure(1)
    plt.draw()
    plt.figure(2)
    plt.draw()
    plt.show()
            
