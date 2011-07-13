from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


class PlotLine():
    
    def __init__(self, directory, file, numPlottedSoFar, totalToPlot):
        self.directory = directory
        self.file = file
        self.filename = directory + file
        # self.filename = filename
        
        self.mouseNumber = (file[9:])[:-4]
        print self.mouseNumber

        self.initialLine = 1
        self.normalizedLine = 2

        self.numPlottedSoFar = numPlottedSoFar

        self.totalToPlot = totalToPlot

        self.labels = {'6639':'GC/C1V1','6638':'GC/C1V1','6637':'GC/C1V1','6617':'GC','6619':'GC','433':'NULL','434':'NULL','432':'NULL','469':'YFP','6634':'GC','6635':'GC','6636':'GC/C1V1'}
 



        self.colors=[]
        self.colorsRGB=[(255,0,0),(227,66,52),(50,20,20),(255,153,102),(183,65,14),(255,168,18),(184,134,11),(141, 182,0),(3,192,60),(19,136,8),(0,123,167),(0,127,255),(0,35,102),(0,0,255),(105,53,156),(139,0,139)]
        
        for color in self.colorsRGB:
            self.colors.append((color[0]/256.0,color[1]/256.0,color[2]/256.0))

#        self.colors = ['Azure','Royal Blue','Blue','Purple Heart','Dark Magenta','Bright Pink']
                    
        print self.colors

        self.lineStyle = ['-','--']

        numStyles = size(self.lineStyle)
        whichStyle = self.numPlottedSoFar%numStyles

        self.plotArgs = dict(color=self.colors[self.numPlottedSoFar], marker=self.lineStyle[whichStyle])


        self.coords = [[0,0],[1000,0],[25000,0]]


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
        
        self.plot = plt.plot(self.x, self.y, **self.plotArgs) 

        #self.plot = plt.plot(self.x, self.y)
        plt.figlegend(self.plot, self.filename, 'upper right')
        


#To pick the points yourself
#        coords = ginput(3, timeout=0)
#        self.coords = [[0,0],[1000,0],[25000,0]]

        plt.figure(2)
        plt.hold(True)
#        plt.figure(self.normalizedLine)

        self.newPlot = PlotLine(self.directory, self.file, self.numPlottedSoFar, self.totalToPlot)
        self.newPlot.x = self.x
        self.newPlot.y = self.y
        self.newPlot.make_new_plot(self.coords)
       
        
        fig3 = plt.figure(3)
        plt.hold(True)
        
        
        ax = fig3.add_axes([0.1, (1.0/self.totalToPlot)*self.numPlottedSoFar, 0.85, 1.0/self.totalToPlot])

        yprops = dict(rotation=0,
                      horizontalalignment='right',
                      verticalalignment='center',
                      x=-0.01)


        ax.set_ylabel('*' + self.labels[self.mouseNumber[:-3]] + '*' +' \n(' + self.mouseNumber[:-3]+ ')' + '\n init: '+str(self.y[self.coords[1][0]]), **yprops)
#        setp(ax.get_yticklabels(), visible=False)
        plt.ylim(.76, 1.06)
        plt.yticks((.85,1))
        if(self.numPlottedSoFar > 1):
            setp(ax.get_xticklabels(), visible=False)


        #fig3.add_subplot(self.numPlottedSoFar, 1, 1)
        plt.hold(True)
        print "numPlottedSoFar", self.numPlottedSoFar
        self.newPlot.normalize()

        plt.draw()
        plt.figure(3)
        plt.hold(False)

        plt.figure(2)
        plt.hold(False)



    def make_new_plot(self, coords):
        x1 = int(self.coords[1][0])
#        y1 = int(self.coords[1][1])
        print 'x1',x1

        x2 = int(self.coords[2][0])
#       y2 = int(self.coords[2][1])
        print 'x2', x2
        
        self.xnew = self.x[0:max(x1, x2)-min(x1, x2)]
        self.ynew = self.y[min(x1,x2):max(x1, x2)]

        print size(self.xnew)
        print size(self.ynew)

#        self.plot = plt.cla()
        self.plot = plt.plot(self.xnew, self.ynew, **self.plotArgs)

    def normalize(self):
        self.yinit = int(self.ynew[0])
       
        print 'yinit',self.yinit

        self.ynew = array(self.ynew)/self.yinit
        print 'ynew[0]: ',self.ynew[0]

        self.plot = plt.plot(self.xnew, self.ynew, **self.plotArgs)

        #self.plot = plt.plot(self.xnew, self.ynew)

if __name__=="__main__":
    directory = raw_input('directory?: ')

    counter = 1

    for fileList in os.walk('/Users/kellyz/Documents/Code/python/Fiberkontrol/code/' + directory):


        files = fileList[2]

        initialPlots = []
        plotLabels = []
        
        numFiles = size(files)

    for file in files: 
        print file
        if file[-3:] == 'npz':

            if file == '#':
                break
            fullDirectory = '/Users/kellyz/Documents/Code/python/Fiberkontrol/code/' + directory + '/'

            filename = '/Users/kellyz/Documents/Code/python/Fiberkontrol/code/' + directory + '/' + file
            
            plot = PlotLine(fullDirectory, file, counter, numFiles)
            plot.make_plot()
           
            counter +=1
            initialPlots.append(plot.plot)
            plotLabels.append(plot.labels[plot.mouseNumber[:-3]]+'\n(' + plot.mouseNumber[:-3]+ ')') # + '\n init: ' +str(plot.y[plot.coords[1][0]]))
            
            
     
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
            
