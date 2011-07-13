from pylab import *
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import os
import sys
from scipy.optimize import leastsq



def GetTheme(mouseType):
    if 'C1V1' in mouseType:
            return 0
    elif 'GC' in mouseType:
            return 3
    elif 'YFP' in mouseType:
            return 2
    else:
            return 1


class PlotLine():
    
    def __init__(self, directory, file, numPlottedSoFar, numInThemesSoFar, totalToPlot, labels, totalOfThemes):
        self.directory = directory
        self.file = file
        self.filename = directory + file
        self.labels = labels
        self.numInThemesSoFar = numInThemesSoFar
        self.totalOfThemes = totalOfThemes
        self.numPlottedSoFar = numPlottedSoFar
        self.totalToPlot = totalToPlot

        self.dateOfTest = file[:8]
        self.mouseNumber = (file[9:])[:-4]
        self.mouseType = self.labels[self.mouseNumber[:-3]];
        self.theme = GetTheme(self.mouseType)


#-----Set this range to be the range of data you want to include-------#
#-----------can use an if statement to use different ranges for different datasets-----#
        self.coords = [[0,0],[1000,0],[20000,0]]



#-----Setup the color themes-----#
        self.colorsRGB=[[(255,0,0),(227,66,52), (164, 0, 0), (128, 0, 0), (179, 27, 27), (227, 66, 52),(183,65,14)], #red
                        [(255, 56, 0), (226, 88, 34), (255, 167, 0), (255, 117, 24), (242, 133, 0),(255, 179, 71), (244, 196, 48)], #orange
                        [(178,236,30),(80,200,120),(3,192,60),(119,221,119),(0,128,0),(19,136,8),(23,114,69),(178, 236, 23)], #green
                        [(0,123,167),(0,127,255),(0,35,102),(0,0,255),(105,53,156),(139,0,139), (141, 182, 0)]] #blue        


        self.colors= []
        for theme in range(len(self.colorsRGB)):
            self.colors.append([])
            counter = 0
            for color in self.colorsRGB[theme]:
                self.colors[theme].append((color[0]/256.0,color[1]/256.0,color[2]/256.0))
                counter += 1
                
                #---If you want to plot out all the colors in order----#
                colorTest=False
                if colorTest:
                    plt.figure(100)
                    plt.hold(True)
                    x=arange(0,1,.01)
                    plt.plot(x, x+counter, color=self.colors[theme][i])
                    plt.hold(False)
                    plt.figure(1)


        self.lineStyle = ['-','--']
        numStyles = size(self.lineStyle)
        whichStyle = self.numPlottedSoFar%numStyles


#----To be used as style kwargs in plt.plot----#
        self.plotArgs = dict(color=self.colors[self.theme][self.numInThemesSoFar[self.theme]], marker=self.lineStyle[whichStyle])



    def get_coords(self, numClicks):

        a = self.plot.ginput(n=1, timeout=10, show_clicks=True,
               mouse_add=1)
        
        return a



    def load_data(self, filename):
        a = np.load(filename)['x']

        dataA = a.tolist()[0][2]
        if dataA.count([]) > 0:
            dataA.remove([])

        return dataA


#####Displays all plots made by PlotLine
#---------------------------------------
#----Including full range line plot,
#----restricted range line plot,
#----normalized side by side,
#----flattened side by side,
#----unflattened histogram,
#----flattened histogram


    def make_plot(self):
        print "make_plot"

        dataA = self.load_data(self.filename)

        length = int(len(dataA))

        self.x = np.linspace(0, length, length)
        self.y = dataA[:length]

#----Plot all data on same plot---#
        plt.figure(1)
        
        self.plot = plt.plot(self.x, self.y, **self.plotArgs) 
        plt.figlegend(self.plot, self.filename, 'upper right')
        

#        coords = ginput(3, timeout=0)



#----Plot restricted range of data (defined by coords)--#
        plt.figure(2)
        plt.hold(True)

        self.newPlot = PlotLine(self.directory, self.file, self.numPlottedSoFar, 
                                self.numInThemesSoFar, self.totalToPlot, self.labels, self.totalOfThemes)
        self.newPlot.x = self.x
        self.newPlot.y = self.y
        self.newPlot.make_new_plot(self.coords)
       
        self.flatten()





        fig3 = plt.figure(3)
        plt.hold(True)
        
        self.side_by_side_line(fig3)
        self.newPlot.normalize(self.newPlot.xnew, self.newPlot.ynew)
        self.plot = plt.plot(self.newPlot.xnew, self.newPlot.ynew, **self.plotArgs)

        plt.figure(3)
        plt.hold(False)

        plt.figure(2)
        plt.hold(False)


        fig4 = plt.figure(4)
        plt.hold(True)
        
        self.side_by_side_line(fig4)
                
        self.ynormFlat = array(self.yflat)/self.newPlot.yinit

        self.plot = plt.plot(self.newPlot.xnew, self.ynormFlat, **self.plotArgs)
        plt.hold(False)



        fig5 = plt.figure(5+self.theme)
        plt.hold(True)
        if self.numInThemesSoFar[self.theme] == 1:

            setp(fig5, frameon=False)
            setp(fig5.get_axes(), visible=False)


        axNew = fig5.add_subplot(self.totalOfThemes[self.theme],1,self.numInThemesSoFar[self.theme])

        self.newPlot.plot_hist(axNew, self.mouseLabel)
        plt.draw()
        plt.hold(False)

    def side_by_side_line(self, fig):
        ax = fig.add_axes([0.1, (1.0/self.totalToPlot)*self.numPlottedSoFar, 0.85, 1.0/self.totalToPlot])

        self.newPlot.yprops = dict(rotation=0,
                      horizontalalignment='right',
                      verticalalignment='center',
                      x=-0.01)

        self.mouseLabel = '*' + self.labels[self.mouseNumber[:-3]] + '*' +' \n(' + self.mouseNumber[:-3] + ', '+str(self.y[self.coords[1][0]]) +')'

        ax.set_ylabel(self.mouseLabel, **self.newPlot.yprops)
        plt.ylim(.76, 1.06)
        plt.yticks((.85,1))
        if(self.numPlottedSoFar > 1):
            setp(ax.get_xticklabels(), visible=False)


        #fig3.add_subplot(self.numPlottedSoFar, 1, 1)                                       
        plt.hold(True)
        print "numPlottedSoFar", self.numPlottedSoFar



    def flatten(self):
        fp = lambda a, x: a[0]*np.exp(a[1]*np.exp(a[2]*x))

        error = lambda a, x, y: (fp(a,x)-y)

        a0 = [self.newPlot.ynew[0], -1.0, -1.0]
        a, success = leastsq(error, a0, args=(self.newPlot.xnew, self.newPlot.ynew), maxfev=2000)

        x = self.newPlot.xnew
        self.yfit = fp(a, x)
        print 'Estimator parameters: ', a

        self.plot = plt.plot(x, self.yfit, color='k')

        dIntensity = fp(a, x[0])-fp(a,x)
        self.yflat = self.newPlot.ynew + dIntensity
        

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

        self.plot = plt.plot(self.xnew, self.ynew, **self.plotArgs)


    def normalize(self, x, y):
        self.yinit = int(y[0])
       
        print 'yinit',self.yinit

        self.ynew = array(y)/self.yinit
        print 'ynew[0]: ',y[0]


        #self.plot = plt.plot(x, y)


    def plot_hist(self, ax, mouseLabel):
        self.plot = ax.hist(self.ynew, bins=1000, range = (0.85, 1.1), histtype='bar')

#        ax.set_title(mouseLabel)
        plt.xticks((0.9,0.95,1,1.05))
        ax.yaxis.set_major_locator(MaxNLocator(3))

        ax.set_ylabel(mouseLabel, **self.yprops)
       # setp(ax.get_yticklabels(), visible=False) 
       # plt.ylim(.76, 1.06)
       # plt.yticks((.85,1))
        
        if(self.numInThemesSoFar[self.theme] < self.totalOfThemes[self.theme]):
            setp(ax, axisbelow=False)
            setp(ax.get_xticklabels(), visible=False)

if __name__=="__main__":
    directory = raw_input('directory?: ')

    counter = 1

    for fileList in os.walk('/Users/kellyz/Documents/Code/python/Fiberkontrol/code/' + directory):


        files = fileList[2]

        initialPlots = []
        plotLabels = []
        
        numFiles = size(files)

        numInThemes = [0,0,0,0]
        totalOfThemes = [0,0,0,0]

        labels = {'6639':'GC/C1V1','6638':'GC/C1V1','6637':'GC/C1V1','6617':'GC','6619':'GC','433':'NULL','434':'NULL','432':'NULL','469':'YFP','6634':'GC','6635':'GC','6636':'GC/C1V1', '6616':'YFP', '6615':'ChR2-YFP?','442':'YFP','121':'YFP','441':'Optod1-YFP','6623':'YFP','177':'YFP'}           

    for file in files:
        if file[-3:] == 'npz':
            mouseType = labels[file[9:-7]]
            theme = GetTheme(mouseType)
            totalOfThemes[theme] += 1

    for file in files: 
        print file
        if file[-3:] == 'npz':

            if file == '#':
                break
            fullDirectory = '/Users/kellyz/Documents/Code/python/Fiberkontrol/code/' + directory + '/'

            filename = '/Users/kellyz/Documents/Code/python/Fiberkontrol/code/' + directory + '/' + file
            

            print file
            mouseType = labels[file[9:-7]];

            theme = GetTheme(mouseType)
           
            numInThemes[theme] += 1

            plot = PlotLine(fullDirectory, file, counter, numInThemes, numFiles, labels, totalOfThemes)
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
            
