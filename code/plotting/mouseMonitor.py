from pylab import *
class MouseMonitor():
    def __init__(self, numClicks=1, color='ro'):
        self.event = None
        self.xdatalist = []
        self.ydatalist = []
        self.numClicks = numClicks
        self.color = color

    def mycall(self, event):
        if(self.numClicks>0):
            self.event = event
            self.xdatalist.append(event.xdata)
            self.ydatalist.append(event.ydata)
            
           # print 'x = %s and y = %s' % (event.xdata,event.ydata)
            
            ax = gca()  # get current axis
            ax.hold(True) # overlay plots.
            
        # Plot a red circle where you clicked.
            ax.plot([event.xdata],[event.ydata],self.color)
            
            draw()  # to refresh the plot.
            print self.numClicks
            self.numClicks -= 1
            if(self.numClicks == 0):
                return self.xdatalist


if __name__=="__main__":
    # Example usage
    mouse = MouseMonitor(2, 'go')
    a=connect('button_press_event', mouse.mycall)
    print a
    plot([1,2,3])
    show()
    mouse = MouseMonitor(3, 'go')
    connect('button_press_event', mouse.mycall)
    print "hello"
    
