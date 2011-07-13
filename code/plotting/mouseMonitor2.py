from pylab import *
import sys
class MouseMonitor:
    event = None
    xdatalist = []
    ydatalist = []

    def my_call(self, event):
        self.event = event

        if event.button == 1:
            self.xdatalist.append(event.xdata)
            self.ydatalist.append(event.ydata)

            sys.stdout.flush()
            print 'x = %s and y = %s' % (event.xdata,event.ydata)
            sys.stdout.flush()
            ax = gca()
            ax.hold(True)
            ax.plot([event.xdata],[event.ydata],'ro')

            draw()

        if event.button == 3:
            disconnect(self.cid)
            self.ask_for_another()

    def ask_for_another(self):
        sys.stdout.flush()
        answer = raw_input('Press n for the next plot, or any other key to stop:  ')
        if answer == 'n':
            self.plot_next()
        else:
            print '::sniff::  bye'
            close('all')

    def set_data(self,x,y):
        self.xlist = x
        self.ylist = y
        self.i = 0

    def plot_next(self):
        if self.i < len(self.xlist):
            cla()
            plot(xlist[self.i],ylist[self.i])
            self.i += 1
            self.connect_to_plot()
            draw()
        else:
            sys.stdout.flush()
            print 'all out of data.'

    def connect_to_plot(self):
        self.cid = connect('button_press_event',self.my_call)

if __name__ == "__main__":
    mouse = MouseMonitor()

    xlist = [[1,2,3],[4,5,6]]
    ylist = [[3,4,5],[8,9,7]]

    mouse.set_data(xlist,ylist)

    mouse.plot_next()

    show()
