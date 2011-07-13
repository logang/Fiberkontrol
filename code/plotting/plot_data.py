import numpy as np
import os


#enthough imports
from enthought.traits.ui.api import Item, View, Group, HGroup, spring, Handler


#chaco imports
from enthought.chaco.api import VPlotContainer, ArrayPlotData, Plot
from enthought.enable.component_editor import ComponentEditor
from enthought.chaco.tools.simple_zoom import SimpleZoom

from enthought.traits.api import HasTraits, Instance
from enthought.traits.ui.api import View, Item
from enthought.chaco.tools.api import RangeSelection

a = np.load('/Users/kellyz/Documents/Code/python/Fiberkontrol/code/20110624-controls.npz')['x']

b = np.load('/Users/kellyz/Documents/Code/python/Fiberkontrol/code/higher-sensitivity-detector-test.npz')['x']



dataA = a.tolist()[0][2]
if dataA.count([]) > 0:
    dataA.remove([])
dataB = b.tolist()[0][2]
if dataB.count([]) > 0:
    dataB.remove([])


class MyPlot(HasTraits):
    plot = Instance(VPlotContainer)
    traits_view = View(Item('plot', editor = ComponentEditor(), show_label = False), 
                       width = 500, height = 500, resizable = True, 
                       title = "Fluorescence signal")
    
    def __init__(self, x, y, *args, **kw):
        super(MyPlot, self).__init__(*args, **kw)
        plotdata = ArrayPlotData(x=x, y=y)
        plotA = Plot(plotdata)
        plotA.plot(("x", "y"), type = "line", color = "blue")
        plotA.title = "Fluorescence plotA"
        plotB = Plot(plotdata)
        plotB.plot(("x", "y"), type = "line", color = "green")
        plotB.title = "Fluorescence plotB"
        container = VPlotContainer(plotA, plotB)
        plotA.controller = RangeSelection(plotA)
        self.plot = container


#zoom_overlay = ZoomOverlay(source=plotA, destination=plotA)

#SimpleZoom.always_on = True

length = int(len(dataA))
#length = 30500
x = np.linspace(0,length, length)
y = dataA[:length]


lineplot = MyPlot(x, y)
lineplot.configure_traits()




def create_zoomed_plot():
    x = np.linspace(0,length, length)
    y = dataA[:length]
    

    main_plot = MyPlot(x, y)
    zoom_plot = MyPlot(x, y)
    outer_container = VPlotContainer(padding=30,
                                     fill_padding=True,
                                     spacing=50,
                                     stack_order='top_to_bottom',
                                     bgcolor='lightgray',
                                     use_backbuffer=True)
    
    outer_container.add(main_plot)
    outer_container.add(zoom_plot)
       
    main_plot.controller = RangeSelection(main_plot)
       
#    zoom_overlay = ZoomOverlay(source=main_plot, destination=zoom_plot)
    outer_container.overlays.append(zoom_overlay)
    
    return outer_container








if __name__ == "__main__":
   create_zoomed_plot()

