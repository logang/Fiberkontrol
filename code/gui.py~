# Major imports
import os
import numpy as np
import serial
import re
import time
import random
import wx

# Enthought imports
from enthought.traits.api \
    import Array, Dict, Enum, HasTraits, Str, Int, Range, Button, String, \
    Bool, Callable, Float, Instance, Trait
from enthought.traits.ui.api import Item, View, Group, HGroup, spring, Handler
from enthought.pyface.timer.api import Timer


# Chaco imports
from enthought.chaco.api import ArrayPlotData, Plot
from enthought.enable.component_editor import ComponentEditor
from enthought.chaco.chaco_plot_editor import ChacoPlotEditor, \
                                                ChacoPlotItem

#------------------------------------------------------------------------------

class FiberViewer( HasTraits ):
    """ This class contains the data arrays that will be updated
    by FiberController, and some control buttons.
    """

    # Public traits
    record        = Button()
    stop          = Button()
    load_data     = Button()
    save_data     = Button()
    save_plot     = Button()
    plot_type     = Enum( "line", "scatter" )

    # Private traits
    _xdata = Array
    _ydata = Array
    
    # View attribute defining how an instance of this class will
    # be displayed.
    view = View( ChacoPlotItem( "_xdata", "_ydata",
                               type_trait       = "plot_type",
                               resizable        = True,
                               x_label          = "Time",
                               y_label          = "Signal",
                               color            = "green",
                               bgcolor          = "white",
                               border_visible   = True,
                               border_width     = 1,
                               padding_bg_color = "lightgray",
                               width            = 800,
                               height           = 380,
                               show_label       = False),
                HGroup( spring,
                        Item( "record",    show_label = False), 
                        Item( "stop",      show_label = False ),
                        Item( "save_data", show_label = False ),
                        Item( "load_data", show_label = False ),
                        Item( "save_plot", show_label = False ),
                        spring ),
                resizable = True,
                buttons   = ["OK"],
                width = 1000, height = 500)

    def __init__( self ):
        self.recording = False
        self.stopped   = False
        self.saveit    = False
        self._ydata = np.zeros( 1 )
        self._xdata = np.zeros( 1 )        

    def save( self, path = None, name = None ):
        if path is not None:
            self.savepath = path
        if name is not None:
            self.outname = name
        np.savez( self.savepath + self.outname, self.output )

    def load( self, path = None, name = None ):
        if path is not None:
            self.savepath = path
        self.loaded = np.load( self.savepath + '/out.npz' )['arr_0']

    def _record_fired( self ):
        self.recording  = True
        self.start_time = time.clock()
        print "record fired"

    def _stop_fired( self ):
        self.recording = False
        self.stop_time = time.clock()
        self.stopped   = True
        print "stop fired"

    def _save_data_fired( self ):
        self.saveit = True

#------------------------------------------------------------------------------

class FiberController( HasTraits ):
    
    # A reference to the plot viewer object
    viewer = Instance( FiberViewer )
    
    # Some parameters controlling the data collected

    analog_in_pin = Int( 0 )
#    T             = Int( 0 ) 
    dt            = Int( 20 ) # in ms; what does this do to accuracy
    window        = Int( 1000 ) 
    savepath      = os.path.abspath('.')
    filename      = 'test'

    # The max number of data points to accumulate and show in the plot
    max_num_points = window 
    
    # The number of data points we have received
    num_ticks = Int(0)
    
    view = View(Group( 'analog_in_pin',
                       'dt',
                       'window',
                       'savepath',
                       'filename',
                       orientation="vertical"),
                buttons=["OK", "Cancel"],
                width = 500, height = 400 )
    
    def __init__(self, **kwtraits):
        super( FiberController, self ).__init__( **kwtraits )

        self.buffer    = ''
        self.rate      = 115200
        try:
            self.ser = serial.Serial( '/dev/tty.usbmodem641', self.rate )
            print "serial initialized"
        except:
            self.ser = serial.Serial( '/dev/tty.usbserial-A600eu7L', self.rate )                
            print "serial initialized"

        self.recording_it = 0
        self.data         = []
        self.alldata      = {}

    def receiving( self, dt = None ):
        """ Serial read function that fills a buffer for self.dt milliseconds,
        removes any zeros, and returns the median.
        """
        buffer = self.buffer
        out    = []

        if dt is not None:
            self.dt = dt

        buffer   = buffer + self.ser.read( self.ser.inWaiting() )

        if '\n' in buffer:
            lines = buffer.split( '\n' ) 
            
            if lines[-2]: 
                full_lines = lines[:-1] 
                 
                for i in range( len( full_lines ) ):
                    o = re.findall( r"\d+", lines[i] )
                    if o:
                        out.append( int( o[0] ) )

            self.buffer  = lines[1]
            out = np.median( out[~(out == 0)] )
        return out

    def timer_tick(self, *args):
        """ Callback function that should get called based on a wx timer
        tick.  This gets  a new datapoint from serial and adds it to
        the _ydata array of our viewer object.
        """
        if self.viewer.recording is True:           
            #---Generate a new number and increment the tick count 
            new_val = self.receiving()
            self.num_ticks += 1
        
            #---Grab the existing data, truncate it, and append the new point.
            cur_data  = self.viewer._ydata
            if len( cur_data ) > self.max_num_points + 2: 
                new_data  = np.hstack(( cur_data[ -self.max_num_points+1:], [new_val] ))
            else:
                #            print cur_data, new_val
                new_data  = np.hstack(( cur_data, new_val ))            
            new_index = self.dt * np.arange(self.num_ticks - len(new_data) + 1, self.num_ticks+0.01)
        
            self.data.append( new_val )
            print len( self.data )

            self.viewer._xdata = new_index
            self.viewer._ydata = new_data

        elif self.viewer.recording is False and self.viewer.stopped is True:
            #---Add data epoch to a list 
            self.alldata[ self.recording_it ] = ( self.viewer.start_time, self.viewer.stop_time, self.data )
            self.recording_it += 1
            self.data = []
            self.viewer.stopped = False

        elif self.viewer.recording is False and self.viewer.saveit is True:
            #---Listen to FiberViewer and save if save_data button fires
            print self.alldata
            np.savez( os.path.join( self.savepath, self.filename ), x = self.alldata ) 
            self.viewer.saveit = False

        return
            
#-----------wxApp used when this file is run from the command line------------#

class FiberRead( wx.PySimpleApp, HasTraits ):
    
#    recording = Bool( False )
    
    def OnInit( self, *args, **kw ):
        viewer     = FiberViewer()
        controller = FiberController( viewer = viewer )

        # Pop up the windows for the two objects
        viewer.edit_traits()
        controller.edit_traits()
        
        # Set up the timer and start it up
        self.setup_timer( controller )
        return True

    def setup_timer( self, controller ):
        # Create a new WX timer
        timerId    = wx.NewId()
        self.timer = wx.Timer( self, timerId )
        
        # Register a callback with the timer event
        self.Bind( wx.EVT_TIMER, controller.timer_tick, id = timerId )

        # Start up the timer!  We have to tell it how many milliseconds
        # to wait between timer events.  For now we will hardcode it
        # to be 20 ms, so we get 50 points per second.
        self.timer.Start( controller.dt, wx.TIMER_CONTINUOUS )
        return

# This is called when this invoked from the command line.
if __name__ == "__main__":
    fr = FiberRead()
    fr.MainLoop()    

# EOF
