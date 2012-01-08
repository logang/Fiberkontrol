#! /usr/bin/env python

# Major imports
import os, serial, re, time, random, wx
import numpy as np

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

#------------------------------------------------------------------------------#

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
        self.start_time = time.time()
        print "record fired"

    def _stop_fired( self ):
        self.recording = False
        self.stop_time = time.time()
        self.stopped   = True
        print "stop fired"

    def _save_data_fired( self ):
        self.saveit = True

#------------------------------------------------------------------------------#

class FiberController( HasTraits ):
    
    # A reference to the plot viewer object
    viewer = Instance( FiberViewer )
    
    # Some parameters controlling the data collected

    analog_in_pin = Int( 0 )
#    T             = Int( 0 ) 
    dt            = Int( 10 ) # in ms; what does this do to accuracy
    window        = Int( 1000 ) 
    #savepath      = os.path.abspath('.')
    savepath = '/Users/kellyz/Documents/Data/Fiberkontrol/20120108'
    filename      = '20120108-'
    time_until_stim = 10.0
    time_of_stim = 10.0
    time_between_stim = 0.0
    stim_frequency = 0.0

    # The max number of data points to accumulate and show in the plot
    max_num_points = window 
    
    # The number of data points we have received
    num_ticks = Int(0)


    
    view = View(Group( 'analog_in_pin',
                       'dt',
                       'window',
                       'savepath',
                       'filename',
                       'time_until_stim',
                       'time_of_stim',
                       'time_between_stim',
                       'stim_frequency',
                       orientation="vertical"),
                buttons=["OK", "Cancel"],
                width = 500, height = 400 )
    
    def __init__(self, **kwtraits):
        super( FiberController, self ).__init__( **kwtraits )

        self.buffer    = ''
        self.rate      = 115200
        try:
#Un comment this for the duemilanove
#            self.ser = serial.Serial( '/dev/tty.usbmodem641', self.rate )
            self.ser = serial.Serial( '/dev/tty.usbmodem411', self.rate )
            print "serial initialized"
        except:
            self.ser = serial.Serial( '/dev/tty.usbserial-A600eu7L', self.rate )                
            print "serial initialized"

        self.recording_it   = 0
        self.data           = []
        self.alldata        = {}
        self.shutterData    = []
        self.allShutterData = {}
        self.timeData       = []
        self.allTimeData    = {}
    

        self.timeData.append(0.0)

       # self.shutterPlan = np.zeros(1) this is just self.stimProtocol
        self.current_time = time.time() 

# CHANGE THIS TO CONTROL THE FREQUENCIES
        #The units are the period of the ON portion of a firing pulse |--|__ in ms
        # i.e. period = 10 yields a frequency of 1000/2*period = 50 Hz
        # right now the minimum period is 2 ms or 250 Hz
       
        self.periodPlan = [0, 9999, 0, 9999, 0, 1000, 0,
                           100, 0, 30, 0, 20, 0,
                           10, 0, 10, 0, 20, 0, 30, 0,
                           100, 0, 1000, 0, 9999, 0, 9999, 0, 0]




        self.periodPlan = [0, 999, 0, 100, 0, 50, 0, 
                           40, 0, 30, 0, 20, 0, 10, 0, 5, 0, 3, 0]
                         #  5, 0, 3, 0, 10, 0, 20, 0, 30, 0,
                         #  40, 0, 50, 0, 100, 0, 999, 0, 0]

        self.periodPlan = [0, 10, 0, 10, 0, 20, 0,
                            50, 0, 50, 0, 100, 0]

        self.periodPlan = [0, 9, 0, 10, 0, 11, 0,
                           11, 0, 10, 0, 9, 0]

        self.periodPlan = [0, 900, 0, 900, 0, 900, 0,
                           900, 0, 900, 0, 900, 0]

    def receiving( self, dt = None ):
        """ 
        Serial read function that fills a buffer for self.dt milliseconds,
        removes any zeros, and returns the median.
        """
        buffer = self.buffer
        out    = []

        if dt is not None:
            self.dt = dt
        
        time.sleep(self.dt/1000)

        print 'numWaiting', self.ser.inWaiting()
        buffer = buffer + self.ser.read( self.ser.inWaiting() )


        if '\n' in buffer:
            lines = buffer.split( '\n' ) 
            
            print 'lines', lines
            if lines[-2]: 
                full_lines = lines[:-1] 
                 
                for i in range( len( full_lines ) ):
                    o = re.findall( r"\d+", lines[i] )
                    if o:
                        out.append( int( o[0] ) )

            self.buffer  = lines[1]
            out = np.median( out[~(out == 0)] )
        print 'buffer', buffer
        print 'size', len(buffer)
        print 'out',out
        return out

# going through all chunks __|--|__|--|__|--|__
# now you are in a chunk: |--|___


    def ToNDigString(self, period, n):
        period_str = str(period)
        for i in range(n-len(period_str)):
            period_str = "0" + period_str

        return period_str

    def control_shutter_state(self, time_at_call):
        self.time_until_stim = float(self.time_until_stim)
        self.time_of_stim = float(self.time_of_stim)
        self.time_between_stim = float(self.time_between_stim)
        self.stim_frequency = float(self.stim_frequency)

        
        time_at_call = time_at_call - self.viewer.start_time

        period_of_whole_chunk = self.time_of_stim + self.time_between_stim
        period_of_whole_fire = period_of_whole_chunk
        period = int(period_of_whole_fire/2)

        frequency = 1000.0/float(period_of_whole_fire)

        if(time_at_call <= self.time_until_stim):
            self.ser.write("000")
            self.shutterData.append(0)

        if(time_at_call >  self.time_until_stim):
            chunkNumber = int(np.ceil(((time_at_call - self.time_until_stim)/period_of_whole_chunk)%len(self.periodPlan)))
            
           
            period = self.periodPlan[int(chunkNumber)]

            period_of_whole_fire = 2*period
            self.ser.write(self.ToNDigString(period, 4))
              
            if period==0:
                self.shutterData.append(0)
                print "frequency 0"
            elif (((time_at_call - self.time_until_stim)%period_of_whole_chunk)%period_of_whole_fire) < self.time_of_stim:
                frequency = 1000.0/float(period_of_whole_fire)
                print "frequency", frequency
                self.shutterData.append(frequency)
            else:
                self.shutterData.append(0)
                print "frequency 0"


    def timer_tick(self, *args):
        """ Callback function that should get called based on a wx timer
        tick.  This gets  a new datapoint from serial and adds it to
        the _ydata array of our viewer object.
        """


        if self.viewer.recording is True:           
            #---Generate a new number and increment the tick count 
            new_val = self.receiving()
            print 'new_val', new_val

            if self.num_ticks == 0:
                print "Init!"
               # self.initShutterPlan()


            self.num_ticks += 1

            #---Grab the existing data, truncate it, and append the new point.
            cur_data  = self.viewer._ydata
            if len( cur_data ) > self.max_num_points + 2: 
                new_data  = np.hstack(( cur_data[ -self.max_num_points+1:], [new_val] ))
            else:
                #            print cur_data, new_val
                new_data  = np.hstack(( cur_data, new_val ))            
            new_index = self.dt * np.arange(self.num_ticks - len(new_data) + 1, self.num_ticks+0.01)
#            print "new_index_old", new_index



            #-----determine the actual time, and timestamp the data to it
            if self.num_ticks == 1:
                self.prev_time = self.viewer.start_time
                self.current_time = self.viewer.start_time + self.dt/1000.0
            else:
                self.prev_time = self.current_time
                self.current_time = time.time()

            self.timeData.append( self.current_time - self.viewer.start_time)
            new_index = np.array(self.timeData[self.num_ticks - len(new_data) + 1:self.num_ticks + 1])


            self.data.append( new_val )
            self.viewer._xdata = new_index
            self.viewer._ydata = new_data
            
            #-----control the shutter frequency run by the arduino
            self.control_shutter_state(self.current_time);
           
        elif self.viewer.recording is False and self.viewer.stopped is True:
            #---Add data epoch to a list 
            self.alldata[ self.recording_it ] = ( self.viewer.start_time, self.viewer.stop_time, self.data )
            self.allShutterData [ self.recording_it ] = ( self.viewer.start_time, self.viewer.stop_time, self.shutterData )
            self.allTimeData [ self.recording_it ] = ( self.viewer.start_time, self.viewer.stop_time, self.timeData )
            self.recording_it += 1
            self.data = []
            self.shutterData = []
            self.viewer.stopped = False

        elif self.viewer.recording is False and self.viewer.saveit is True:


            #---Listen to FiberViewer and save if save_data button fires
            np.savez( os.path.join( self.savepath, self.filename ), x = self.alldata ) 
            
            shutter_filename = self.filename + '_s'
            np.savez( os.path.join( self.savepath, shutter_filename), x = self.allShutterData ) 

            time_filename = self.filename + '_t'
            np.savez( os.path.join( self.savepath, time_filename), x = self.allTimeData )

            self.viewer.saveit = False
            self.ser.close()
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
