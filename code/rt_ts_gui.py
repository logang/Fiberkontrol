# Major imports
from numpy import linspace, pi
import numpy as np
import serial
import re
import time

# Enthought imports
from enthought.traits.api \
    import Array, Dict, Enum, HasTraits, Str, Int, Range, Button, String
from enthought.traits.ui.api import Item, View, HGroup, spring
from enthought.traits.api import Instance

# Chaco imports
from enthought.chaco.api import ArrayPlotData, Plot
from enthought.enable.component_editor import ComponentEditor
from enthought.chaco.chaco_plot_editor import ChacoPlotEditor, \
                                                ChacoPlotItem

#------------------------------------------------------------------------------------------

class FiberPlot( HasTraits ):

    # Public Traits
    plot          = Instance( Plot )
    plot_type     = Enum( "record", "demo" )
    analog_in_pin = Int( 0 )
    record        = Button()
    stop          = Button()
    load_data     = Button()
    save_data     = Button()
    save_plot     = Button()
    T             = Range( 0., 1e10 ) # Do we need a stop time? Stop button instead?
    dt            = Int( 20 ) # in ms; what does this do to accuracy
    band_low      = Range( 0., 1024 )
    band_high     = Range( 0., 1024 )
    output        = Array
    plotdata      = ArrayPlotData()

    # Private Traits
    #_ydata  = Array
    #_xdata  = Array

    # Default TraitsUI view
    traits_view = View(
        Item('plot', editor=ComponentEditor(), show_label=False),
        # Items
        HGroup( spring,
                Item( "record",    show_label = False ), spring,
                Item( "stop",      show_label = False ), spring,
                Item( "load_data", show_label = False ), spring,
                Item( "save_data", show_label = False ), spring,
                Item( "save_plot", show_label = False ), spring,
                Item( "plot_type" ),                     spring,
                Item( "analog_in_pin" ),                 spring,
                Item( "dt" ),                            spring ),

        Item( "T" ),
        HGroup( Item( "band_high" ),
                Item( "band_low" ) ),
        # GUI window
        resizable = True,
        width     = 1000, 
        height    = 700 ,
        kind      = 'live' )

    def __init__(self, **kwtraits):
        super( FiberPlot, self ).__init__( **kwtraits )

#        self.plotdata = ArrayPlotData()
        self.plotdata.set_data( "y", np.array(()) ) 
        self.plotdata.set_data( "x", range( len( self.plotdata.get_data("y")  ) ) )

        plot = Plot( self.plotdata )
        plot.plot(("x", "y"), type="line", color="green")
        self.plot = plot

        self.band_low  = 0
        self.band_high = 1024
#        self.T          = 1000000

        if self.plot_type is "record":
            self.rate      = 115200
            self.recording = False
            try:
                self.ser = serial.Serial( '/dev/tty.usbserial-A600eu7L', self.rate )
                print "serial initialized"
            except:
                self.ser = serial.Serial( '/dev/tty.usbmodem411', self.rate )                
                print "serial initialized"

    @property
    def listen( self ):

        if self.recording is True:
            try:
                hnew = np.array( self.receiving() )
            except:
                hnew = 0. #self._ydata[-1] 
                print "missed a serial read"

        output  = np.append( self.plotdata.get_data("y"), hnew )
        print self.plotdata.get_data("y"), hnew, output

#        self.plotdata.set_data( "y",  output ) 
#        self.plotdata.set_data( "x", range( len( self.plotdata.get_data("y")  ) ) )

        return output

    def receiving( self, dt = None ):
        buffer, i, out, fill = '', 0, [], 0

        if dt is not None:
            self.dt = dt

        timediff = 0.
        starttime = time.time()
        while timediff < self.dt:
            try:
                buffer   = buffer + self.ser.read( self.ser.inWaiting() )
                now      = time.time()
                timediff = 1000 * (now - starttime)
                print timediff, self.dt
                1/0
            except:
                print "missed a serial read"

        if '\n' in buffer:
            lines = buffer.split( '\n' ) 

            for j in range( len(lines) ):
                o = re.findall( r"\d+", lines[j] )
                if o:
                    out.append( int( o[0] ) )
                else:
                    if len( self.plotdata.get_data("y") ) == 0:
                        out.append( 0. )
                    else:
                        out.append( 0. ) # self._ydata[-1] )                        
                    
        return out

    def plot_out( self ):
        pl.plot( self.out )

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
        self.recording = True
        self.run()

    def run( self ):
        counter = 0
        self.T = 100
        for i in range( self.T ):
            if self.recording is True:
                self.output = self.listen
                print self.output
                self.plotdata.set_data( "y",  self.output ) 
                self.plotdata.set_data( "x", range( len( self.plotdata.get_data("y")  ) ) )
                print self.plotdata.get_data("y")

                counter += 1
            else:
                counter = 10

    def _stop_fired( self ):
        self.recording = False

    def _load_data_fired( self ):
        pass

    def _save_data_fired( self ):
        S = Save( save = Save(), display=TextDisplay() )
        S.configure_traits()
        
        self.save( path = S.savepath, name = S.outname )

    def _save_plot_fired( self ):
        pass

#    @property
#    def _get_slice( self ):
#        if self.plot_type is "demo":
#            data_array = np.load( "../../../../../Data/20101119_fiberkontrol/gc_baseline_to_iso.npz" )
#            return data_array['arr_0'][0:int(self.T)]
#        else:
#            pass

#    def _output_changed( self ):
#        self._ydata = self.output
#        self._xdata = range( len( self._ydata ) )                

#    def _counter_changed( self ):
#        self.listen()
#        self._ydata = self.output
#        self._xdata = range( len( self._ydata ) )                

#    def _T_changed( self ):
#        self._ydata = self._get_slice
#        self._xdata = range( len( self._ydata ) )        



#------------------------------------------------------------------------------------------

class Save( HasTraits ):
    """ Save object """

    savepath  = Str( '../../../../../Data/20101119_fiberkontrol/',
                     desc="Location to save data file",
                     label="Save location:", )

    outname   = Str( '',
                     desc="Filename",
                     label="Save file as:", )

#------------------------------------------------------------------------------------------

class TextDisplay(HasTraits):
    string = String()

    view= View( Item('string', show_label = False, springy = True, style = 'custom' ) )

#------------------------------------------------------------------------------------------

class SaveDialog(HasTraits):
#    save = Instance( Save )
#    display = Instance( TextDisplay )

    view = View(
                Item('save', style='custom', show_label=False, ),
                Item('display', style='custom', show_label=False, ),
            )

#------------------------------------------------------------------------------------------


if __name__ == "__main__":
    F = FiberPlot()
    F.recording = True
    F.listen
    #F.configure_traits()
