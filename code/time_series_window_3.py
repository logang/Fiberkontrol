# Major imports
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

# Timer
from enthought.pyface.timer.api import Timer


from threading import Thread

#-----------------------------------------------------------------------------------------#

class FiberModel( HasTraits ):

    # Public Traits
    plot_type     = Enum( "record", "demo" )
    analog_in_pin = Int( 0 )
    T             = Int( 0 ) 
    dt            = Int( 1 ) # in ms; what does this do to accuracy
    analog_in_0   = Int( 1 ) 
    analog_in_1   = Int( 0 ) 
    analog_in_2   = Int( 0 ) 
    analog_in_3   = Int( 0 ) 

    savepath = '/Users/kellyz/Documents/Data/Fiberkontrol/20120118/'
    filename = '20120118-'

    view = View(Group( 'savepath',
                       'filename',
                       orientation="vertical"),
                buttons=["OK", "Cancel"],
                width = 500, height = 200 )


    # Private Traits
    _ydata  = Array()
    _xdata  = Array()
    _tdata  = Array() #time data
    _sdata  = Array() #shutter data
    

    shutterPlan = [0, 1, 0, 3, 0, 5, 0, 7] #in Hz
    


    def __init__(self, **kwtraits):
#        self.view.edit_traits()

        super( FiberModel, self ).__init__( **kwtraits )


 
        # debugging flag
        self.debug = True

        if self.plot_type is "record":
            try:
                import u6
                self.labjack = u6.U6()
                print "---------------------------------------------------"
                print "             Labjack U6 Initialized                "
                print "---------------------------------------------------"
                if self.debug:
                    self.labjack.debug = True
            except:
                raise IOError( "No Labjack detected." )

            # setup analog channel registers
            self.AIN0_REGISTER = 0
            self.num_analog_channels = self.analog_in_0 + self.analog_in_1 + self.analog_in_2 + self.analog_in_3

            # set recording to be true
            self.recording = True

    def _get_current_data( self ):
        """ If currently recording, query labjack for current analog input registers """
        if self.recording is True:
            current = self.labjack.readRegister( self.AIN0_REGISTER, numReg = self.num_analog_channels*2 )
            self._ydata = np.append( self._ydata, current )
#            self._xdata = range( len( self._ydata ) ) 
            self._xdata = range( 0, len( self._ydata ))

            #Normalize to seconds
#            self._xdata = self._xdata*self.dt/1000

            print 'ydata', self._ydata

    def _set_frequency( self ):
        """ Set the frequency of the shutter, according to a pretedermined plan """
        d = self.labjack
        print 'frequency'
        


    def plot_out( self ):
        pl.plot( self.out )

    def save( self, path = None, name = None ):
        if path is not None:
            self.savepath = path
        if name is not None:
            self.filename = name
#        np.savez( self.savepath + self.outname, self.output )
        full_save_path = self.savepath + self.filename
        print full_save_path
        np.savez( full_save_path, self._ydata )



    def load( self, path = None, name = None ):
        if path is not None:
            self.savepath = path
        self.loaded = np.load( self.savepath + '/out.npz' )['arr_0']

class FiberView( HasTraits ):

    timer         = Instance( Timer )
    model         =  FiberModel()

# Pop up the save window
    model.edit_traits()

    plot_data     = Instance( ArrayPlotData )
    plot          = Instance( Plot )
    record        = Button()
    stop          = Button()
    load_data     = Button()
    save_data     = Button()
    save_plot     = Button()

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
        # GUI window
        resizable = True,
        width     = 1000, 
        height    = 700 ,
        kind      = 'live' )

    def __init__(self, **kwtraits):
        super( FiberView, self ).__init__( **kwtraits )


        self.plot_data = ArrayPlotData( x = self.model._xdata, y = self.model._ydata )

        self.plot = Plot( self.plot_data )
        renderer  = self.plot.plot(("x", "y"), type="line", color="green")[0]

        # Start the timer!
        self.timer = Timer(self.model.dt, self.time_update) # update every 1 ms

    def run( self ):
#        for i in range( 100 ):
        self._plot_update()
        self.model._get_current_data()

    def time_update( self ):
        """ Callback that gets called on a timer to get the next data point over the labjack """
        if self.recording is True:
            self.model._set_frequency()

            self.model._get_current_data()
            self._plot_update()

    def _plot_update( self ):

        self.plot_data.set_data( "y", self.model._ydata ) 
        self.plot_data.set_data( "x", self.model._xdata )
        self.plot = Plot( self.plot_data )
        self.plot.plot(("x", "y"), type="line", color="green")[0]
        self.plot.request_redraw()

    # Note: These should be moved to a proper handler

    def _record_fired( self ):
        self.recording = True
        self.run()

    def _stop_fired( self ):
        self.recording = False

    def _load_data_fired( self ):
        pass

    def _save_data_fired( self ):
#        S = Save( save = Save(), display = TextDisplay() )
#        S.configure_traits()
        
#        self.save( path = S.savepath, name = S.outname )
        self.model.save( path = self.model.savepath, name = self.model.filename )


    def _save_plot_fired( self ):
        pass


#-----------------------------------------------------------------------------------------

class Save( HasTraits ):
    """ Save object """

    savepath  = Str( '/Users/kellyz/Documents/Data/Fiberkontrol/Data/20111208/',
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
    F = FiberView()
    F.recording = False
#    F.run()
#    1/0
#    F._get_current_data()
    F.configure_traits()
