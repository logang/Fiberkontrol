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
    # Traits - other functions are listening for this to change

    # Public Traits
    plot_type     = Enum( "record", "demo" )
    analog_in_pin = Int( 0 )
    T             = Int( 0 ) 
    dt            = Int( 1 ) # in ms; what does this do to accuracy
    analog_in_0   = Int( 1 ) 
    analog_in_1   = Int( 1 ) 
    analog_in_2   = Int( 1 ) 
    analog_in_3   = Int( 1 ) 

    savepath = '/Users/kellyz/Documents/Data/Fiberkontrol/20120130/'
    filename = '20120130-'

    view = View(Group( 'savepath',
                       'filename',
                       orientation="vertical"),
                buttons=["OK", "Cancel"],
                width = 500, height = 200 )


    # Private Traits
    _ydata  = Array()
    _i1data = Array() #input 1 data
    _i2data = Array() #input 2 data
    _i3data = Array() #input 3 data
    _tdata  = Array() #time data
    _sdata  = Array() #shutter data

    _ydata  = np.zeros(300000)
    _i1data = np.zeros(300000) #input 1 data
    _i2data = np.zeros(300000) #input 2 data
    _i3data = np.zeros(300000) #input 3 data
    _tdata  = np.zeros(300000) #time data
    _sdata  = np.zeros(300000) #shutter data
    master_index = 0


    def __init__(self, **kwtraits):
#        self.view.edit_traits()

        super( FiberModel, self ).__init__( **kwtraits )


 
        # debugging flag
        self.debug = False

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
            self.AIN1_REGISTER = 2
            self.AIN2_REGISTER = 4
            self.AIN3_REGISTER = 6

            self.num_analog_channels = self.analog_in_0 + self.analog_in_1 + self.analog_in_2 + self.analog_in_3

            #setup streaming
            print "configuring U6 stream"
            self.labjack.streamConfig(  NumChannels = 4, ChannelNumbers = [ 0, 1, 2, 3], 
                                        ChannelOptions = [ 0, 0, 0, 0], SettlingFactor = 1, 
                                        ResolutionIndex = 1, ScanInterval =  6000, SamplesPerPacket = 25)

            self.prev_time = 0

            # setup frequency control

            #---Constants for PWM frequency-----
            self.TIME_CLOCK_BASE = 4000000 #in Hz
            self.FREQUENCY_CONVERSION_CONSTANT = 65536
            self.PWM_CONSTANT = 65536
            

            self.labjack.writeRegister(6002, 1) #change FI02 to output
            self.labjack.writeRegister(50501, 1) # enable one timer
            self.labjack.writeRegister(50500, 2) #change the offset (output pin) to FIO2            

            self.labjack.writeRegister(7100, 0) #for 16bit PWM output
            self.labjack.writeRegister(7000, 4) #for 4 MHz/divisor base clock
            self.labjack.writeRegister(7200, 65535) #init the  duty cycle to 0%


            #---Shutter plan settings ------------
            self.start_time = time.time()
            self.shutterChunkTime = 10 #in seconds

            base = 1
            alt = 5
            self.shutterPlan = [0, base, 0, alt, 0, base, 0, alt,
                                0, base, 0, alt, 0, base, 0, alt,
                                0, base, 0, alt, 0, base, 0, alt]

            freq = 0.5 #in Hz
            if(True):
                self.shutterPlan = [0, freq, 0, freq, 0, freq, 0, freq, 0,
                                    freq, 0, freq, 0, freq, 0, freq, 0, freq,
                                    0, freq, 0, freq, 0, freq, 0, freq, 0, freq,
                                    0, freq, 0, freq, 0, freq, 0, freq, 0, freq,
                                    0, freq, 0, freq, 0, freq, 0, freq, 0, freq,
                                    0, freq, 0, freq, 0, freq, 0, freq, 0, freq,
                                    0, freq, 0, freq, 0, freq, 0, freq, 0, freq]


            # set recording to be true
            self.recording = True




    def _get_current_data( self ):
        """ If currently recording, query labjack for current analog input registers """
        if self.recording is True:

            rgen = self.labjack.streamData()
            rdata = rgen.next()

            self.curr_time = time.time() - self.start_time
            
            if rdata is None:
                print 'ERROR rdata is NONE'
            else:
                last_y = 0
                last_i1 = 0
                last_i2 = 0
                last_i3 = 0

                maxNumData = len(rdata['AIN0'])
                if maxNumData > len(rdata['AIN1']):
                    maxNumData = len(rdata['AIN1'])
                if maxNumData > len(rdata['AIN2']):
                    maxNumData = len(rdata['AIN2'])
                if maxNumData > len(rdata['AIN3']):
                    maxNumData = len(rdata['AIN3'])

                print 'maxNumData', maxNumData
                print 'master_index', self.master_index

                for dataCount in range(maxNumData):

                    # add all data points
                    # account for weird blips where detector saturates for one time step
                    
                    if rdata['AIN0'][dataCount] < 10.1:
                        self._ydata[self.master_index + dataCount -1] = rdata['AIN0'][dataCount]
#                        self._ydata = np.append( self._ydata, rdata['AIN0'][dataCount])            
                        last_y = rdata['AIN0'][dataCount]
                    else: 
                        self._ydata[self.master_index + dataCount -1] = last_y
#                        self._ydata = np.append( self._ydata, last_y)            

                    numData = len(rdata['AIN0'])
                    timeChunk = dataCount*(self.curr_time - self.prev_time)/numData
                    self._tdata[self.master_index + dataCount -1] = self.prev_time + timeChunk
#                    self._tdata = np.append( self._tdata, self.prev_time + timeChunk)            

                    self._sdata[self.master_index + dataCount -1] = self.actual_freq
#                    self._sdata = np.append( self._sdata, self.actual_freq)            



                    # 'AIN1'
#                    print 'AIN1', rdata['AIN1']

                    if rdata['AIN1'][dataCount] < 10.1:
                        self._i1data[self.master_index + dataCount -1] = rdata['AIN1'][dataCount]
#                        self._i1data = np.append( self._i1data, rdata['AIN1'][dataCount])             
                        last_i1 = rdata['AIN1'][dataCount]
                    else: 
                        self._i1data[self.master_index + dataCount -1] = last_i1
#                        self._i1data = np.append( self._i1data, last_i1)             

                    # 'AIN2'
                    if rdata['AIN2'][dataCount] < 10.1:
                        self._i2data[self.master_index + dataCount -1] = rdata['AIN2'][dataCount]
#                        self._i2data = np.append( self._i2data, rdata['AIN2'][dataCount])             
                        last_i2 = rdata['AIN2'][dataCount]
                    else: 
                        self._i2data[self.master_index + dataCount -1] = last_i2
#                        self._i2data = np.append( self._i2data, last_i2)             

                    # 'AIN3'
                    if rdata['AIN3'][dataCount] < 10.1:
                        self._i3data[self.master_index + dataCount -1] = rdata['AIN3'][dataCount]
#                        self._i3data = np.append( self._i3data, rdata['AIN3'][dataCount])             
                        last_i3 = rdata['AIN3'][dataCount]
                    else: 
                        self._i3data[self.master_index + dataCount -1] = last_i3
#                        self._i3data = np.append( self._i3data, last_i3)             

            print 'overall time difference: ', self.curr_time - self.prev_time
            self.prev_time = self.curr_time

            self.master_index += dataCount - 1

    def _set_frequency( self ):
        """ Set the frequency of the shutter, according to a pretedermined plan """
#        self.labjack
        
#        print 'frequency'
        
        currTime = time.time() - self.start_time
        index = int(currTime/self.shutterChunkTime)
#        print 'index', index

        desired_freq = self.shutterPlan[index] #in Hz
#        print 'desired_freq', desired_freq

        pin = 2 #FI02
        pulsewidth = 50 #PWM %

        if (desired_freq == 0):
            pulsewidth = 0
            self.actual_freq = 0
 #           self._sdata = np.append( self._sdata, 0)            
        else:
            divisor = int(self.TIME_CLOCK_BASE/(self.FREQUENCY_CONVERSION_CONSTANT*desired_freq))
            self.labjack.writeRegister(7002, divisor) #set the divisor
            self.actual_freq = self.TIME_CLOCK_BASE/(self.FREQUENCY_CONVERSION_CONSTANT*float(divisor))
#            self._sdata = np.append( self._sdata, actual_freq)

        self.labjack.writeRegister(7200, self.PWM_CONSTANT - (self.PWM_CONSTANT*pulsewidth/100)) #set the duty cycle


        ## To test readout of AIN1, 2, 3
        if (desired_freq == 0):
            self.labjack.writeRegister(5000, 3.7)
        else:
            self.labjack.writeRegister(5000, 0)

        
    def plot_out( self ):
        pl.plot( self.out )

    def save( self, path = None, name = None ):

        if path is not None:
            self.savepath = path
        if name is not None:
            self.filename = name

        full_save_path = self.savepath + self.filename
        print full_save_path

        full_save_path_t = full_save_path + '_t'
        full_save_path_s = full_save_path + '_s'
        full_save_path_i1= full_save_path + '_i1'
        full_save_path_i2= full_save_path + '_i2'
        full_save_path_i3= full_save_path + '_i3'

        np.savez( full_save_path, self._ydata )
        np.savez( full_save_path_t, self._tdata )
        np.savez( full_save_path_s, self._sdata )
        np.savez( full_save_path_i1, self._i1data )
        np.savez( full_save_path_i2, self._i2data )
        np.savez( full_save_path_i3, self._i3data )


        self.labjack.streamStop()
        self.labjack.close()



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

        self.plot_data = ArrayPlotData( x = self.model._tdata, y = self.model._ydata )
#        self.plot_data = ArrayPlotData( x = self.model._tdata, y = self.model._i1data )
        

        self.plot = Plot( self.plot_data )
        renderer  = self.plot.plot(("x", "y"), type="line", name='old', color="green")[0]
        self.plot.delplot('old')

        # Start the timer!

        self.timer = Timer(self.model.dt, self.time_update) # update every 1 ms

    def run( self ):
        self.model.start_time = time.time()
        self.model.prev_time = 0
        
        self.model.labjack.streamStart()

        self.model._set_frequency()
        self._plot_update()
        self.model._get_current_data()

        
#        while(self.recording):
#            self.time_update()
#            self.plot.request_redraw()

    def time_update( self ):
        """ Callback that gets called on a timer to get the next data point over the labjack """

        

        if self.recording is True:
            time1 = time.time()-self.model.start_time

            self.model._set_frequency()

            time2 = time.time()-self.model.start_time
            print 'set_freq:', time2-time1

            self.model._get_current_data()     

            time3 = time.time()-self.model.start_time
            print 'get_current_data', time3-time2

            self._plot_update()

            time4 = time.time()-self.model.start_time
            print 'plot_update', time4-time3

    def _plot_update( self ):

        num_display_points = 7000
#        down_sample_factor = 50
    
        if self.model.master_index > num_display_points:
            disp_begin = self.model.master_index - num_display_points
            disp_end = self.model.master_index
            print 'disp_begin', disp_begin
            print 'disp_end', disp_end
            ydata = np.array(self.model._ydata[disp_begin:disp_end])
            xdata = np.array(self.model._tdata[disp_begin:disp_end])
                        
#            down_ydata = np.zeros(num_display_points/down_sample_factor)
#            down_xdata = np.zeros(num_display_points/down_sample_factor)
#            for i in range(num_display_points):
#                if i%down_sample_factor == 0:
#                    down_ydata[i/down_sample_factor] = ydata[i]
#                    down_xdata[i/down_sample_factor] = xdata[i]

            
            self.plot_data.set_data("y", ydata)
            self.plot_data.set_data( "x", xdata)
            

        else:        
            self.plot_data.set_data( "y", self.model._ydata[0:self.model.master_index] ) 
        #        self.plot_data.set_data( "y", self.model._i1data ) 
            self.plot_data.set_data( "x", self.model._tdata[0:self.model.master_index] )
        self.plot = Plot( self.plot_data )

    
#        self.plot.delplot('old')
        the_plot = self.plot.plot(("x", "y"), type="line", name = 'old', color="green")[0]
        self.plot.request_redraw()

    # Note: These should be moved to a proper handler (ModelViewController)

    def _record_fired( self ):
        self.recording = True
        self.run()
            

    def _stop_fired( self ):
        self.recording = False

    def _load_data_fired( self ):
        pass

    def _save_data_fired( self ):
        self.model.save( path = self.model.savepath, name = self.model.filename )
        print 'time',self.model._tdata


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
    F.configure_traits()
