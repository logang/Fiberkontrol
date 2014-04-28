# Major imports
import numpy as np
import sys, serial, re, time, random, wx, threading, os, Queue

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

# streaming data reader
from streaming_record import StreamDataReader

#-----------------------------------------------------------------------------------------#

class FiberModel( HasTraits ):
    # Traits - other functions are listening for these to change

    # Public Traits
    plot_type     = Enum( "record" )
    dt            = Int( 1 ) # in ms; what does this do to accuracy

    savepath = '/Users/kellyz/Documents/Data/Fiberkontrol/20130108/'
    filename = '20130108-'

    if not os.path.isdir(savepath):
        os.mkdir(savepath)
        print "Made directory ", savepath

    view = View(Group( 'savepath',
                       'filename',
                       orientation="vertical"),
                buttons=["OK", "Cancel"],
                width = 500, height = 200 )

    # Private Traits
    _ydata  = Array()
    _tdata  = Array() #time data

    master_index = 0

    def __init__(self, options, **kwtraits):
        super( FiberModel, self).__init__(**kwtraits )
 
        # debugging flag
        self.debug = False

        # to record or not
        self.recording = False
        self.ttl_start = False

        # counter
        self.master_index = 0

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

            # number of packets to be read
            self.max_packets = options.max_packets # ~4 min

            # At high frequencies ( >5 kHz), the number of samples will be max_packets
            # times 48 (packets per request) times 25 (samples per packet)
            self.num_samples = self.max_packets*25 # For 1000 Hz

            # preallocate data plotting arrays to zeros
            self.preallocate_arrays()

            # For applying the proper calibration to readings.
            self.labjack.getCalibrationData()

            # number of analog channels 
            self.num_analog_channels = options.num_analog_channels

            # configure stream
            print "configuring U6 stream"
            self.labjack.streamConfig( NumChannels = self.num_analog_channels,
                                       ChannelNumbers = range(self.num_analog_channels),
                                       ChannelOptions = [0]*self.num_analog_channels,
                                       SettlingFactor = 1,
                                       ResolutionIndex = 1,
                                       SampleFrequency = 1000/self.num_analog_channels ) # XXX hard coded!!!

            # get stream reader
            self.sdr = StreamDataReader(self.labjack,max_packets=self.max_packets)
            self.sdrThread = threading.Thread(target = self.sdr.readStreamData)

            # turn on LED
            self.labjack.writeRegister(6002, 1)
            time.sleep(0.1)
            self.labjack.writeRegister(6002, 0)

            
            # output list
            self.data = []

            # initialize timer
            self.prev_time = 0

            self.start_time = time.time()


            #---Constants for Labjack pulse width modulation (PWM) -----
            self.LED_DUTYCYCLE = 50 #PWM duty cycle in %
            self.LED_FREQUENCYhz = 1 #in Hz
            
#            self.LED_PERIODms = 1000
#            self.LED_DURATIONms = 500

            print "Initialization complete."

    def start_LED(self):
        self.labjack.writeRegister(6002, 0)
        self.labjack.writeRegister(6002, 1)
        self.labjack.writeRegister(6002, 0)
        self.labjack.writeRegister(6002, 1) #change FI02 to output           


        self.TIME_CLOCK_BASE = 4000000 #in Hz                                
        self.FREQUENCY_CONVERSION_CONSTANT = 65536
        self.PWM_CONSTANT = 65536

        self.labjack.writeRegister(50501, 1) # enable one timer
        self.labjack.writeRegister(50500, 2) #change the offset (output) to FIO2 
        self.labjack.writeRegister(7100, 0) #for 16bit PWM output            
        self.labjack.writeRegister(7000, 4) #for 4 MHz/divisor base clock    
        self.labjack.writeRegister(7200, 65536) #init the  duty cycle to 0%  

        #set the duty cycle:
        self.labjack.writeRegister(7200, self.PWM_CONSTANT 
                                   - (self.PWM_CONSTANT*self.LED_DUTYCYCLE/100)) 


        divisor = int(self.TIME_CLOCK_BASE/(self.FREQUENCY_CONVERSION_CONSTANT
                                            *self.LED_FREQUENCYhz))

        self.labjack.writeRegister(7002, divisor) #set the pwm frequency
        actual_freq = self.TIME_CLOCK_BASE/(self.FREQUENCY_CONVERSION_CONSTANT
                                                 *float(divisor))
        print "ACTUAL_FREQ ", actual_freq

        """#original LED control based on computer's timer. switched to pwm
        cur_time = time.time() - self.start_time
        print cur_time
        if (int(cur_time*1000) % self.LED_PERIODms) < self.LED_DURATIONms:
            self.labjack.writeRegister(6002, 1)
            print "LED"
        else:
            self.labjack.writeRegister(6002, 0) 
        """

    def preallocate_arrays(self):
        self.num_samples = self.max_packets*25 # Hard coded for 1000 Hz
        self._ydata  = np.zeros(self.num_samples)
        self._tdata  = np.zeros(self.num_samples) #time data

    def _get_current_data( self ):
        """ If currently recording, query labjack for current input registers """
        
        if self.recording:

            # Start the stream and begin loading the result into a Queue
            if not self.sdr.running:
                self.sdrThread.start()
                print "Thread started."

            # main run loop
            errors = 0
            missed = 0
            try:
                if self.debug:
                    tic = time.clock()

                # Pull results out of the Queue in a blocking manner.
                result = self.sdr.data.get(True, 1)
                
                # If there were errors, print them to terminal.
                if result['errors'] != 0:
                    errors += result['errors']
                    missed += result['missed']
                    print "Total Errors: %s, Total Missed: %s" % (errors, missed)

                # Convert the raw bytes (result['result']) to voltage data.
                self.result = self.labjack.processStreamData(result['result'])
#                self.result = result['result']
                self.data.append( self.result )
                self.block_size = len(self.result[self.result.keys()[0]])

                # append data to time series
                self._ydata[self.master_index:(self.master_index+self.block_size)] = np.array(self.result[self.result.keys()[0]])
#                self._tdata[self.master_index:(self.master_index+self.block_size)] = np.array(range(self.master_index, self.master_index+self.block_size))                


                if self.master_index == 0:
                    cur_time = time.time()
                    self._tdata[self.master_index:(self.master_index+self.block_size)] = (cur_time - self.start_time)/(self.block_size+1)*np.array(range(1, self.block_size+1))
                else:
                    cur_time = time.time()
                    prev_time = self._tdata[self.master_index - 1] 
                    #print "prev_time", prev_time
                    self._tdata[self.master_index:(self.master_index+self.block_size)] = prev_time*np.ones(self.block_size) + ((cur_time-prev_time - self.start_time)/(self.block_size+1))*np.array(range(1, self.block_size+1))

#                print self.result.keys()[0]

                if self.debug:
                    # how long does each block take?
                    print time.clock() - tic

            except Queue.Empty:
                print "Queue is empty. Stopping..."
                self.sdr.running = False
                self.recording = False

            except KeyboardInterrupt:
                self.sdr.running = False
                self.recording = False

            except Exception, e:
                print type(e), e
                self.sdr.running = False
                self.recording = False

        self.master_index += self.block_size
        
#------------------------------------------------------------------------------

class FiberView( HasTraits ):

    timer         = Instance( Timer )
#    model         =  FiberModel(options)

    plot_data     = Instance( ArrayPlotData )
    plot          = Instance( Plot )
    start_stop    = Button()
    exit          = Button()

    # Default TraitsUI view
    traits_view = View(
        Item('plot', editor=ComponentEditor(), show_label=False),
        # Items
        HGroup( spring,
                Item( "start_stop", show_label = False ),
                Item( "exit", show_label = False ), spring),
        HGroup( spring ),

        # GUI window
        resizable = True,
        width     = 1000, 
        height    = 700,
        kind      = 'live' )

    def __init__(self, options, **kwtraits):
        super( FiberView, self).__init__(**kwtraits )
        self.model = FiberModel(options)

        # debugging
        self.debug = options.debug
        self.model.debug = options.debug

        # timing parameters
        self.max_packets = options.max_packets
        self.hertz = options.hertz

        # extend options to model
        self.model.max_packets = options.max_packets
        self.model.preallocate_arrays()
        self.model.num_analog_channels = options.num_analog_channels

        # generate traits plot
        self.plot_data = ArrayPlotData( x = self.model._tdata, y = self.model._ydata )
        self.plot = Plot( self.plot_data )
        renderer  = self.plot.plot(("x", "y"), type="line", name='old', color="green")[0]
#        self.plot.delplot('old')

        # recording flags
        self.model.recording = False
        self.model.trialEnded = True


        print 'Viewer initialized.'

        # should we wait for a ttl input to start?
        if options.ttl_start:
            self.model.ttl_start = True
            self.ttl_received = False

            # initialize FIO0 for TTL input
            self.FIO0_DIR_REGISTER = 6100 
            self.FIO0_STATE_REGISTER = 6000 
            self.model.labjack.writeRegister(self.FIO0_DIR_REGISTER, 0) # Set FIO0 low

        # initialize output array
        self.out_arr = None

        # keep track of number of runs
        self.run_number = 0

        self.timer = Timer(self.model.dt, self.time_update) # update every 1 ms


    def run( self ):
        self._plot_update()
        self.model._get_current_data()


    def time_update( self ):
        """ Callback that gets called on a timer to get the next data point over the labjack """

#        print "time_update"
#        print "self.model.ttl_start", self.model.ttl_start
#        print "self.ttl_received", self.ttl_received
#        print "self.model.recording", self.model.recording

        if self.model.ttl_start and not self.ttl_received:
            ttl = self.check_for_ttl()
            if ttl:
                self.ttl_received = True
#                self.model.recording = True
#                self.model.trialEnded = False
                self._start_stop_fired()
               
        if self.model.recording and not self.model.ttl_start:
            self.run()

        elif self.model.ttl_start:
            if self.model.recording and self.ttl_received:
                self.run()
#            elif self.model.recording:
#                self._start_stop_fired()
        else:
            if self.debug:
                pass #print "--timer tic--"
            pass

    def check_for_ttl(self):
        start_ttl = self.model.labjack.readRegister(self.FIO0_STATE_REGISTER)
#        print "start_ttl: ", start_ttl
        return start_ttl

    def clean_time_series(self, time_series, blip_thresh = 10.0):
        """ Removes blips, NAs, etc. """
        blip_idxs = time_series > blip_thresh
        time_series[blip_idxs] = np.median(time_series)
        return time_series

    def _plot_update( self ):
        num_display_points = 100*25 # For 1000 Hz
    
        if self.model.master_index > num_display_points:
            disp_begin = self.model.master_index - num_display_points
            disp_end = self.model.master_index
#            print 'disp_begin', disp_begin
#            print 'disp_end', disp_end
            ydata = self.clean_time_series( np.array(self.model._ydata[disp_begin:disp_end]) )
            xdata = np.array(self.model._tdata[disp_begin:disp_end])

            self.plot_data.set_data("y", ydata)
            self.plot_data.set_data("x", xdata)
            
        else:        
            self.plot_data.set_data( "y", self.clean_time_series( self.model._ydata[0:self.model.master_index] )) 
            self.plot_data.set_data( "x", self.model._tdata[0:self.model.master_index] )

        self.plot = Plot( self.plot_data )

#        self.plot.delplot('old')
        the_plot = self.plot.plot(("x", "y"), type="line", name = 'old', color="green")[0]
        self.plot.request_redraw()

    # Note: These should be moved to a proper handler (ModelViewController)

    def _start_stop_fired( self ):
        self.model.start_time = time.time()
        self.model.recording = not self.model.recording
        self.model.trialEnded = not self.model.trialEnded
        if self.model.trialEnded:
            self.save()
        
        #Quickly turn LED off to signal button push
        self.model.start_LED()


    def _exit_fired(self):
        print "Closing connection to LabJack..."
        self.ttl_start = False
        self.recording = False
        self.model.sdr.running = False
        self.model.recording = False

#        self.model.labjack.streamStop()
        self.model.labjack.close()
        print "connection closed. Safe to close..."
        #raise SystemExit
        #sys.exit()

    def save( self ):
            print "Saving!"        
            # Finally, construct and save full data array
            print "Saving acquired data..."
            for i in xrange( len( self.model.data ) ):
                new_array = 1
                block = self.model.data[i]
                for k in self.model.result.keys():
                    print k
                    if k != 0:
                        if new_array == 1:
                            array = np.array(block[k])
                            array.shape = (len(array),1)
                            new_array = 0
                        else:
                            new_data = np.array(block[k])
                            new_data.shape = (len(block[k]),1)
                            # if new_data is short by one, fill in with last entry
                            if new_data.shape[0] < array.shape[0]:
                                new_data = np.append( new_data, new_data[-1] )
                                new_data.shape = (new_data.shape[0], 1)
                                print "Appended point to new_data, shape now:",new_data.shape
                            if new_data.shape[0] > array.shape[0]:
                                new_data = new_data[:-1]
                                print "Removed point from new_data, shape now:",new_data.shape
                            print "array shape, new_data shape:", array.shape, new_data.shape
                            array = np.hstack((array, new_data ))
                if i == 0:
                    self.out_arr = array
                    print "Array shape:", self.out_arr.shape
                else:
                    self.out_arr = np.vstack( (self.out_arr, array) )
                    print "Array shape:", self.out_arr.shape
            date = time.localtime()
            outfile = self.model.savepath + self.model.filename
            outfile += str(date[0]) + '_' + str(date[1]) + '_' + str(date[2]) + '_'
            outfile += str(date[3]) + '-' + str(date[4]) + '_run_number_' + str(self.run_number) 
            np.savez(outfile, data=self.out_arr, time_stamps=self.model._tdata)

#             time_outfile = outfile + '_t'
#             np.savez(time_outfile, self.model._tdata)
            print "Saved ", outfile

            # Plot the data collected this last run
            self.plot_last_data()

            # clean up
            self.reset_variables()

    def reset_variables(self):
        self.out_arr = None
        self.ttl_received = False
        self.run_number += 1
        self.model.recording = False
        self.trialEnded = True
        self.plot.delplot('old')

    def plot_last_data(self):
        import pylab as pl
        if self.out_arr.shape[1] == 4:
            pl.figure()
            pl.subplot(411)
            pl.plot(self.out_arr[:,0])
            pl.subplot(412)
            pl.plot(self.out_arr[:,1])
            pl.subplot(413)
            pl.plot(self.out_arr[:,2])
            pl.subplot(414)
            pl.plot(self.out_arr[:,3])
            pl.show()

#-----------------------------------------------------------------------------------------


if __name__ == "__main__":

    # Parse command line options
    from optparse import OptionParser
    
    parser = OptionParser()

    parser.add_option('', "--hertz", dest="hertz", type=int, default=250, 
                      help="Sampling rate (per channel) in Hz.")
    parser.add_option('', "--max-packets", dest="max_packets", type=int, default=200000, # about 2500 per minute
                      help="The maximum number of packets collected in a run.")
    parser.add_option('', "--num-analog-channels", dest="num_analog_channels", type=int, default=4, 
                      help="The number of analog channels recorded on the LabJack.")
    parser.add_option('', "--ttl-start",
                      action="store_true", dest="ttl_start", default=False,
                      help="Wait for a TTL input on FIO0 to start recording.")
    parser.add_option('', "--debug",
                      action="store_true", dest="debug", default=False,
                      help="Turn on debugging output.")

    (options, args) = parser.parse_args()

    F = FiberView(options)
    F.model.edit_traits()
    F.configure_traits()

