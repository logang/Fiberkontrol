# import libraries
import struct, threading, Queue, ctypes, copy, sys, u6, LabJackPython, time
from time import sleep
from datetime import datetime
import numpy as np

#---------------------------------------------------------------------------------

class StreamDataReader(object):
    """
    Stream data off the LabJack, using threading for speed. 
    """
    def __init__( self, device, max_packets = None ):
        self.device = device
        if max_packets is not None:
            self.max_packets = max_packets
        else:
            self.max_packets = 1000
        self.data = Queue.Queue()
        self.dataCount = 0
        self.missed = 0
        self.running = False

    def readStreamData(self):
        self.running = True
        start = datetime.now()
        self.device.streamStart()
        while self.running:
            # Calling with convert = False, because we are going to convert in
            # the main thread.
            returnDict = self.device.streamData(convert = False).next()
            
            self.data.put_nowait(copy.deepcopy(returnDict))
            
            self.dataCount += 1
            if self.dataCount > self.max_packets:
                self.running = False
        
        print "stream stopped."
        self.device.streamStop()
        stop = datetime.now()
        total = ( self.dataCount * self.device.packetsPerRequest *
                  self.device.streamSamplesPerPacket )
        print "%s requests with %s packets per request with %s samples per packet = %s samples total." % ( self.dataCount,
                                                                                                           self.device.packetsPerRequest,
                                                                                                           self.device.streamSamplesPerPacket,
                                                                                                           total )
        
        print "%s samples were lost due to errors." % self.missed
        total -= self.missed
        print "Adjusted number of samples = %s" % total
        
        runTime = (stop-start).seconds + float((stop-start).microseconds)/1000000
        print "The experiment took %s seconds." % runTime
        print ( "%s samples / %s seconds = %s Hz" %
                ( total, runTime, float(total)/runTime ) )

#---------------------------------------------------------------------------------

class WholeFish( object ):
    """
    This class coordinates the collection of light field data of a fish brain using the Neo with
    high speed recording of the entire fish from below (currently with a Stanford Photonics 620G).
    """
    def __init__( self, parameters ):
        # number of packets to be read
        self.max_packets = parameters['max_packets']
        
        # At high frequencies ( >5 kHz), the number of samples will be max_packets
        # times 48 (packets per request) times 25 (samples per packet)
        self.num_samples = self.max_packets*48*25

        # initialize LabJack U6 device
        self.d = u6.U6()
    
        # For applying the proper calibration to readings.
        self.d.getCalibrationData()

        # number of analog channels
        self.num_analog_channels = parameters['num_analog_channels']

        # configure stream
        print "configuring U6 stream"
        self.d.streamConfig( NumChannels = self.num_analog_channels,
                             ChannelNumbers = range(self.num_analog_channels),
                             ChannelOptions = [0]*self.num_analog_channels,
                             SettlingFactor = 1,
                             ResolutionIndex = 1,
                             SampleFrequency = 50000/self.num_analog_channels )

        # get stream reader
        self.sdr = StreamDataReader(self.d,max_packets=self.max_packets)
        self.sdrThread = threading.Thread(target = self.sdr.readStreamData)

        # output list
        self.out = []
        self.out_arr = None
        
    def standby( self ):
        """
        Waits for signal from Arduino on register FIO3 indicating that the trial has begun.
        Once FIO3 goes high, starts run.
        Once run is finished, emits signal on FIO2 to let ardino know that
        the trial has ended. 
        """
        FIO0 = 6000 # input trigger
        FIO2 = 6102 # output to arduino
        wait = 0
        while wait != 1:
            try:
                wait = self.d.readRegister( FIO0 ) # read Neo analog in
            except KeyboardInterrupt:
                break
            except:
                print "Could not read input from Arduino."

        print "Running..."
        self.run()
        self.d.writeRegister( FIO2, 0 ) # tell ardiuno trial has ended

    def run( self ):
        # Start the stream and begin loading the result into a Queue
        self.sdrThread.start()

        # main run loop
        errors = 0
        missed = 0
        while True:
            try:
                #tic = time.clock()
                # Check if the thread is still running
                if not self.sdr.running:
                    break

                # Pull results out of the Queue in a blocking manner.
                result = self.sdr.data.get(True, 1)

                # If there were errors, print as much.
                if result['errors'] != 0:
                    errors += result['errors']
                    missed += result['missed']
                    print "+++++ Total Errors: %s, Total Missed: %s" % (errors, missed)

                # Convert the raw bytes (result['result']) to voltage data.
                r = self.d.processStreamData(result['result'])
                self.out.append( r )

                # how long does each block take?
                #print time.clock() - tic

            except Queue.Empty:
                print "Queue is empty. Stopping..."
                self.sdr.running = False
                break
            except KeyboardInterrupt:
                self.sdr.running = False
            except Exception, e:
                print type(e), e
                self.sdr.running = False
                break

        # Finally, construct and save data array
        print "Saving acquired data..."
        for i in xrange( len( self.out ) ):
            new_array = 1
            block = self.out[i]
            for k in r.keys():
                if k != 0:
                    if new_array == 1:
                        array = np.array(block[k])
                        new_array = 0
                    else:
                        array = np.vstack((array, np.array(block[k]) ))
            if i == 0:
                self.out_arr = array.T
            else:
                self.out_arr = np.vstack( (self.out_arr, array.T) )

    def save(self, outfile):
        """ Save out_arr to outfile.npz """
        if self.out_arr is not None:
            np.savez(outfile, self.out_arr)
        else:
            raise IOError( "Nothing to save in out_arr.")
        self.d.close()

#---------------------------------------------------------------------------------

if __name__ is "__main__":

    parameters = { 'max_packets': 28000, # about 2500 per minute
                   'num_analog_channels': 4,
                   'testing': False }

    WF = WholeFish(parameters)
    WF.standby()
    WF.save('20120622_mouse_8346_2_3CT')

# EOF
