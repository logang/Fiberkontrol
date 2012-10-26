import os,sys
import numpy as np
import scipy as sp
import pylab as pl
from debleach import exponential_basis_debleach
from scipy.stats import ranksums
from scipy.interpolate import UnivariateSpline

def get_event_window( event_ts, window_size = 1000 ):
    on_side = np.where(event_ts > 1)[0]
    diff = np.diff(event_ts)
    enter_event = np.where(diff > 1)[0]
    exit_event = np.where(diff < -1)[0]
    return on_side, enter_event, exit_event

class FiberAnalyze( object ):

    def __init__(self, options):
        self.smoothness = options.smoothness
        self.plot_type = options.plot_type
        self.input_path = options.input_path
        self.output_path = options.output_path

        self.fluor_channel = 0
        self.trigger_channel = 3
        
    def load( self ):
        """
        Load time series and events from npz file. 
        """
        self.data = np.load( self.input_path )['data']
        self.time_stamps = np.load( self.input_path )['time_stamps']

        # normalized fluor data, which is in arbitrary units
        self.fluor_data = self.data[:,self.fluor_channel]
        self.fluor_data -= np.min(self.fluor_data)
        self.fluor_data /= np.max(self.fluor_data)
        
        # normalize triggers to fluor data
        self.trigger_data = self.data[:,self.trigger_channel]
        self.trigger_data /= np.max(self.trigger_data)
        self.trigger_data -= np.min(self.trigger_data)
        self.trigger_data *= -1
        self.trigger_data += 1
        
    def plot_basic_tseries( self ):
        pl.clf()
        time_vals = self.time_stamps[range(len(self.fluor_data))]
        pl.plot( time_vals, self.data[:,3], 'r-', alpha=0.3 )
        pl.plot( time_vals, self.fluor_data, 'k-')
        pl.ylabel('Fluorescence Intensity (a.u.)')
        pl.xlabel('Time (seconds)')
        pl.show()

    def low_pass_filter(self, cutoff):
         rawsignal = self.fluor_data
         fft = sp.fft(rawsignal)
         bp = fft[:]
         for i in range(len(bp)):
              if i>= cutoff: bp[i] = 0
         ibp = sp.ifft(bp)
         low_pass_y = np.real(ibp)
         low_pass_y += np.median(self.fluor_data) - np.median(low_pass_y)
         return low_pass_y

    def plot_periodogram( self, window = None ):
        fft = sp.fft(self.fluor_data)
        1/0


if __name__ == "__main__":

        # Parse command line options
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-o", "--output-path", dest="output_path",
                      help="Specify the output path.")
    parser.add_option("-i", "--input-path", dest="input_path",
                      help="Specify the input path.")
    parser.add_option('-p', "--plot-type", default = 'tseries', dest="plot_type",
                      help="Type of plot to produce.")
    parser.add_option('-s', "--smoothness", default = None, dest="smoothness",
                      help="Should the time series be smoothed, and how much.")

    (options, args) = parser.parse_args()

    FA = FiberAnalyze( options )
    FA.load()
#    FA.plot_periodogram()
    FA.plot_basic_tseries()
    
# EOF
