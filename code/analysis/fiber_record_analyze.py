import os,sys
import numpy as np
import scipy as sp
import pylab as pl
import scipy.signal as signal
#from debleach import exponential_basis_debleach
from scipy.stats import ranksums
from scipy.interpolate import UnivariateSpline

#-----------------------------------------------------------------------------------------

class FiberAnalyze( object ):

    def __init__(self, options):
        """
        Initialize the FiberAnalyze class using options object from OptionsParser.
        """
        # values from option parser
        self.smoothness = options.smoothness
        self.plot_type = options.plot_type
        self.input_path = options.input_path
        self.output_path = options.output_path
        self.time_range = options.time_range
        if options.trigger_input is not None:
            self.s_file = options.trigger_input[0]
            self.e_file = options.trigger_input[1]
        else:
            self.trigger_input = None
            
        # hard coded values
        self.fluor_channel = 0
        self.trigger_channel = 3
        
    def load( self ):
        """
        Load time series and events from NPZ file. 
        """
        self.data = np.load( self.input_path )['data']
        self.time_stamps = np.load( self.input_path )['time_stamps']

        # normalized fluor data, which is in arbitrary units
        self.fluor_data = self.data[:,self.fluor_channel]
        self.fluor_data -= np.min(self.fluor_data)
        self.fluor_data /= np.max(self.fluor_data)
        
        # normalize triggers to fluor data
        if self.trigger_input is None:
            self.trigger_data = self.data[:,self.trigger_channel]
        else:
            try:
                trigger_files = self.trigger_input.split(":")
                time_tuples = self.load_trigger_data(trigger_files[0], trigger_files[1])
            except Exception, e:
                print "Error loading trigger data:"
                print "\t-->",e
                
        self.trigger_data /= np.max(self.trigger_data)
        self.trigger_data -= np.min(self.trigger_data)
        self.trigger_data *= -1
        self.trigger_data += 1

        # if time range is specified, crop data    
        if options.time_range != None:
            tlist = options.time_range.split(':')
            if len(tlist) != 2:
                print 'Error parsing --time-range argument.  Be sure to use <start-time>:<end-time> syntax.'
                sys.exit(1)
            t_start = int(tlist[0])
            t_end   = int(tlist[1])
            self.fluor_data = self.fluor_data[t_start:t_end]
            self.trigger_data = self.trigger_data[t_start:t_end]

    def load_trigger_data( self, s_filename, e_filename ):
        self.s_vals = np.load(s_filename)['arr_0']
        self.e_vals = np.load(e_filename)['arr_0']
        return zip(self.s_vals,self.e_vals)

    def plot_basic_tseries( self ):
        """
        Generate a plot showing the raw calcium time series, with triggers
        corresponding to events (e.g. licking for sucrose) superimposed.
        Note this is currently rather overfit to our setup.
        """
        pl.clf()
        time_vals = self.time_stamps[range(len(self.fluor_data))]
        pl.plot( time_vals, self.data[:,3], 'r-', alpha=0.3 )
        pl.plot( time_vals, self.fluor_data, 'k-')
        pl.ylabel('Fluorescence Intensity (a.u.)')
        pl.xlabel('Time (seconds)')
        pl.show()

    def low_pass_filter(self, cutoff):
        """
        Low pass filter the data with frequency cutoff: 'cutoff'.
        """
        rawsignal = self.fluor_data
        fft = sp.fft(rawsignal)
        bp = fft[:]
        for i in range(len(bp)):
            if i>= cutoff: bp[i] = 0
        ibp = sp.ifft(bp)
        low_pass_y = np.real(ibp)
        low_pass_y += np.median(self.fluor_data) - np.median(low_pass_y)
        return low_pass_y

    def get_peaks( self ):
        """
        Heuristic for finding local peaks in the calcium data. 
        """
        peak_widths = np.array([50,100,500,1000])
        self.peak_inds = signal.find_peaks_cwt(self.fluor_data, widths=peak_widths, wavelet=None, max_distances=None, gap_thresh=None, min_length=None, min_snr=5, noise_perc=50)
        self.peak_vals = self.fluor_data[self.peak_inds]
        self.peak_times = self.time_stamps[self.peak_inds]
        return self.peak_inds, self.peak_vals, self.peak_times

    # --- not yet implemented --- #

    def debleach( self ):
        """
        Remove trend from data due to photobleaching. 
        """
        pass

    def plot_periodogram( self, window = None ):
        """
        Plot periodogram of fluoroscence data.
        """
        fft = sp.fft(self.fluor_data)
        1/0 # not finished
        
    def plot_peak_data( self ):
        """
        Plot fluorescent data with chosen peak_inds overlayed as lines.
        """
        lines = np.zeros(len(self.fluor_data))
        lines[self.peak_inds] = 1.0
        pl.plot(self.fluor_data)
        pl.plot(lines)
        pl.show()
    
    def plot_perievent_hist( self, event_times ):
        """
        Peri-event time histogram for given event times.
        """
        pass

    def plot_peak_statistics( self, peak_times, peak_vals ):
        """
        Plots showing statistics of calcium peak data.
          --> Peak height as function of time since last peak
          --> Histograms of peak times and vals
        """
        pass

#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------

def get_event_window( event_ts, window_size = 1000 ):
    on_side = np.where(event_ts > 1)[0]
    diff = np.diff(event_ts)
    enter_event = np.where(diff > 1)[0]
    exit_event = np.where(diff < -1)[0]
    return on_side, enter_event, exit_event

#-----------------------------------------------------------------------------------------

def test_FiberAnalyze(options):
    """
    Test the FiberAnalyze class.
    """
    FA = FiberAnalyze( options )
    FA.load()
#    FA.plot_periodogram()
    FA.plot_basic_tseries()
    peak_inds, peak_vals, peak_times = FA.get_peaks()
    FA.plot_peak_data()
    1/0

#-----------------------------------------------------------------------------------------

if __name__ == "__main__":

    # Parse command line options
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-o", "--output-path", dest="output_path",
                      help="Specify the output path.")
    parser.add_option("", "--trigger-input", dest="trigger_input",default=None,
                      help="Specify filenames for NPZ files with trigger start and end times in format s_file:e_file.")
    parser.add_option("-i", "--input-path", dest="input_path",
                      help="Specify the input path.")
    parser.add_option("", "--time-range", dest="time_range",default=None,
                      help="Specify a time window over which to analyze the time series.")
    parser.add_option('-p', "--plot-type", default = 'tseries', dest="plot_type",
                      help="Type of plot to produce.")
    parser.add_option('-s', "--smoothness", default = None, dest="smoothness",
                      help="Should the time series be smoothed, and how much.")

    (options, args) = parser.parse_args()
    
    # Test the class
    test_FiberAnalyze(options)
    
# EOF
