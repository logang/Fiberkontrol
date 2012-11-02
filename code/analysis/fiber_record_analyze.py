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
        self.trigger_path = options.trigger_path

        if self.trigger_path is not None:
            self.s_file = options.trigger_path + '_s.npz'
            self.e_file = options.trigger_path + '_e.npz'
        else:
            self.trigger_path= None
            
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
        if self.trigger_path is None:
            self.trigger_data = self.data[:,self.trigger_channel]
            self.time_tuples = None
        else:
            try:
                self.time_tuples = self.load_trigger_data(self.s_file, self.e_file)
                time_vec = np.asarray(self.time_tuples).flatten()
                time_vec = np.append(time_vec,np.inf)
                self.trigger_data = np.zeros(len(self.fluor_data))
                j = 0
                for i in xrange(len(self.trigger_data)):
                    if self.time_stamps[i] < time_vec[j]:
                        self.trigger_data[i] = np.mod(j,2)
                    else:
                        j+=1
                        self.trigger_data[i] = np.mod(j,2)
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
            if tlist[0] == '-1':
                self.t_start = 0
            if tlist[1] == '-1':
                self.t_end = len(self.fluor_data)

            if len(tlist) != 2:
                print 'Error parsing --time-range argument.  Be sure to use <start-time>:<end-time> syntax.'
                sys.exit(1)
            self.t_start = int(tlist[0])
            self.t_end   = int(tlist[1])
            self.fluor_data = self.fluor_data[self.t_start:self.t_end]
            self.trigger_data = self.trigger_data[self.t_start:self.t_end]
            self.time_stamps = self.time_stamps[self.t_start:self.t_end]

        self.fft = None #place holder for fft of rawsignal, if calculated
        self.fft_freq = None #place holder for frequency labels of fft
        self.filt_fluor_data = None #place holder for filtered rawsignal, if calculated

    def load_trigger_data( self, s_filename, e_filename ):
        """
        Load start and end times for coded events. 
        """
        self.s_vals = np.load(s_filename)['arr_0']
        self.e_vals = np.load(e_filename)['arr_0']
        return zip(self.s_vals,self.e_vals)

    def plot_basic_tseries( self, out_path=None ):
        """
        Generate a plot showing the raw calcium time series, with triggers
        corresponding to events (e.g. licking for sucrose) superimposed.
        """
        pl.clf()
        time_vals = self.time_stamps[range(len(self.fluor_data))]
        pl.fill( time_vals, self.trigger_data, facecolor='r', alpha=0.5 )
        pl.plot( time_vals, self.fluor_data, 'k-')
        pl.ylabel('Fluorescence Intensity (a.u.)')
        pl.xlabel('Time (seconds)')
        if out_path is None:
            pl.show()
        else:
            pl.savefig(os.path.join(out_path,"basic_time_series"))

    def event_vs_baseline_barplot( self, out_path=None ):
        """
        Make a simple barplot of intensity during coded events vs during non-event times.
        """
        pl.clf()
        event = self.trigger_data*self.fluor_data
        baseline = self.fluor_data[ 0:(self.time_tuples[0][0]-self.t_start) ]
        pl.boxplot([event,baseline])
        if out_path is None:
            pl.show()
        else:
            pl.savefig(os.path.join(out_path,"event_vs_baseline_barplot"))
        
    def low_pass_filter(self, cutoff):
        """
        Low pass filter the data with frequency cutoff: 'cutoff'.
        """

        if self.filt_fluor_data is None:
            rawsignal = self.fluor_data
        else:
            rawsignal = self.filt_fluor_data

        if self.fft is None:
            self.get_fft()
        bp = self.fft

        for i in range(len(bp)):
            if self.fft_freq[i]>= cutoff: bp[i] = 0
        self.fft = bp #update fft of the class to the filtered version

        ibp = sp.ifft(bp)
        low_pass_y = np.real(ibp)
        low_pass_y += np.median(self.fluor_data) - np.median(low_pass_y)
        self.filt_fluor_data = low_pass_y
        return low_pass_y

    def get_peaks( self ):
        """
        Heuristic for finding local peaks in the calcium data. 
        """
        peak_widths = np.array([100,250,500,750,1000])
        self.peak_inds = signal.find_peaks_cwt(self.fluor_data, widths=peak_widths, wavelet=None, max_distances=None, gap_thresh=None, min_length=None, min_snr=5, noise_perc=20)
        self.peak_vals = self.fluor_data[self.peak_inds]
        self.peak_times = self.time_stamps[self.peak_inds]
        return self.peak_inds, self.peak_vals, self.peak_times

    def notch_filter(self, low, high):
        """
        Notch filter that cuts out frequencies (Hz) between [low:high].
        """
        if self.filt_fluor_data is None:
            rawsignal = self.fluor_data
        else:
            rawsignal = self.filt_fluor_data

        if self.fft is None:
            self.get_fft()
        bp = self.fft

        for i in range(len(bp)):
            if self.fft_freq[i]<= high and self.fft_freq[i]>= low:
                bp[i] = 0
        self.fft = bp #update fft of the class to the filtered version

        ibp = sp.ifft(bp)
        notch_filt_y = np.real(ibp)
        notch_filt_y += np.median(self.fluor_data) - np.median(notch_filt_y)
        self.filt_fluor_data = notch_filt_y
        return notch_filt_y

    def plot_periodogram( self, out_path = None, plot_type="log", window_len = 20):
        """
        Plot periodogram of fluoroscence data.
        """
        print "Plotting periodogram"
        pl.clf()
        
        if self.filt_fluor_data is None:
            rawsignal = self.fluor_data
        else:
            rawsignal = self.filt_fluor_data
            
        if self.fft is None:
            self.get_fft()

        Y = np.abs(self.fft)**2 #power spectral density
        freq = self.fft_freq
        
        ## Smooth spectral density 
        s = np.r_[Y[window_len-1:0:-1],Y,Y[-1:-window_len:-1]] #periodic boundary
        w = np.bartlett(window_len)
        Y = np.convolve(w/w.sum(), s, mode='valid')
                                     
        num_values = int(min(len(Y), len(freq))*.5) #Cut off negative frequencies
        start_freq = 0.25*(num_values/100) #i.e. start at 0.25 Hz
        freq_vals = freq[range(num_values)]
        Y = Y[range(num_values)]

        pl.figure()
        if plot_type == "standard":
            pl.plot( freq_vals[start_freq:num_values], Y[start_freq:num_values], 'k-')
            pl.ylabel('Spectral Density (a.u.)')
            pl.xlabel('Frequency (Hz)')
            pl.title(self.input_path)
            pl.axis([1, 100, 0, 1.1*np.max(Y[start_freq:num_values])])
        elif plot_type == "log":
            pl.plot( freq_vals[start_freq:num_values], np.log(Y[start_freq:num_values]), 'k-')
            pl.ylabel('Log(Spectral Density) (a.u.)')
            pl.xlabel('Frequency (Hz)')
            pl.title(self.input_path)
            pl.axis([1, 100, 0, 1.1*np.log(np.max(Y[start_freq:num_values]))])
        else:
            print "Currently only 'standard' and 'log' plot types are available"

        if out_path is None:
            pl.show()
        else:
            print "Saving periodogram..."
            pl.savefig(os.path.join(out_path,'periodogram'))
        
    def plot_peak_data( self, out_path=None ):
        """
        Plot fluorescent data with chosen peak_inds overlayed as lines.
        """
        pl.clf()
        lines = np.zeros(len(self.fluor_data))
        lines[self.peak_inds] = 1.0
        pl.plot(self.fluor_data)
        pl.plot(lines)
        pl.ylabel('Fluorescence Intensity (a.u.)')
        pl.xlabel('Time (samples)')

        if out_path is None:
            pl.show()
        else:
            pl.savefig(os.path.join(out_path,'peak_finding'))

    def plot_perievent_hist( self, event_times, window_size, out_path=None ):
        """
        Peri-event time histogram for given event times.
        Plots the time series and their median over a time window around
        each event in event_times, with before and after event durations
        specified in window_size as [before, after] (in seconds).
        """
        pl.clf()
        time_chunks = []
        for e in event_times:
            try:
                e_idx = np.where(e<self.time_stamps)[0][0]
                chunk = self.fluor_data[range((e_idx-window_size[0]),(e_idx+window_size[1]))]
                print [range((e_idx-window_size[0]),(e_idx+window_size[1]))]
                time_chunks.append(chunk)
            except:
                print "Unable to extract window:", [(e-window_size[0]),(e+window_size[1])]
        time_arr = np.asarray(time_chunks).T
        x = self.time_stamps[0:time_arr.shape[0]]-self.time_stamps[window_size[0]]
        ymax = np.max(time_arr)
        ymax += 0.1*ymax
        for i in xrange(time_arr.shape[1]):
            pl.plot(x, time_arr[:,i], color=pl.cm.jet(255-255*i/time_arr.shape[1]), alpha=0.75, linewidth=2)
            pl.ylim([0,ymax])
        pl.axvline(x=0,color='black',linewidth=2,linestyle='--')
        pl.ylabel('Fluorescence Intensity (a.u.)')
        pl.xlabel('Time from event (seconds)')
        
        if out_path is None:
            pl.show()
        else:
            print "Saving peri-event time series..."
            pl.savefig(os.path.join(out_path,'perievent_tseries'))

    def plot_peritrigger_edge( self, window_size, edge="rising", out_path=None ):
        """
        Wrapper for plot_perievent histograms specialized for
        loaded event data that comes as a list of pairs of
        event start and end times.
        """
        if self.time_tuples is not None:
            event_times = []
            for pair in self.time_tuples:
                if edge == "rising":
                    event_times.append(pair[0])
                elif edge == "falling":
                    event_times.append(pair[1])
                else:
                    raise ValueError("Edge type must be 'rising' or 'falling'.")
            self.plot_perievent_hist( event_times, window_size, out_path=out_path )
        else:
            print "No event times loaded. Cannot find edges."        

    def get_fft(self):
        if self.filt_fluor_data is None:
            rawsignal = self.fluor_data
        else:
            rawsignal = self.filt_fluor_data
        
        fft = sp.fft(rawsignal)
        self.fft = fft[:]

        n = rawsignal.size
        timestep = np.max(self.time_stamps[1:] - self.time_stamps[:-1])
        self.fft_freq = np.fft.fftfreq(n, d=timestep)

    # --- not yet implemented --- #

    def debleach( self ):
        """
        Remove trend from data due to photobleaching.
        Is this necessary?
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
    
    FA.notch_filter(10.0, 10.3)
#    FA.plot_periodogram(plot_type="log",out_path = options.output_path)
    FA.plot_basic_tseries(out_path = options.output_path)
#    FA.event_vs_baseline_barplot(out_path = options.output_path)
#    peak_inds, peak_vals, peak_times = FA.get_peaks()
    FA.plot_peritrigger_edge(window_size=[100,1000],out_path = options.output_path)
#    FA.plot_peak_data()

    1/0

#-----------------------------------------------------------------------------------------

if __name__ == "__main__":

    # Parse command line options
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-o", "--output-path", dest="output_path", default=None,
                      help="Specify the ouput path.")
    parser.add_option("", "--trigger-path", dest="trigger_path", default=None,
                      help="Specify path to files with trigger times, minus the '_s.npz' and '_e.npz' suffixes.")
    parser.add_option("-i", "--input-path", dest="input_path",
                      help="Specify the input path.")
    parser.add_option("", "--time-range", dest="time_range",default=None,
                      help="Specify a time window over which to analyze the time series in format start:end. -1 chooses the appropriate extremum")
    parser.add_option('-p', "--plot-type", default = 'tseries', dest="plot_type",
                      help="Type of plot to produce.")
    parser.add_option('-s', "--smoothness", default = None, dest="smoothness",
                      help="Should the time series be smoothed, and how much.")

    (options, args) = parser.parse_args()
    
    # Test the class
    test_FiberAnalyze(options)
    
# EOF
