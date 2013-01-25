import os,sys
import h5py
import numpy as np
import scipy as sp
import pylab as pl
import scipy.signal as signal
#from debleach import exponential_basis_debleach
from scipy.stats import ranksums
from scipy.interpolate import UnivariateSpline
import tkFileDialog

#from wavelet import *
#-----------------------------------------------------------------------------------------

class FiberAnalyze( object ):

    def __init__(self, options):
        """
        Initialize the FiberAnalyze class using options object from OptionsParser.
        """
        # values from option parser
        self.smoothness = options.smoothness
        self.plot_type = options.plot_type
        self.time_range = options.time_range
        self.fluor_normalization = options.fluor_normalization

        if options.selectfiles:
            self.input_path = tkFileDialog.askopenfilename()
            self.output_path = self.input_path[:-4] + '_out'
            self.trigger_path = tkFileDialog.askopenfilename()
        else:
            self.input_path = options.input_path
            self.output_path = options.output_path
            print self.output_path
            self.trigger_path = options.trigger_path

        self.save_txt = options.save_txt
        self.save_to_h5 = options.save_to_h5
        self.save_and_exit = options.save_and_exit
        
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
        print self.input_path
        self.data = np.load( self.input_path )['data']
        self.time_stamps = np.load( self.input_path )['time_stamps']

        # normalized fluor data, which is in arbitrary units
        self.fluor_data = self.data[:,self.fluor_channel]

    

        if self.fluor_normalization == "deltaF":
            median = np.median(self.fluor_data)
            self.fluor_data = (self.fluor_data-median)/median #dF/F
        else:
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
        if self.fluor_normalization == "deltaF":
            self.trigger_data *= 1.5*np.max(self.fluor_data)

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
        print "\t--> Finished loading data."

        if self.save_txt:
            self.save_time_series(self.output_path)
        if self.save_to_h5 is not None:
            self.save_time_series(self.output_path, output_type="h5", h5_filename=self.save_to_h5)

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
        # clear figure
        pl.clf()

        # get appropriate time values for x-axis
        time_vals = self.time_stamps[range(len(self.fluor_data))]

        # make filled blocks for trigger onset/offset
        ymax = 1.1*np.max(self.fluor_data)
        ymin = 1.1*np.min(self.fluor_data)
        pl.fill( time_vals, self.trigger_data, facecolor='r', alpha=0.5 )
        pl.plot( time_vals, self.fluor_data, 'k-')
        pl.ylim([ymin,ymax])
        if self.fluor_normalization == "deltaF":
            pl.ylabel(r'$\delta F/F$')
        else:
            pl.ylabel('Fluorescence Intensity (a.u.)')
        pl.xlabel('Time since recording onset (seconds)')
        if out_path is None:
            pl.show()
            out_path = '.'
        else:
           # pl.savefig(os.path.join(out_path,"basic_time_series.pdf"))
            pl.savefig(out_path + "basic_time_series.pdf")
            pl.savefig(out_path + "basic_time_series.png")

    def save_time_series( self, save_path='.', output_type="txt", h5_filename=None ):
        """
        Save the raw calcium time series, with triggers corresponding to events
        (e.g. licking times for sucrose, approach/interaction times for novel object
        and social) in the same time frame of reference, outputting either
          -- a txt file, if ouput_type is txt
          -- write to an hdf5 file specified by self.save_to_h5 is output_type is h5
        """
        # get appropriate time values 
        time_vals = self.time_stamps[range(len(self.fluor_data))]

        # make output array
        out_arr = np.zeros((len(time_vals),3))
        out_arr[:,0] = time_vals
        out_arr[:,1] = self.trigger_data
        out_arr[:,2] = self.fluor_data

        # get data prefix
        prefix=self.input_path.split("/")[-1].split(".")[0]
        outfile_name = prefix+"_tseries.txt"
        out_path = os.path.join(save_path,outfile_name)

        if output_type == "txt":
            print "Saving to file:", out_path
            np.savetxt(os.path.join(out_path), out_arr)
            if self.save_and_exit:
                sys.exit(0)
            
        elif output_type == "h5":
            print "\t--> Writing to HDF5 file", self.save_to_h5
            # check if specified h5 file already exists
            h5_exists = os.path.isfile(self.save_to_h5)
            try:
                if h5_exists:
                    # write to existing h5 file
                    h5_file = h5py.File(self.save_to_h5)
                    print "\t--> Writing to exising  HDF5 file:", self.save_to_h5
                else:
                    # create new h5 file
                    h5_file = h5py.File(self.save_to_h5,'w')
                    print "\t--> Created new HDF5 file:", self.save_to_h5
            except Exception, e:
                print "Unable to open HDF5 file", self.save_to_h5, "due to error:"
                print e

            # save output array to folder in h5 file creating a data set named after the subject number
            # with columns corresponding to time, triggers, and fluorescence data, respectively.

            # group by animal number, subgroup by date, subsubgroup by run type
            if prefix.split("-")[3] not in list(h5_file):
                print "\t---> Creating group:", prefix.split("-")[3]
                subject_num= h5_file.create_group(prefix.split("-")[3])
            else:
                print "\t---> Loading group:", prefix.split("-")[3]
                subject_num = h5_file[prefix.split("-")[3]]
                
            if prefix.split("-")[0] not in list(subject_num):
                print "\t---> Creating subgroup:", prefix.split("-")[0]
                date = subject_num.create_group(prefix.split("-")[0])
            else:
                print "\t---> Loading subgroup:", prefix.split("-")[0]
                date = subject_num[prefix.split("-")[0]]
            try:
                print "\t---> Creating subsubgroup:", prefix.split("-")[2]
                run_type = date.create_group(prefix.split("-")[2])
                dset = run_type.create_dataset("time_series_arr", data=out_arr)
                dset.attrs["time_series_arr_names"] = ("time_stamp", "trigger_data", "fluor_data")
                dset.attrs["original_file_name"] = prefix
            except Exception, e:
                print "Unable to write data array due to error:", e
                
            h5_file.close() # close the file
            
            if self.save_and_exit:
                sys.exit(0)
        else:
            raise NotImplemented("The entered output_type has not been implemented.")
                
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
           # pl.savefig(os.path.join(out_path,"event_vs_baseline_barplot"))
            pl.savefig(out_path + "event_vs_baseline_barplot.pdf")
        
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
        print "get_peaks"
        peak_widths = np.array([100,250,500,750,1000])
        self.peak_inds = signal.find_peaks_cwt(self.fluor_data, widths=peak_widths, wavelet=None, max_distances=None, gap_thresh=None, min_length=None, min_snr=5, noise_perc=50)
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

        #plot 
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
            #pl.axis([1, 100, 0, 1.1*np.log(np.max(Y[start_freq:num_values]))])
            pl.axis([1, 100, -5, 10])
        else:
            print "Currently only 'standard' and 'log' plot types are available"

        if out_path is None:
            pl.show()
        else:
            print "Saving periodogram..."
          #  pl.savefig(os.path.join(out_path,'periodogram'))
            pl.savefig(out_path + "periodogram.pdf")

        
    def plot_peak_data( self, out_path=None ):
        """
        Plot fluorescent data with chosen peak_inds overlayed as lines.
        """
        print "plot_peak_data"
        pl.clf()
        lines = np.zeros(len(self.fluor_data))
        lines[self.peak_inds] = 1.0
        pl.plot(self.fluor_data)
        pl.plot(lines)
        if self.fluor_normalization == "deltaF":
            pl.ylabel(r'$\delta F/F$')
        else:
            pl.ylabel('Fluorescence Intensity (a.u.)')
        pl.xlabel('Time (samples)')

        if out_path is None:
            pl.show()
        else:
          #  pl.savefig(os.path.join(out_path,'peak_finding'))
            pl.savefig(out_path + "peak_finding.pdf")


    def get_time_chunks_around_events(self, event_times, window_size):
        """
        Extracts chunks of fluorescence data around each event in 
        event_times, with before and after event durations
        specified in window_size as [before, after] (in seconds).
        """

        window_indices = [self.convert_seconds_to_index( window_size[0]),
                          self.convert_seconds_to_index( window_size[1])]

        time_chunks = []
        for e in event_times:
            try:
                e_idx = np.where(e<self.time_stamps)[0][0]
                chunk = self.fluor_data[range((e_idx-window_indices[0]),(e_idx+window_indices[1]))]
                #print [range((e_idx-window_indices[0]),(e_idx+window_indices[1]))]
                time_chunks.append(chunk)
            except:
                print "Unable to extract window:", [(e-window_indices[0]),(e+window_indices[1])]
        return time_chunks


    def plot_perievent_hist( self, event_times, window_size, out_path=None ):
        """
        Peri-event time histogram for given event times.
        Plots the time series and their median over a time window around
        each event in event_times, with before and after event durations
        specified in window_size as [before, after] (in seconds).
        """
        # new figure
        pl.clf()
        fig = pl.figure()
        ax = fig.add_subplot(111)

        # get blocks of time series for window around each event time
        time_chunks = self.get_time_chunks_around_events(event_times, window_size)

        window_indices = [ self.convert_seconds_to_index(window_size[0]),
                           self.convert_seconds_to_index(window_size[1]) ]

        # plot each time window, colored by order
        time_arr = np.asarray(time_chunks).T
        x = self.time_stamps[0:time_arr.shape[0]]-self.time_stamps[window_indices[0]] ###IS THIS RIGHT?
        ymax = np.max(time_arr)
        ymax += 0.1*ymax
        ymin = np.min(time_arr)
        ymin -= 0.1*ymin
        for i in xrange(time_arr.shape[1]):
            ax.plot(x, time_arr[:,i], color=pl.cm.winter(255-255*i/time_arr.shape[1]), alpha=0.75, linewidth=1)
            x.shape = (len(x),1) 
            x_padded = np.vstack([x[0], x, x[-1]])
            time_vec = time_arr[:,i]; time_vec.shape = (len(time_vec),1)
            time_vec_padded = np.vstack([0, time_vec,0]) 
            pl.fill(x_padded, time_vec_padded, facecolor=pl.cm.winter(255-255*i/time_arr.shape[1]), alpha=0.25 )            
            #pl.ylim([0,ymax])
            pl.ylim([ymin, ymax])
            
        # add a line for the event onset time
        pl.axvline(x=0,color='black',linewidth=1,linestyle='--')

        # label the plot axes
        if self.fluor_normalization == "deltaF":
            pl.ylabel(r'$\delta F/F$')
        else:
            pl.ylabel('Fluorescence Intensity (a.u.)')
        pl.xlabel('Time from onset of social bout (seconds)')

        # show plot now or save of an output path was specified
        if out_path is None:
            pl.show()
        else:
            print "Saving peri-event time series..."
            #pl.savefig(os.path.join(out_path,'perievent_tseries'))
            pl.savefig(out_path + "perievent_tseries.pdf")


    def plot_peritrigger_edge( self, window_size, edge="rising", out_path=None ):
        """
        Wrapper for plot_perievent histograms specialized for
        loaded event data that comes as a list of pairs of
        event start and end times.
        """
        event_times = self.get_event_times(edge)
        if event_times != -1:
            self.plot_perievent_hist( event_times, window_size, out_path=out_path )
        else:
            print "No event times loaded. Cannot plot perievent."        

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


    def get_event_times( self, edge="rising"):
        """
        Extracts a list of the times (in seconds) corresponding to
        interaction events to be time-locked with the signal
        specialized for loaded event data that comes as a list of 
        pairs of event start and end times.
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
            return event_times
        else:
            print "No event times loaded. Cannot find edges."        
            return -1

    def convert_seconds_to_index( self, time_in_seconds):
        return np.where( self.time_stamps >= time_in_seconds)[0][0]

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

    def get_areas_under_curve( self, start_times, window_size, normalize=False):
        """
        Returns a vector of the area under the fluorescence curve within the provided
        window [before, after] (in seconds), that surrounds each start_time.
        Normalize determines whether to divide the area by the maximum fluorescence
        value of the window
        """
        time_chunks = self.get_time_chunks_around_events(start_times, window_size)
        
        areas = []
        for chunk in time_chunks:
            if normalize:
                if max(chunk) < 0.01: 
                    areas.append(sum(chunk)/len(chunk)/0.01)
                else:
                    areas.append(sum(chunk)/len(chunk)/(max(abs(chunk))))
            else: 
                areas.append(sum(chunk)/len(chunk))

        return areas
            

    def plot_area_under_curve( self, start_times, end_times, window_size, normalize=False, out_path=None):
        """
        Plots of area under curve for each event_time 
        with before and after event durationsspecified in window_size as 
        [before, after] (in seconds).
        -- choosing the window around the event onset is still somewhat arbitrary, 
        we need to discuss how to choose this well...
        """
        
        #change the normalization depending on whether you wish to divide (normalize)
        # by the maximum fluorescence value in the window following
        # each event

        areas = self.get_areas_under_curve(start_times, window_size)


        window_indices = [ self.convert_seconds_to_index(window_size[0]),
                           self.convert_seconds_to_index(window_size[1]) ]
        #Plot the area vs the time of each event
        pl.clf()
        ymax = 1.1*np.max(areas) + 0.1
        ymin = 1.1*np.min(areas) - 0.1
        pl.stem( start_times, areas, linefmt='k-', markerfmt='ko', basefmt='k-')
        pl.plot([0, start_times[-1]], [0, 0], 'k-')
        pl.ylim([ymin,ymax])
        pl.xlim([0, np.max(self.time_stamps)])

        if self.fluor_normalization == "deltaF":
            if normalize:
                pl.ylabel('Sharpness of peak: ' r'$\frac{\sum\delta F/F}{\max(peak)}}$' + ' with window of ' + "{0:.2f}".format(self.time_stamps[window_indices[1]]) + ' s')
            else:
                pl.ylabel('Sharpness of peak: ' r'$\sum\delta F/F}$' + ' with window of ' + "{0:.2f}".format(self.time_stamps[window_indices[1]]) + ' s')
        else:
            pl.ylabel('Fluorescence Intensity (a.u.) integrated over window of ' + "0:.2f}".format(self.time_stamps[window_indices[1]]) + ' s')
        pl.xlabel('Time in trial (seconds)')
        pl.title(out_path)

        if out_path is None:
            pl.title("No output path given")
            pl.show()
        else:
            if normalize:
                pl.savefig(out_path + "plot_area_under_curve_normal" + str(int(10*self.time_stamps[window_indices[1]])) + "s.pdf")
                np.savez(out_path + "normalized_area_under_peaks_" + str(int(10*self.time_stamps[window_indices[1]])) + "s.npz", scores=areas, event_times=start_times, end_times=end_times, window_size=self.time_stamps[window_indices[1]])
                # Assume not using windows longer than a few seconds. Thus save the file so as to display one decimal point
            else:
                pl.savefig(out_path + "plot_area_under_curve_non_normal" + str(int(10*self.time_stamps[window_indices[1]])) + "s.pdf")
                np.savez(out_path + "non-normalized_area_under_peaks_" + str(int(10*self.time_stamps[window_indices[1]]))+ "s.npz", scores=areas, event_times=start_times, end_times=end_times, window_size=self.time_stamps[window_indices[1]])


    def plot_area_under_curve_wrapper( self, window_size, edge="rising", normalize=True, out_path=None):
        """
        Wrapper for plot_area_under_curve vs. time plots specialized for
        loaded event data that comes as a list of pairs of
        event start and end times (in seconds)
        """

        start_times = self.get_event_times("rising")
        end_times = self.get_event_times("falling")
        if start_times != -1:
            self.plot_area_under_curve( start_times, end_times, window_size, normalize, out_path=out_path )
        else:
            print "No event times loaded. Cannot plot perievent."  


    def get_peak( self, start_time, end_time ):
        """
        Return the maximum fluorescence value found between
        start_time and end_time (in seconds)
        """
        start_time_index = self.convert_seconds_to_index(start_time)
        end_time_index = self.convert_seconds_to_index(end_time)
        if start_time_index < end_time_index:
            return np.max(self.fluor_data[start_time_index : end_time_index])
        else:
            return 0

    def eNegX(self, p, x):
        x0, y0, c, k=p
        #Set c=1 to normalize all of the trials, since we
        # are only interested in the rate of decay
        y = (1 * np.exp(-k*(x-x0))) + y0
        return y

    def eNegX_residuals(self, p, x, y):
        return y - self.eNegX(p, x)

    def fit_exponential(self, x, y):
        # Because we are optimizing over a nonlinear function
        # choose a number of possible starting values of (x0, y0, c, k)
        # and use the results from whichever produces the smallest 
        # residual
        kguess = [0, 0.1, 0.5, 1.0, 100, 500, 1000]
        max_r2 = -1
        maxvalues = ()
        for kg in kguess:
            p_guess=(np.min(x), 0, 1, kg)
            p, cov, infodict, mesg, ier = sp.optimize.leastsq(
                self.eNegX_residuals, p_guess, args=(x, y), full_output=1)

            x0,y0,c,k=p 
            print('''Reference data:\  
                    x0 = {x0}
                    y0 = {y0}
                    c = {c}
                    k = {k}
                    '''.format(x0=x0,y0=y0,c=c,k=k))

            numPoints = np.floor((np.max(x) - np.min(x))*100)
            xp = np.linspace(np.min(x), np.max(x), numPoints)
            #pxp = np.exp(-1*xp)
            pxp = self.eNegX(p, xp)
            yxp = self.eNegX(p, x)

            sstot = np.sum(np.multiply(y - np.mean(y), y - np.mean(y)))
            sserr = np.sum(np.multiply(y - yxp, y - yxp))
            r2 = 1 - sserr/sstot
            if max_r2 == -1:
                maxvalues = (xp, pxp, x0, y0, c, k, r2)
            if r2 > max_r2:
                max_r2 = r2
                maxvalues = (xp, pxp, x0, y0, c, k, r2)

        return maxvalues


    def plot_peaks_vs_time( self, out_path=None ):
        """
        Plot the maximum fluorescence value within each interaction event vs the start time
        of the event
        """

        start_times = self.get_event_times("rising")
        end_times = self.get_event_times("falling")
        if start_times != -1:

            peaks = np.zeros(len(start_times))
            for i in range(len(start_times)):
                peak = self.get_peak(start_times[i], end_times[i])
                peaks[i] = peak


            fig = pl.figure()
            ax = fig.add_subplot(111)
            print np.max(peaks) + .3
            if np.max(peaks) > 0.8:
                ax.set_ylim([0, 1.3])
            else:
                ax.set_ylim([0, 1.1])

            ax.set_xlim([100, 500])
            ax.plot(start_times, peaks, 'o')
            pl.xlabel('Time [s]')
            pl.ylabel('Fluorescence [dF/F]')
            pl.title('Peak fluorescence of interaction event vs. event start time')


            try:
                xp, pxp, x0, y0, c, k, r2 = self.fit_exponential(start_times, peaks + 1)
                ax.plot(xp, pxp-1)
                ax.text(min(200, np.min(start_times)), np.max(peaks) + 0.20, "y = c*exp(-k*(x-x0)) + y0")
                ax.text(min(200, np.min(start_times)), np.max(peaks) + 0.15, "k = " + "{0:.2f}".format(k) + ", c = " + "{0:.2f}".format(c) + 
                                                ", x0 = " + "{0:.2f}".format(x0) + ", y0 = " + "{0:.2f}".format(y0) )
                ax.text(min(200, np.min(start_times)), np.max(peaks) + 0.1, "r^2 = " + str(r2))
            except:
                print "Exponential Curve fit did not work"



            if out_path is None:
                pl.title("No output path given")
                pl.show()
            else:
                pl.savefig(out_path + "plot_peaks_vs_time.pdf")
                np.savez(out_path + "peaks_vs_time.npz", scores=peaks, event_times=start_times, end_times=end_times, window_size=0)

        else:
            print "No event times loaded. Cannot plot peaks_vs_time."  


    def wavelet_plot( self ):
        wavelet=Morlet
        maxscale=4
        notes=16
        scaling="log" #or "linear"
        #scaling="linear"
        plotpower2d=True

        # set up some data
        Ns=1024
        #limits of analysis
        Nlo=0 
        Nhi=Ns
        # sinusoids of two periods, 128 and 32.
        x = self.time_stamps
        A = self.fluor_data

        # Wavelet transform the data
        cw=wavelet(A,maxscale,notes,scaling=scaling)
        scales=cw.getscales()     
        cwt=cw.getdata()
        # power spectrum
        pwr=cw.getpower()
        scalespec=np.sum(pwr,axis=1)/scales # calculate scale spectrum
        # scales
        y=cw.fourierwl*scales
        x=np.arange(Nlo*1.0,Nhi*1.0,1.0)

        fig=pl.figure(1)

        # 2-d coefficient plot
        ax=pl.axes([0.4,0.1,0.55,0.4])
        pl.xlabel('Time [s]')
        plotcwt=np.clip(np.fabs(cwt.real), 0., 1000.)
        if plotpower2d: plotcwt=pwr
        im=pl.imshow(plotcwt,cmap=pl.cm.jet,extent=[x[0],x[-1],y[-1],y[0]],aspect='auto')
        #colorbar()
        if scaling=="log": ax.set_yscale('log')
        pl.ylim(y[0],y[-1])
        ax.xaxis.set_ticks(np.arange(Nlo*1.0,(Nhi+1)*1.0,100.0))
        ax.yaxis.set_ticklabels(["",""])
        theposition=pl.gca().get_position()

        # data plot
        ax2=pl.axes([0.4,0.54,0.55,0.3])
        pl.ylabel('Data')
        pos=ax.get_position()
        pl.plot(x,A,'b-')
        pl.xlim(Nlo*1.0,Nhi*1.0)
        ax2.xaxis.set_ticklabels(["",""])
        pl.text(0.5,0.9,"Wavelet example with extra panes",
             fontsize=14,bbox=dict(facecolor='green',alpha=0.2),
             transform = fig.transFigure,horizontalalignment='center')

        # projected power spectrum
        ax3=pl.axes([0.08,0.1,0.29,0.4])
        pl.xlabel('Power')
        pl.ylabel('Period [s]')
        vara=1.0
        if scaling=="log":
            pl.loglog(scalespec/vara+0.01,y,'b-')
        else:
            pl.semilogx(scalespec/vara+0.01,y,'b-')
        pl.ylim(y[0],y[-1])
        pl.xlim(1000.0,0.01)

        pl.show()

#-----------------------------------------------------------------------------------------

def get_event_window( event_ts):

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
#    FA.wavelet_plot()
#    FA.notch_filter(10.0, 10.3)
  #  FA.plot_periodogram(plot_type="log",out_path = options.output_path)
  #  FA.plot_basic_tseries(out_path = options.output_path)
#    FA.event_vs_baseline_barplot(out_path = options.output_path)

    #FA.plot_peritrigger_edge(window_size=[1, 3],out_path = options.output_path)
    FA.plot_area_under_curve_wrapper( window_size=[0, 1], edge="rising", normalize=False, out_path = options.output_path)
    #FA.plot_peaks_vs_time(out_path = options.output_path)

    #peak_inds, peak_vals, peak_times = FA.get_peaks()
    #FA.plot_peak_data()

#-----------------------------------------------------------------------------------------

if __name__ == "__main__":

    # Parse command line options
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-o", "--output-path", dest="output_path", default=None,
                      help="Specify the ouput path.")
    parser.add_option("-t", "--trigger-path", dest="trigger_path", default=None,
                      help="Specify path to files with trigger times, minus the '_s.npz' and '_e.npz' suffixes.")
    parser.add_option("-i", "--input-path", dest="input_path",
                      help="Specify the input path.")
    parser.add_option("", "--time-range", dest="time_range",default=None,
                      help="Specify a time window over which to analyze the time series in format start:end. -1 chooses the appropriate extremum")
    parser.add_option('-p', "--plot-type", default = 'tseries', dest="plot_type",
                      help="Type of plot to produce.")
    parser.add_option('', "--fluor_normalization", default = 'deltaF', dest="fluor_normalization",
                      help="Normalization of fluorescence trace. Can be a.u. between [0,1]: 'stardardize' or deltaF/F: 'deltaF'.")
    parser.add_option('-s', "--smoothness", default = None, dest="smoothness",
                      help="Should the time series be smoothed, and how much.")
    parser.add_option('-x', "--selectfiles", default = False, dest = "selectfiles",
                       help="Should you select filepaths in a pop window instead of in the command line.")
    parser.add_option("", "--save-txt", action="store_true", default=False, dest="save_txt",
                      help="Save data matrix out to a text file.")
    parser.add_option("", "--save-to-h5", default=None, dest="save_to_h5",
                      help="Save data matrix to a dataset in an hdf5 file.")
    parser.add_option("", "--save-and-exit", action="store_true", default=False, dest="save_and_exit",
                      help="Exit immediately after saving data out.")

    (options, args) = parser.parse_args()
    
    # Test the class
    test_FiberAnalyze(options)
    
# EOF
