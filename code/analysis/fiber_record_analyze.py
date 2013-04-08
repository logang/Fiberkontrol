import os,sys
import h5py
import numpy as np
import scipy as sp
import pylab as pl
import scipy.signal as signal
import scipy.stats as ss
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
        # attributes to set for hdf5 loading. Can use these to specify individual trials when using the FiberAnalyze class
        # in another program
        self.subject_id = None
        self.exp_date = None
        self.exp_type = None

        # values from option parser
        self.smoothness = int(options.smoothness)
        self.plot_type = options.plot_type
        self.time_range = options.time_range
        self.fluor_normalization = options.fluor_normalization
        self.filter_freqs = options.filter_freqs
        self.exp_type = options.exp_type
        self.event_spacing = float(options.event_spacing)
        self.mouse_type = options.mouse_type


        if options.selectfiles:
            self.input_path = tkFileDialog.askopenfilename()
            self.output_path = self.input_path[:-4] + '_out'
            self.trigger_path = tkFileDialog.askopenfilename()
        else:
            self.input_path = options.input_path
            self.output_path = options.output_path
            self.trigger_path = options.trigger_path

        self.save_txt = options.save_txt
        self.save_to_h5 = options.save_to_h5
        self.save_and_exit = options.save_and_exit
        self.save_debleach = options.save_debleach
        
        if self.trigger_path is not None:
            self.s_file = options.trigger_path + '_s.npz'
            self.e_file = options.trigger_path + '_e.npz'
        else:
            self.trigger_path= None
            
        # hard coded values
        self.fluor_channel = 0
        self.trigger_channel = 3


        self.fft = None #place holder for fft of rawsignal, if calculated
        self.fft_freq = None #place holder for frequency labels of fft
        self.filt_fluor_data = None #place holder for filtered rawsignal, if calculated
        self.event_start_times = None #place holder event_times that may be calculated by get_sucrose_event_times
        self.event_end_times = None

        
    def load( self, file_type="npz" ):
        """
        Load time series and events from NPZ or HDF5 file. 
        """
        self.time_tuples = None
        if file_type == "npz":
            print ""
            print "--> Loading: ", self.input_path
            self.data = np.load( self.input_path )['data']
            self.time_stamps = np.load( self.input_path )['time_stamps']
            self.fluor_data = self.data[:,self.fluor_channel]
            self.load_trigger_data()
            
        elif file_type == "hdf5":
            print "hdf5 file to load: ", self.subject_id, self.exp_date, self.exp_type
            #try:
            h5_file = h5py.File( self.input_path, 'r' )
            self.data = np.asarray( h5_file[self.subject_id][self.exp_date][self.exp_type]['time_series_arr'] )
            if self.exp_type != 'sucrose':
                self.time_tuples = np.asarray( h5_file[self.subject_id][self.exp_date][self.exp_type]['event_tuples'] )
            else:
                self.time_typles = None

            self.time_stamps = self.data[:,0]
            self.trigger_data = self.data[:,1]
            
            load_flat = True
            if (load_flat):
                try:
                    self.fluor_data = np.asarray( h5_file[self.subject_id][self.exp_date][self.exp_type]['flat'] )[:, 0]
                    print "--> Loading flattened data"
                except:
                    print "--> Flattened data UNAVAILABLE"
                    self.fluor_data = self.data[:,2] 
            else:
                self.fluor_data = self.data[:,2] #to use unflattened, original data
                print "--> Loading UNFLATTENED data"

            # except Exception, e:
            #     print "Unable to open HDF5 file", self.subject_id, self.exp_date, self.exp_type, "due to error:"
            #     print e
            #     return -1

        self.normalize_fluorescence_data()
        self.crop_data() #crop data to range specified at commmand line

        if self.smoothness != 0:
            print "--> Smoothing data with parameter: ", self.smoothness
            self.fluor_data = self.smooth(int(self.smoothness), window_type='gaussian')
        else:
            print "--> No smoothing parameter specified."

            
        if self.filter_freqs is not None:
            freqlist = self.filter_freqs.split(':')
            print freqlist
            self.fluor_data = self.notch_filter(freqlist[0], freqlist[1])
        
        if self.save_txt:
            self.save_time_series(self.output_path)
        if self.save_to_h5 is not None:
            self.save_time_series(self.output_path, output_type="h5", h5_filename=self.save_to_h5)
        if self.save_debleach:
            self.debleach(self.output_path)

        return self.fluor_data, self.trigger_data


    def crop_data(self):
        """
        ---Crop data to specified time range--- 
        Range is provided as a command line argument 
        in the format:      <start-time>:<end-time>
        Default is no cropping, specified by 0:-1
        """
        if self.time_range != None:
            tlist = self.time_range.split(':')
            print "--> Crop data to range: ", tlist
            if len(tlist) != 2:
                print 'Error parsing --time-range argument.  Be sure to use <start-time>:<end-time> syntax.'
                sys.exit(1)
            
            self.t_start = 0 if tlist[0] == '-1' else int(self.convert_seconds_to_index(int(tlist[0])))
            self.t_end = len(self.fluor_data) if tlist[1] == '-1' else int(self.convert_seconds_to_index(int(tlist[1])))
            
            self.fluor_data = self.fluor_data[self.t_start:self.t_end]
            self.trigger_data = self.trigger_data[self.t_start:self.t_end]
            self.time_stamps = self.time_stamps[self.t_start:self.t_end]
        else:
            print "--> Data not cropped. No range has been specified."


    def load_trigger_data(self):
        """
        ---Load trigger data---
        These are the times corresponding to behavioral events 
        such as licks, or social interactions.
        For lickometer data, the trigger data is recorded during data
        acquisition and is included in the previously loaded data file
        For social  behavior, event times must be loaded
        from a separate file location: trigger_path
        """

        if self.trigger_path is None: 
            self.trigger_data = self.data[:,self.trigger_channel]
        else:
            try:
                self.time_tuples = self.load_event_data(self.s_file, self.e_file)
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

        self.trigger_data *= 3/np.max(self.trigger_data)
        self.trigger_data -= np.min(self.trigger_data)
        if self.exp_type == 'sucrose': #event times are recorded differently by behavior handscoring vs. by lickometer
            self.trigger_data = np.max(self.trigger_data) - self.trigger_data
       # self.trigger_data *= -1
       # self.trigger_data += 1

        if self.fluor_normalization == "deltaF":
            self.trigger_data *= 1.5*np.max(self.fluor_data)

        print "--> Trigger data loaded"

    def normalize_fluorescence_data(self):
        """
        Normalize data to either 'deltaF', the standard metric used
        in publications which shows deviation from the median value
        of the entire time series, 'standardize', which shifts and scales
        the time series to be between 0 and 1, and 'raw', which does
        not alter the time series at all.
        """
            
        if self.fluor_normalization == "deltaF":
            median = np.median(self.fluor_data)
            print "--> Normalization: deltaF. Median of raw fluorescent data: ", median
            self.fluor_data = (self.fluor_data-median)/median #dF/F
            
        elif self.fluor_normalization == "standardize":
            print "--> Normalization: standardized to between 0 and 1. Max of raw fluorescent data: ", np.max(self.fluor_data)
            self.fluor_data -= np.min(self.fluor_data)
            self.fluor_data /= np.max(self.fluor_data)
            self.fluor_data +=0.0000001 # keep strictly positive

        elif self.fluor_normalization == "raw":
            print "--> Normalization: raw (no normalization). Max of raw fluorescent data: ", np.max(self.fluor_data)
            pass
        else:
            raise ValueError( self.fluor_normalization, "is not a valid entry for --fluor-normalization.")


    def load_event_data( self, s_filename, e_filename ):
        """
        Load start and end times for coded events. 
        """
        self.s_vals = np.load(s_filename)['arr_0']
        self.e_vals = np.load(e_filename)['arr_0']
        return zip(self.s_vals,self.e_vals)

    def plot_basic_tseries( self, out_path=None, window=None, resolution=30 ):
        """
        Generate a plot showing the raw calcium time series, with triggers
        corresponding to events (e.g. licking for sucrose) superimposed.
        Here the window indicates which segment of the entire time series to plot
        Make the resolution parameter smaller when plotting zoomed in views of the time series.
        """
        # clear figure
        pl.clf()

        start = int(self.time_range.split(':')[0])
        end = int(self.time_range.split(':')[1])
        end = end if end != -1 else max(self.time_stamps)
        start = start if start != -1 else min(self.time_stamps)

        print "--> Length of time series: ", end-start
        
        # if end == -1:
        #     end = max(FA.time_stamps)
        # if start == -1:
        #     start = min(FA.time_stamps)

        #Set a larger resolution to ensure the plotted file size is not too large
        if end - start <= 100 and end - start > 0:
            resolution = 1
        elif end - start < 500 and end - start > 100:
            resolution = 10
        elif end - start >= 500 and end - start < 1000:
            resolution = 30
        else:
            resolution = 40

        print "--> Resolution: ", resolution

        # get appropriate time values for x-axis
        time_vals = self.time_stamps[range(len(self.fluor_data))]
        fluor_data = self.fluor_data
        trigger_data = self.trigger_data

        if window is not None:
            window_indices = [self.convert_seconds_to_index( window[0]),
                              self.convert_seconds_to_index( window[1])]
            time_vals = time_vals[window_indices[0]:window_indices[1]] 
            fluor_data = fluor_data[window_indices[0]:window_indices[1]]
            trigger_data = trigger_data[window_indices[0]:window_indices[1]]


        trigger_low = min(trigger_data) + 0.2
        #print "trigger_low", trigger_low
        #print "trigger_data", trigger_data
        trigger_high_locations = [time_vals[i] for i in range(len(trigger_data)) if trigger_data[i] > trigger_low]
        # Be careful whether event is recorded by trigger high or trigger low (i.e. > or < trigger_low)

        # make filled blocks for trigger onset/offset
        #ymax = 1.1*np.max(fluor_data)
        #ymin = 1.1*np.min(fluor_data)

        if self.exp_type == 'sucrose':
            ymax = 3.0
            ymin = -1
        elif self.exp_type == 'homecagesocial':
            ymax = 0.5
            ymin = -ymax/3.0
        elif self.exp_type == 'homecagenovel':
            ymax = 0.5
            ymin = -ymax/3.0
        elif self.exp_type == 'EPM':
            ymax = 0.5
            ymin = -ymax/3.0

        if self.fluor_normalization == 'raw':
            ymax = 10.0
            ymin = -1.0

#        pl.fill( time_vals[::2], 10*trigger_data[::2] - 2, color='r', alpha=0.3 )
#        pl.fill( time_vals[::2], trigger_data[::2], color='r', alpha=0.3 )
        pl.vlines(trigger_high_locations, -20, 20, edgecolor='r', linewidth=0.5, facecolor='r' )
        pl.plot( time_vals[::resolution], fluor_data[::resolution], 'k-') #Only plot some of the points to non-noticeably decrease plot file size
        pl.ylim([ymin,ymax])
        if window is not None:
            pl.xlim([window[0], window[1]])
        else:
            pl.xlim([min(self.time_stamps), max(self.time_stamps)])
        if self.fluor_normalization == "deltaF":
            pl.ylabel('deltaF/F')
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
         #   pl.savefig(out_path + "basic_time_series.svg")
         #   pl.savefig(out_path + "basic_time_series.tiff")

    def save_time_series( self, save_path='.', output_type="txt", h5_filename=None ):
        """
        Save the raw calcium time series, with triggers corresponding to events
        (e.g. licking times for sucrose, approach/interaction times for novel object
        and social) in the same time frame of reference, outputting either
          -- a txt file, if ouput_type is txt
          -- write to an hdf5 file specified by self.save_to_h5 if output_type is h5
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

            subject_num.attrs['mouse_type'] = self.mouse_type
                
            if prefix.split("-")[0] not in list(subject_num):
                print "\t---> Creating subgroup:", prefix.split("-")[0]
                date = subject_num.create_group(prefix.split("-")[0])
            else:
                print "\t---> Loading subgroup:", prefix.split("-")[0]
                date = subject_num[prefix.split("-")[0]]

##Isaac changed this below on 20130202
            if prefix.split("-")[2] not in list(date):
                print "\t---> Creating subsubgroup:", prefix.split("-")[2]
                run_type = date.create_group(prefix.split("-")[2])
            else:
                print "\t---> Loading subsubgroup:", prefix.split("-")[2]
                run_type = date[prefix.split("-")[2]]

            try:
                dset = run_type.create_dataset("time_series_arr", data=out_arr)
                if np.shape(self.time_tuples) != ():
                    print "self.time_tuples", np.shape(self.time_tuples)
                    dset = run_type.create_dataset("event_tuples", data=self.time_tuples)
                else:
                    print "NO TIME TUPLES"

                dset.attrs["time_series_arr_names"] = ("time_stamp", "trigger_data", "fluor_data")
                dset.attrs["original_file_name"] = prefix
            except Exception, e:
                print "Unable to write data array due to error:", e

                
            h5_file.close() # close the file
            
            if self.save_and_exit:
                sys.exit(0)
        else:
            raise NotImplemented("The entered output_type has not been implemented.")               

    def plot_next_event_vs_intensity( self, intensity_measure="peak", next_event_measure="onset", window=[1, 3], out_path=None, plotit=True):
        """
        Generate a plot of next event onset delay (onset) or event length (length) as a function
        of an intensity measure that can be one of
          -- peak intensity of last event (peak)
          -- integrated intensity of last event (integrated)
          -- integrated intensity over history window (window)
        """
        start_times = self.get_event_times("rising")
        end_times = self.get_event_times("falling")
        if start_times == -1:
            raise ValueError("Event times seem to have failed to load.")

        # get intensity values
        if intensity_measure == "peak":
            intensity = np.zeros(len(start_times))
            for i in xrange(len(start_times)):
                peak = self.get_peak(start_times[i]-window[0], start_times[i]+window[1]) # end_times[i])
                intensity[i] = peak
        elif intensity_measure == "integrated":
            intensity = self.get_areas_under_curve( start_times, window, baseline_window=window, normalize=False)
        elif intensity_measure == "window":
            window[1] = 0
            intensity = self.get_areas_under_curve( start_times, window, baseline_window=window, normalize=False)
        else:
            raise ValueError("The entered intensity_measure is not one of peak, integrated, or window.")
        # get next event values
        if next_event_measure == "onset":
            next_vals = np.zeros(len(start_times)-1)
            for i in xrange(len(next_vals)):
                next_vals[i] = start_times[i+1] - end_times[i]
        elif next_event_measure == "length":
            next_vals = np.zeros(len(start_times)-1)
            for i in xrange(len(next_vals)):
                next_vals[i] = end_times[i+1] - start_times[i+1]
        else:
            raise ValueError("Entered next_event_measure not implemented.")

        # lag intensities relative to events (except in history window case)
        if intensity_measure == "window":
            intensity = intensity[1::]
        else:
            intensity = intensity[0:-1]

        if plotit:
            #Plot the area vs the time of each event
            pl.clf()
            pl.loglog(intensity, next_vals, 'ro')
            pl.ylabel('Next event value')
            pl.xlabel('Intensity')

            if out_path is None:
                pl.show()
            else:
                pl.savefig(os.path.join(out_path,"next_event_vs_intensity.pdf"))
        else:
            return intensity, next_vals

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

    def smooth(self, num_time_points, window_type='gaussian'):
        """
        Convolve a simple averaging window of length num_time_points
        and type window_type ('gaussian, 'rect')
        """

        print "smoothing."
        if window_type == 'gaussian':
            window = signal.gaussian(100, num_time_points)
        elif window_type == 'rect':
            window = np.ones(num_time_points)
        else:
            window = np.ones(num_times_points)

        smoothed_fluor_data = np.convolve(self.fluor_data, window, mode='same') #returns an array of the original length
        print "done smoothing."
        return smoothed_fluor_data/np.sum(window)


    def get_peaks( self, window_in_seconds=1 ):
        """
        Heuristic for finding local peaks in the calcium data. 
        """
        print "get_peaks"
       # peak_widths = np.array([100,250,500,750,1000])
        window_in_indices = self.convert_seconds_to_index(window_in_seconds)
        print "window_in_indices", window_in_indices
        peak_widths = np.array([window_in_indices])
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

        print "fft calculated."

        bpfilt = np.where(self.fft_freq <=high and self.fft_freq >= low, bp, 0)
#        for i in range(len(bp)):
#            if self.fft_freq[i]<= high and self.fft_freq[i]>= low:
#                bp[i] = 0
#        self.fft = bp #update fft of the class to the filtered version
        self.fft=bpfilt

        ibp = sp.ifft(bp)
        notch_filt_y = np.real(ibp)
        notch_filt_y += np.median(self.fluor_data) - np.median(notch_filt_y)
        self.filt_fluor_data = notch_filt_y
        return notch_filt_y

    def get_chunks_not_during_events(self, fluor_data, event_times, end_times):
        """
        Returns a time series from which all event periods have been removed
        """
        time_chunks = np.zeros(0)
        time_stamp_chunks = np.zeros(0)

        for i in range(len(event_times)):
            e = event_times[i]
            if i == 0:
                n = 0
            else:
                n = event_times[i-1]
           # try:
            e_idx = np.where(e<self.time_stamps)[0][0]
            n_idx = np.where(n<self.time_stamps)[0][0]

            chunk = fluor_data[range((n_idx),(e_idx))]
                #print [range((e_idx-window_indices[0]),(e_idx+window_indices[1]))]
            time_chunks = np.append(time_chunks, chunk)
            time_stamp_chunks = np.append(time_stamp_chunks, self.time_stamps[range((n_idx),(e_idx))])

            #except:
             #   print "Unable to extract window:", [(e-window_indices[0]),(e+window_indices[1])]
        return (time_chunks, time_stamp_chunks)

    def analyze_spectrum( self, peak_index=None, out_path=None ):
        """
        Compares the spectrum during lick epochs vs not during lick epochs
        -->Still in progress
        """

        type = self.exp_type

        if type == "sucrose":
            if self.event_start_times is None:
                event_times, end_times = self.get_sucrose_event_times()
                self.event_start_times = event_times
                self.event_end_times = end_times
            else:
                event_times = self.event_start_times
                end_times = self.event_end_times
        elif type == "homecagenovel" or type == "homecagesocial" or type == 'EPM':
            event_times = self.get_event_times(edge)
        else:
            print "Experiment type not implemented. use --exp-type flag with 'sucrose', 'homecagenovel', or 'homecagesocial'."
            sys.exit(0)


        event_lengths = np.array(end_times) - np.array(event_times)
        window = [0, np.max(event_lengths) + 10] #10 is right now an arbitrary buffer (not seconds but indices)
        baseline_window = [np.max(event_lengths) + 10, np.max(event_lengths) + 10]
        time_chunks = self.get_time_chunks_around_events(self.fluor_data, event_times, window, baseline_window=baseline_window )

        window_indices = [self.convert_seconds_to_index( window[0]),
                              self.convert_seconds_to_index( window[1])]

        time_arr = np.asarray(time_chunks).T
        x = self.time_stamps[0:time_arr.shape[0]]-self.time_stamps[window_indices[0]]

        if peak_index is None:
            for i in range(len(event_times)):
                self.plot_chunk_periodogram(x, time_arr[:, i], out_path=out_path)
        else:
            self.plot_chunk_periodogram(x, time_arr[:, peak_index], out_path=out_path, peak_index=peak_index)


        no_events_chunk, no_events_times = self.get_chunks_not_during_events(self.fluor_data, event_times, end_times)
        self.plot_chunk_periodogram(no_events_times, no_events_chunk, out_path=out_path)


    def plot_chunk_periodogram( self, time_stamps, data, out_path=None, plot_type="log", window_len=20, peak_index=None, title=None):
        """
        Given a time series or section of a time series, plot the frequency content 
        """

        fft = sp.fft(data)
        fft = fft[:]

        n = data.size
        timestep = np.max(time_stamps[1:] - time_stamps[:-1])
        fft_freq = np.fft.fftfreq(n, d=timestep)

        Y = np.abs(fft)**2
        freq = fft_freq

        s = np.r_[Y[window_len-1:0:-1],Y,Y[-1:-window_len:-1]] #periodic boundary
        w = np.bartlett(window_len)
        Y = np.convolve(w/w.sum(), s, mode='valid')

        num_values = int(min(len(Y), len(freq))*.5) #Cut off negative frequencies
        #start_freq = 0.25*(num_values/100) #i.e. start at 0.25 Hz
        start_freq = 0#(num_values/100) #i.e. start at 0.25 Hz
        freq_vals = freq[range(num_values)]
        Y = Y[range(num_values)]

        pl.plot( freq_vals[start_freq:num_values], np.log(Y[start_freq:num_values]), 'k-')
        pl.ylabel('Log(Spectral Density) (a.u.)')
        pl.xlabel('Frequency (Hz)')
        pl.title(self.input_path)
        #pl.axis([1, 100, 0, 1.1*np.log(np.max(Y[start_freq:num_values]))])
        pl.axis([0, 100, -5, 10])
        
        if out_path is None:
            pl.show()
        else:
            print "Saving periodogram..."
            if title is not None:
                pl.savefig(out_path + "periodogram_" + title + ".pdf")
                pl.savefig(out_path + "periodogram_" + title + ".png")

            if peak_index is None:
                pl.savefig(out_path + "periodogram.pdf")
                pl.savefig(out_path + "periodogram.png")
            else:
                pl.savefig(out_path + "periodogram_" + peak_index + ".pdf")
                pl.savefig(out_path + "periodogram_" + peak_index + ".png")

        return (freq_vals, Y)


    def plot_full_periodogram( self, out_path = None, plot_type="log", window_len = 20):
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
            pl.savefig(out_path + "peak_finding.png")


    def get_time_chunks_around_events(self, data, event_times, window, baseline_window=None, end_event_times=None):
        """
        Extracts chunks of fluorescence data around each event in 
        event_times, with before and after event durations
        specified in window as [before, after] (in seconds).
        Subtracts the baseline value from each chunk (i.e. sets the minimum value in a chunk to 0)
        Set baseline_window = -1 for no baseline normalization
        """
        window_indices = [self.convert_seconds_to_index( window[0]),
                          self.convert_seconds_to_index( window[1])]

        if baseline_window is not None and baseline_window != -1:
            baseline_indices = [self.convert_seconds_to_index( baseline_window[0]),
                                self.convert_seconds_to_index( baseline_window[1])]

        time_chunks = []
        for i in range(len(event_times)):
#        for e in event_times:
            e = event_times[i]
            if end_event_times is not None:
                full_window = [0, end_event_times[i] - event_times[i]]
                window_indices = [self.convert_seconds_to_index( full_window[0]),
                          self.convert_seconds_to_index( full_window[1])]
                print "len(epoch) = ", full_window[1]

           # try:
            e_idx = np.where(e<self.time_stamps)[0][0]
            if (e_idx + window_indices[1] < len(data)-1) and (e_idx - window_indices[0] > 0):
                chunk = data[range(max(0, (e_idx-window_indices[0])),min(len(data)-1, (e_idx+window_indices[1])))]
                if baseline_window is not None and baseline_window != -1:
                    baseline_chunk = data[range(max(0, (e_idx-window_indices[0])), min(len(data)-1, (e_idx+window_indices[1])))]
                    baseline = np.min(baseline_chunk)
                elif baseline_window == -1:
                    baseline = 0
                else:
                    baseline = np.min(chunk)



                time_chunks.append(chunk - baseline)
            #except:
             #   print "Unable to extract window:", [(e-window_indices[0]),(e+window_indices[1])]
        return time_chunks

    def plot_perievent_hist( self, event_times, window, out_path=None, plotit=True, subplot=None, baseline_window=None ):
        """
        Peri-event time histogram for given event times.
        Plots the time series and their median over a time window around
        each event in event_times, with before and after event durations
        specified in window as [before, after] (in seconds).
        """
        # new figure
        if plotit and subplot is None:
            pl.clf()
            fig = pl.figure()
            ax = fig.add_subplot(111)
        elif subplot is not None:
            ax = subplot

        print "Generating peri-event plot..."
        print "\t--> Number of bouts:", len(event_times)
        print "\t--> Window used for peri-event plot:", window

        # get blocks of time series for window around each event time
        time_chunks = self.get_time_chunks_around_events(self.fluor_data, event_times, window, baseline_window=baseline_window)

        # get time values from frame indices
        window_indices = [ self.convert_seconds_to_index(window[0]),
                           self.convert_seconds_to_index(window[1]) ]

        # plot each time window, colored by order
        time_arr = np.asarray(time_chunks).T
        x = self.time_stamps[0:time_arr.shape[0]]-self.time_stamps[window_indices[0]] ###IS THIS RIGHT?
        ymax = np.max(time_arr)
        ymax += 0.1*ymax
        ymin = np.min(time_arr)
        ymin -= 0.1*ymin
        for i in xrange(time_arr.shape[1]):
            if plotit:
                ax.plot(x, time_arr[:,i], color=pl.cm.jet(255-255*i/time_arr.shape[1]), alpha=0.75, linewidth=1)
            x.shape = (len(x),1) 
            x_padded = np.vstack([x[0], x, x[-1]])
            time_vec = time_arr[:,i]; time_vec.shape = (len(time_vec),1)
            time_vec_padded = np.vstack([0, time_vec,0]) 

            if plotit:
                pl.fill(x_padded, time_vec_padded, facecolor=pl.cm.jet(255-255*i/time_arr.shape[1]), alpha=0.25 )            
                pl.ylim([ymin, ymax])
            
        if plotit:
            # add a line for the event onset time
            pl.axvline(x=0,color='black',linewidth=1,linestyle='--')

            # label the plot axes
            if self.fluor_normalization == "deltaF":
                pl.ylabel(r'$\delta F/F$')
            else:
                pl.ylabel('Fluorescence Intensity (a.u.)')
            pl.xlabel('Time from onset of social bout (seconds)')

            # show plot now or save of an output path was specified
            if out_path is None and subplot is None:
                pl.show()
            elif subplot is None:
                print "Saving peri-event time series..."
                pl.savefig(out_path + "perievent_tseries.pdf")
                pl.savefig(out_path + "perievent_tseries.png")

    def plot_peritrigger_edge( self, window, edge="rising", out_path=None ):
        """
        Wrapper for plot_perievent histograms specialized for
        loaded event data that comes as a list of pairs of
        event start and end times.
        type can be "homecage" or "sucrose"
        """

        type = self.exp_type

        if type == "sucrose":
            if self.event_start_times is None:
                event_times, end_times = self.get_sucrose_event_times()
                self.event_start_times = event_times
                self.event_end_times = end_times
            else:
                event_times = self.event_start_times
                end_times = self.event_end_times
        elif type == "homecagesocial" or type == "homecagenovel" or type == 'EPM' :
            event_times = self.get_event_times(edge)
        else:
            print "Experiment type not implemented. use --exp-type flag with 'sucrose', 'homecagenovel', or 'homecagesocial'."
            sys.exit(0)

        if event_times[0] != -1:
            self.plot_perievent_hist( event_times, window, out_path=out_path )
        else:
            print "No event times loaded. Cannot plot perievent."        

    def get_fft(self):
        print "getting fft."
        if self.filt_fluor_data is None:
            rawsignal = self.fluor_data
        else:
            rawsignal = self.filt_fluor_data
        
        print "calculating fft."
        fft = sp.fft(rawsignal)
        self.fft = fft[:]

        n = rawsignal.size
        print "rawsignal.size", n
        timestep = np.max(self.time_stamps[1:] - self.time_stamps[:-1])
        self.fft_freq = np.fft.fftfreq(n, d=timestep)

    def get_event_times( self, edge="rising", nseconds=0):
        """
        Extracts a list of the times (in seconds) corresponding to
        interaction events to be time-locked with the signal
        specialized for loaded event data that comes as a list of 
        pairs of event start and end times.
        nseconds defines a minimum distance between the end of one event
        and the start of a previous event. if you do not wish to impose
        such a restriction, set nseconds=None.
        """
        if self.event_spacing is not None:
            nseconds = self.event_spacing


        if self.time_tuples is not None:
            event_times = []
            for i in range(len(self.time_tuples)):
                pair = self.time_tuples[i]

                if i==0 or nseconds is None or (self.time_tuples[i][0] - self.time_tuples[i-1][1] >= nseconds):
                    if edge == "rising":
                        event_times.append(pair[0])
                    elif edge == "falling":
                        event_times.append(pair[1])
                    else:
                        raise ValueError("Edge type must be 'rising' or 'falling'.")
            return event_times
        else:            
            print "No event times loaded. Cannot find edges."        
            return [-1]

    def get_peaks_convolve( self, window, out_path=None, type="sucrose"):
        """
        Find peaks based on the difference in signal integrated
        over a window of nseconds before and nseconds after
        each time point
        """

        nindex_after = self.convert_seconds_to_index(window[1])
        nindex_before = self.convert_seconds_to_index(window[0])
        mask = np.ones(nindex_after)
        mask = np.append(mask, -np.ones(nindex_before))

        time_vals = self.time_stamps[range(len(self.fluor_data))]
        smoothed_gradient = np.convolve(self.fluor_data, mask)
        smoothed_gradient = smoothed_gradient[nindex-1:]/(2*nindex)

        diff_nseconds = 0.1
        diff_nindex = self.convert_seconds_to_index(0.1)
        diff_mask = np.ones(diff_nindex)
        diff_mask = np.append(diff_mask, -np.ones(diff_nindex))
        smoothed_curve = np.convolve(smoothed_gradient, diff_mask)


        type = self.exp_type

        #Now plot it to see if it worked...
        if type == "sucrose":
            if self.event_start_times is None:
                event_times, end_times = self.get_sucrose_event_times()
                self.event_start_times = event_times
                self.event_end_times = end_times
            else:
                event_times = self.event_start_times
                end_times = self.event_end_times
        elif type == "homecagesocial" or type == "homecagenovel" or type == 'EPM':
            event_times = self.get_event_times("rising")
        else:
            print "Experiment type not implemented. use --exp-type flag with 'sucrose', 'homecagenovel', or 'homecagesocial'."
            sys.exit(0)


        window = [25, 25]
        fluor_chunks = self.get_time_chunks_around_events(self.fluor_data, event_times, window, baseline_window=window)
        grad_chunks = self.get_time_chunks_around_events(smoothed_gradient, event_times, window, baseline_window=window)
        curve_chunks = self.get_time_chunks_around_events(smoothed_curve, event_times, window, baseline_window=window)


        window_indices = [ self.convert_seconds_to_index(window[0]),
                           self.convert_seconds_to_index(window[1]) ]


        time_arr = np.asarray(fluor_chunks).T
        grad_arr = np.asarray(grad_chunks).T
        curve_arr = np.asarray(curve_chunks).T
        x = self.time_stamps[0:time_arr.shape[0]]-self.time_stamps[window_indices[0]] ###IS THIS RIGHT?
    
        ymax = np.max(time_arr)
        ymax += 0.1*ymax
        ymin = np.min(time_arr)
        #ymin -= 0.1*ymin
        ymin = -1
       
        for i in xrange(time_arr.shape[1]):

            grad_peaks = [time_vals[i] for k in range(len(curve_arr[:,i])) if (np.round(10000*curve_arr[k, i]) == 0)]

            fig = pl.figure()
            ax = fig.add_subplot(111)
            ax.plot(x, time_arr[:,i], color=pl.cm.winter(255-255*i/time_arr.shape[1]), alpha=0.75, linewidth=1)
            ax.plot(x, grad_arr[:,i], 'r')
            ax.plot(x, 1000*curve_arr[:,i], 'k')
            x.shape = (len(x),1) 
            x_padded = np.vstack([x[0], x, x[-1]])
            time_vec = time_arr[:,i]; time_vec.shape = (len(time_vec),1)
            time_vec_padded = np.vstack([0, time_vec,0]) 
            pl.fill(x_padded, time_vec_padded, facecolor=pl.cm.winter(255-255*i/time_arr.shape[1]), alpha=0.25 )   
            #pl.ylim([0,ymax])
            pl.ylim([ymin, ymax])
            
            for j in range(len(grad_peaks)):
                pl.axvline(x=grad_peaks[j], color='r', linewidth=1)
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
        #else:
          #  print "Saving peri-event time series..."
            #pl.savefig(os.path.join(out_path,'perievent_tseries'))
          #  pl.savefig(out_path + "smoothed_perievent_tseries.png")


       ##TO DO >>>>>>>>

        # print np.shape(smoothed_gradient)
        # print np.shape(time_vals)
        # print np.shape(self.fluor_data)
        # pl.plot(time_vals, self.fluor_data)
        # pl.plot(time_vals, smoothed_gradient, 'r')
        # pl.ylabel('Gradient of time series')
        # pl.xlabel('Time [s]')
        # if out_path is not None:
        #      pl.savefig(out_path + "smoothed_gradient.png")
        # else:
        #      pl.show()

        return smoothed_gradient

    def get_lick_density_vs_time( self, window_after, out_path=None):
        """
        For each time point, plot the number of licks occuring
        in the window_after (in seconds) after that time point.
        """

        nindex = self.convert_seconds_to_index(window_after)
        mask = np.ones(nindex)
        self.trigger_data = np.floor(self.trigger_data)

        time_vals = self.time_stamps[range(len(self.trigger_data))]
        density = np.convolve(self.trigger_data, mask)
        density = density[nindex-1:]

        # print "nindex:", nindex
        # print "shape(density)", np.shape(density[nindex-1:])
        # print "shape(time)", np.shape(time_vals)
        # pl.plot(time_vals, density)
        # pl.fill(time_vals, 100*self.trigger_data, facecolor='r', alpha=0.5)
        # pl.ylabel('Density of licks calculated over window of ' + str(nseconds))
        # pl.xlabel('Time [s]')
        # pl.title('Density of licks')
        # pl.show()

        return density

        # if out_path is not None:
        #     pl.savefig(out_path + "lick_density.png")
        # else:
        #     pl.show()

    def get_sucrose_event_times( self, nseconds=15, density=None, edge="rising"):
        """
        Extracts a list of the times (in seconds) corresponding
        to sucrose lick epochs. Epochs are determined by first calculating
        the density of licks (using a window of nseconds).
        The start of a licking epoch is then determined by calculating
        the location of the rising edges of this density plot.
        The end of a licking epoch can be determined by calculating
        the time at which this density returns to zero minus nseconds
        """

        if self.event_spacing is not None:
            nseconds = self.event_spacing

        nindex = self.convert_seconds_to_index(nseconds)
        mask = np.ones(nindex)

        self.trigger_data = np.floor(2*self.trigger_data) #make sure that no licks is represented by 0
        time_vals = self.time_stamps[range(len(self.trigger_data))]
        print time_vals
        if density is None:
            density = np.convolve(self.trigger_data, mask)
            density = density[nindex-1:] 

        dmed = np.median(density)


        start_times = np.zeros(0)
        end_times = np.zeros(0)
        for i in range(nindex, len(density)-2):
            if np.round(density[i-1]) == np.round(dmed) and density[i] > dmed:
                start_times = np.append(start_times, time_vals[i+nindex]) #?Should I subtract -0.013, the length of the lickometer signal
            if np.round(density[i+1]) == np.round(dmed) and density[i] > dmed:
                end_times = np.append(end_times, time_vals[i])

        #filter out all of the single licks
        filt_start_times = np.zeros(0)
        filt_end_times = np.zeros(0)
        for i in range(len(start_times)):
            if start_times[i] + 5 < end_times[i]: #only use licking epochs that last at least longer than 5 seconds ????
                filt_start_times = np.append(filt_start_times, start_times[i])
                filt_end_times = np.append(filt_end_times, end_times[i])

        print "start_times ", filt_start_times
        print "end_times ", filt_end_times
    
        # pl.plot(time_vals, density)
        # pl.fill(time_vals, 100*self.trigger_data, facecolor='r', alpha=0.5)
        # pl.show()

        return (filt_start_times, filt_end_times)

    def convert_seconds_to_index( self, time_in_seconds):
        return np.where( self.time_stamps >= time_in_seconds)[0][0]

    def debleach( self, out_path=None ):
        """
        Remove trend from data due to photobleaching by fitting the time serie with an exponential curve
        and then subtracting the difference between the curve and the median value of the time series. 
        """
        print "--> Debleaching"
        
        fluor_data = self.fluor_data
        time_stamps = self.time_stamps[range(len(self.fluor_data))]

        trigger_data = self.trigger_data

        #print np.shape(fluor_data), np.shape(time_stamps), np.shape(trigger_data)

        xp, pxp, x0, y0, c, k, r2, yxp = self.fit_exponential(time_stamps, fluor_data)
        w, r2lin, yxplin = self.fit_linear(time_stamps, fluor_data)
        if r2lin > r2:
            flat_fluor_data = fluor_data - yxplin + np.median(fluor_data)
            r2 = r2lin
        else:
            flat_fluor_data = fluor_data - yxp + np.median(fluor_data)

        #flat_fluor_data = flat_fluor_data - min(flat_fluor_data) + 0.000001

        orig, = pl.plot(time_stamps, fluor_data)
        pl.plot(time_stamps, yxp, 'r')
        debleached, = pl.plot(time_stamps, flat_fluor_data)
        pl.xlabel('Time [s]')
        pl.ylabel('Raw fluorescence (a.u.)')
        pl.title('Debleaching the fluorescence curve')
        pl.legend([orig, debleached], [ "Original raw", "Debleached raw"])


        out_arr = np.zeros((len(flat_fluor_data),4))
        out_arr[:,0] = flat_fluor_data

        trigger_data = -1*(trigger_data/np.max(trigger_data) - 1)
        out_arr[:,3] = trigger_data #these numbers are hardcoded above

        if out_path is None:
                pl.title("No output path given")
                #pl.show()
        else:
            pl.savefig(out_path + "_debleached.png")
            np.savez(out_path + "_debleached.npz", data=out_arr, time_stamps=time_stamps)
            print "--> Debleached output: " 
            print out_path + "_debleached.npz"
            print ""


        if self.save_and_exit:
            sys.exit(0)


    def plot_peak_statistics( self, peak_times, peak_vals ):
        """
        Plots showing statistics of calcium peak data.
          --> Peak height as function of time since last peak
          --> Histograms of peak times and vals
        """
        pass


    def get_areas_under_curve( self, start_times, window, baseline_window=None, normalize=False):
        """
        Returns a vector of the area under the fluorescence curve within the provided
        window [before, after] (in seconds), that surrounds each start_time.
        Normalize determines whether to divide the area by the maximum fluorescence
        value of the window (this is more if you want to look at the "shape" of the curve)
        """
        print "window: ", window
        time_chunks = self.get_time_chunks_around_events(self.fluor_data, start_times, window, baseline_window)
        
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
            

    def plot_area_under_curve( self, start_times, end_times, window, normalize=False, out_path=None):
        """
        Plots of area under curve for each event_time 
        with before and after event durationsspecified in window as 
        [before, after] (in seconds).
        -- we decided to use a 1s window because it captures the median length
        of both social and novel object interaction events

        """
        
        print "plot_area_under_curve"
        #change the normalization depending on whether you wish to divide (normalize)
        # by the maximum fluorescence value in the window following
        # each event

        areas = self.get_areas_under_curve(start_times, window, baseline_window=window)

        window_indices = [ self.convert_seconds_to_index(window[0]),
                           self.convert_seconds_to_index(window[1]) ]
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
                np.savez(out_path + "normalized_area_under_peaks_" + str(int(10*self.time_stamps[window_indices[1]])) + "s.npz", scores=areas, event_times=start_times, end_times=end_times, window=self.time_stamps[window_indices[1]])
                # Assume not using windows longer than a few seconds. Thus save the file so as to display one decimal point
            else:
                pl.savefig(out_path + "plot_area_under_curve_non_normal" + str(int(10*self.time_stamps[window_indices[1]])) + "s.pdf")
                np.savez(out_path + "non-normalized_area_under_peaks_" + str(int(10*self.time_stamps[window_indices[1]]))+ "s.npz", scores=areas, event_times=start_times, end_times=end_times, window=self.time_stamps[window_indices[1]])


    def plot_area_under_curve_wrapper( self, window, edge="rising", normalize=True, out_path=None):
        """
        Wrapper for plot_area_under_curve vs. time plots specialized for
        loaded event data that comes as a list of pairs of
        event start and end times (in seconds)
        """
        start_times = self.get_event_times("rising")
        end_times = self.get_event_times("falling")
        if start_times != -1:
            self.plot_area_under_curve( start_times, end_times, window, normalize, out_path=out_path )
        else:
            print "No event times loaded. Cannot plot perievent."  

    def get_sucrose_peak(self, start_time, end_time):
        """
        Return the maximum fluorescence value found between
        start_time and end_time (in seconds)
        May eventually account for single licks and possibly for looking at a window
        beyond the length of time of a single lick
        """
        start_time_index = self.convert_seconds_to_index(start_time)
        end_time_index = self.convert_seconds_to_index(end_time)
        print "start ", start_time, " end ", end_time

        if start_time != end_time:
            if start_time_index < end_time_index:
                return np.max(self.fluor_data[start_time_index : end_time_index])
            else:
                return 0

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

    def invGamX(self, p, x):
        a, b = p
        #y = pow(b, a)/sp.special.gamma(a)*pow(x, -a-1)*np.exp(-b/x)
        y = sp.stats.invgamma.pdf(x, a, scale=b)
        return y

    def invGamX_residuals(self, p, x, y):
        return y - self.invGamX(p, x)

    def fit_invGam(self, x, y, num_point=100):
        aguess = [0, 1]
        bguess = [0, 0.1, 0.5, 1.0, 10, 100, 500, 1000]
        max_r2 = -1
        maxvalues = ()
        for ag in aguess:
            print "ag", ag
            for bg in bguess:
                print "bg", bg
                p_guess=(ag, bg)
                p, cov, infodict, mesg, ier = sp.optimize.leastsq(
                    self.invGamX_residuals, p_guess, args=(x, y), full_output=1)

                a, b = p 
                # print('''Reference data:\  
                #         a {a}
                #         b = {b}
                #         '''.format(a=a,b=b))

                numPoints = np.floor((np.max(x) - np.min(x))*num_points)
                xp = np.linspace(np.min(x), np.max(x), numPoints)
                #pxp = np.exp(-1*xp)
                pxp = self.invGamX(p, xp)
                yxp = self.invGamX(p, x)

                sstot = np.sum(np.multiply(y - np.mean(y), y - np.mean(y)))
                sserr = np.sum(np.multiply(y - yxp, y - yxp))
                r2 = 1 - sserr/sstot
                print "max_r2", max_r2
                if max_r2 == -1:
                    maxvalues = (xp, pxp, a, b, r2, yxp)
                if r2 > max_r2:
                    max_r2 = r2
                    maxvalues = (xp, pxp, a, b, r2, yxp)

        return maxvalues


    def eNegX(self, p, x):
        x0, y0, c, k=p
        #Set c=1 to normalize all of the trials, since we
        # are only interested in the rate of decay
        y = (1 * np.exp(-k*(x-x0))) + y0
        return y

    def eNegX_residuals(self, p, x, y):
        return y - self.eNegX(p, x)

    

    def fit_exponential(self, x, y, num_points=100):
        # Because we are optimizing over a nonlinear function
        # choose a number of possible starting values of (x0, y0, c, k)
        # and use the results from whichever produces the smallest 
        # residual
        # num_points gives the number of points in the returned curve, pxp

        kguess = [0, 0.1, 0.5, 1.0, 10, 100, 500, 1000]
        yguess = [0, 1]
        max_r2 = -1
        maxvalues = ()
        for kg in kguess:
            for yg in yguess:
                p_guess=(np.min(x), yg, 1, kg)
                p, cov, infodict, mesg, ier = sp.optimize.leastsq(
                    self.eNegX_residuals, p_guess, args=(x, y), full_output=1)

                x0,y0,c,k=p 
                # print('''Reference data:\  
                #         x0 = {x0}
                #         y0 = {y0}
                #         c = {c}
                #         k = {k}
                #         '''.format(x0=x0,y0=y0,c=c,k=k))

                numPoints = np.floor((np.max(x) - np.min(x))*num_points)
                xp = np.linspace(np.min(x), np.max(x), numPoints)
                #pxp = np.exp(-1*xp)
                pxp = self.eNegX(p, xp)
                yxp = self.eNegX(p, x)

                sstot = np.sum(np.multiply(y - np.mean(y), y - np.mean(y)))
                sserr = np.sum(np.multiply(y - yxp, y - yxp))
                r2 = 1 - sserr/sstot
                if max_r2 == -1:
                    maxvalues = (xp, pxp, x0, y0, c, k, r2, yxp)
                if r2 > max_r2:
                    max_r2 = r2
                    maxvalues = (xp, pxp, x0, y0, c, k, r2, yxp)

        return maxvalues

    def fit_peak(self, peak_index=None, type="sucrose", out_path=None):
        """
        Fits each individual peak using an inverse gamma function.
        Currently works moderately well, however there is an issue 
        with accurately choosing when the peak actually begins.
        """

        type = self.exp_type

        if type == "sucrose":
            if self.event_start_times is None:
                print "getting sucrose event times"
                event_times, end_times = self.get_sucrose_event_times()
                self.event_start_times = event_times
                self.event_end_times = end_times
                print "finished getting sucrose event times"
            else:
                event_times = self.event_start_times
                end_times = self.event_end_times
        elif type == "homecagesocial" or type == "homecagenovel" or type == 'EPM':
            event_times = self.get_event_times()
        else:
            print "Experiment type not implemented. use --exp-type flag with 'sucrose', 'homecagenovel', or 'homecagesocial'."
            sys.exit(0)


        event_lengths = np.array(end_times) - np.array(event_times)
        window = [0, np.max(event_lengths) + 10] #10 is right now an arbitrary buffer
        baseline_window = [np.max(event_lengths) + 10, np.max(event_lengths) + 10]
        time_chunks = self.get_time_chunks_around_events(self.fluor_data, event_times, window, baseline_window=baseline_window)

        window_indices = [self.convert_seconds_to_index( window[0]),
                              self.convert_seconds_to_index( window[1])]

        time_arr = np.asarray(time_chunks).T
        x = self.time_stamps[0:time_arr.shape[0]]-self.time_stamps[window_indices[0]]

        if peak_index is None:
            for i in range(len(event_times)):
                self.fit_invgamma(x, time_arr[:, i], out_path, i)
        else:
            self.fit_invgamma(x, time_arr[:, peak_index], out_path, peak_index)
           


    def fit_invgamma(self, x, y, out_path=None, peak_index=0):
        print "fitting invgamma"
        
        mask = np.ones(1000)
        yorig = y
      #  y = np.convolve(y, mask)
      #  y = y[len(mask)-1:]/len(mask)

        x = x - x[0]
        y = y/(np.sum(y)*(x[1]-x[0]))
        print np.sum(y)*(x[1]-x[0])

        max_r2 = -1
        maxvalues = ()


        b_parameters = range(0, 100)#[0.1, 1, 1.5, 2, 3, 5, 7, 8, 11, 20, 100, 500, 1000]
        a_parameters = [0.01]#, 0.1, 0.5, 1, 2, 5]
        for b in b_parameters:
                fit_alpha,fit_loc,fit_beta=ss.invgamma.fit(y, loc=0, scale=b)
                rv = ss.invgamma(fit_alpha, fit_loc, fit_beta)
                yxp = rv.pdf(x)


                sstot = np.sum(np.multiply(y - np.mean(y), y - np.mean(y)))
                sserr = np.sum(np.multiply(y[1:] - yxp[1:], y[1:] - yxp[1:]))
                r2 = 1 - sserr/sstot
                #print  "sstot: ", sstot, "sserr: ", sserr, "r2: ", r2

                if max_r2 == -1:
                    maxvalues = (fit_alpha, fit_loc, fit_beta)
                if r2 > max_r2:
                    max_r2 = r2
                    maxvalues = (fit_alpha, fit_loc, fit_beta)

        print "maxvalues", maxvalues, "max_r2: ", max_r2
        rv = ss.invgamma(maxvalues[0], maxvalues[1], maxvalues[2])
        pl.figure()
        pl.plot(x, yorig/(np.sum(yorig)*(x[1]-x[0])), 'b')
        pl.plot(x, y, 'k')
        fitplot, = pl.plot(x,rv.pdf(x), 'r')
        pl.title('Inverse gamma distribution model of calcium dynamics')
        pl.xlabel('Time since onset of licking epoch (seconds)')
        pl.ylabel('Fluorescence (normalized deltaF/F)')
        pl.legend([fitplot], [ r"$\alpha$ = " + "{0:.2f}".format(maxvalues[0]) + r", $\beta $= " + "{0:.2f}".format(maxvalues[2]) +  
                                r", $r^2 = $" + "{0:.2f}".format(max_r2)])
        if out_path is None:
            pl.show()
        else:
            pl.savefig(out_path + "inv_gamma_spike_" + str(peak_index) +  ".pdf")
            pl.savefig(out_path + "inv_gamma_spike_" + str(peak_index) +  ".png")


    def fit_lognorm(self, x, y):
        print "fitting lognorm"
        
        mask = np.ones(1000)
        yorig = y
      #  y = np.convolve(y, mask)
      #  y = y[len(mask)-1:]/len(mask)

        x = x - x[0]
        y = y/(np.sum(y)*(x[1]-x[0]))
        print np.sum(y)*(x[1]-x[0])

        max_r2 = -1
        maxvalues = ()

        a_parameters = [0.1, 1, 1.5, 2, 2.5, 3, 5, 10, 100, 500, 1000]
        b_parameters = [0.01, 0.1, 0.5, 1, 2, 5]
        for a in a_parameters:
            for b in b_parameters:
                logy = np.log(y - np.min(y) + 0.000001)
                print np.exp(np.mean(logy))
                fit_alpha,fit_loc,fit_beta=ss.lognorm.fit(y, shape=b*np.std(logy), scale=a*np.exp(np.mean(y)))
                print(fit_alpha,fit_loc,fit_beta)

                rv = ss.lognorm(fit_alpha, fit_loc, fit_beta)
                yxp = rv.pdf(x)

                sstot = np.sum(np.multiply(y - np.mean(y), y - np.mean(y)))
                sserr = np.sum(np.multiply(y - yxp, y - yxp))
                r2 = 1 - sserr/sstot

                print  "sstot: ", sstot, "sserr: ", sserr, "r2: ", r2
                if max_r2 == -1:
                    maxvalues = (fit_alpha, fit_loc, fit_beta)
                if r2 > max_r2:
                    max_r2 = r2
                    maxvalues = (fit_alpha, fit_loc, fit_beta)

            print "maxvalues", maxvalues
            rv = ss.lognorm(maxvalues[0], maxvalues[1], maxvalues[2])
            pl.figure()
            pl.plot(x, yorig/(np.sum(yorig)*(x[1]-x[0])), 'b')
            pl.plot(x, y, 'k')
            pl.plot(x,rv.pdf(x), 'r')

        pl.show()

    def fit_linear(self, x, y):
        A = np.array([x, np.ones(len(x))])
        w = np.linalg.lstsq(A.T, y)[0]
        yxp = w[0]*x + w[1]

        sstot = np.sum(np.multiply(y - np.mean(y), y - np.mean(y)))
        sserr = np.sum(np.multiply(y - yxp, y - yxp))
        r2lin = 1 - sserr/sstot
        #print "r2lin", r2lin
        
        return (w, r2lin, yxp)

    def plot_peaks_vs_time( self, type="homecage", out_path=None ):
        """
        Plot the maximum fluorescence value within each interaction event vs the start time
        of the event
        type can be "homecage", for novel object and social, where event times were hand scored
        or "sucrose", where event times are from the lickometer
        """
        type = self.exp_type

        if type == "homecagesocial" or type == "homecagenovel" or type == 'EPM':
            start_times = self.get_event_times("rising")
            end_times = self.get_event_times("falling")
        elif type == "sucrose":
            if self.event_start_times is None:
                start_times, end_times = self.get_sucrose_event_times()
                self.event_start_times = start_times
                self.event_end_times = end_times
            else:
                start_times = self.event_start_times
                end_times = self.event_end_times
        else:
            print "Experiment type not implemented. use --exp-type flag with 'sucrose', 'homecagenovel', or 'homecagesocial'."
            sys.exit(0)


        filt_start_times = []
        if start_times[0] != -1:
            peaks = np.zeros(len(start_times))
            for i in range(len(start_times)):
                if type == "sucrose":
                    peak = self.get_sucrose_peak(start_times[i], end_times[i])
                elif type == "homecagesocial" or type == "homecagenovel" or type == 'EPM':
                    peak = self.get_peak(start_times[i], end_times[i])
                else:
                    print "Experiment type not implemented. use --exp-type flag with 'sucrose', 'homecagenovel', or 'homecagesocial'."
                    sys.exit(0)

                if peak != 0:
                    peaks[i] = peak


            fig = pl.figure()
            ax = fig.add_subplot(111)
            print np.max(peaks) + .3
            if type == "sucrose":
                ax.set_ylim([0, np.max(peaks) + 0.4*np.max(peaks)])
            elif type == "homecagesocial" or type == "homecagenovel" or type == 'EPM':
                if np.max(peaks) > 0.8:
                    ax.set_ylim([0, 1.3])
                else:
                    ax.set_ylim([0, 1.1])
                ax.set_xlim([100, 500])
            else:
                ax.set_xlim([0, 1.1*np.max(start_times)])
           
            ax.plot(start_times, peaks, 'o')
            pl.xlabel('Time [s]')
            pl.ylabel('Fluorescence [dF/F]')
            pl.title('Peak fluorescence of interaction event vs. event start time')


            try:
                xp, pxp, x0, y0, c, k, r2, yxp = self.fit_exponential(start_times, peaks + 1)
                ax.plot(xp, pxp-1)
                ax.text(min(200, np.min(start_times)), np.max(peaks) + 0.20*np.max(peaks), "y = c*exp(-k*(x-x0)) + y0")
                ax.text(min(200, np.min(start_times)), np.max(peaks) + 0.15*np.max(peaks), "k = " + "{0:.2f}".format(k) + ", c = " + "{0:.2f}".format(c) + 
                                                ", x0 = " + "{0:.2f}".format(x0) + ", y0 = " + "{0:.2f}".format(y0) )
                ax.text(min(200, np.min(start_times)), np.max(peaks) + 0.1*np.max(peaks), "r^2 = " + str(r2))
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

    def compare_before_and_after_event(self, window=[5, 5], metric="slice", slice_time=1, out_path=None):
        """
        Compares the fluorescence in the periods before and the onset of each bout.
        Window is provided in seconds.
        Can measure that fluorescence based on "slice": the fluorescence value at slice_time
        before and after the onset; "area": the area under the curve in the window before and
        after the onset; "peak": the peak value in the window before and after the onset.

        """

        print window
        window_indices = [self.convert_seconds_to_index(window[0]), self.convert_seconds_to_index(window[1]) ]
        #get event times

        type = self.exp_type
        if type == 'sucrose':
            start_times, end_times = self.get_sucrose_event_times()
        elif type == 'homecagesocial' or type == 'homecagenovel' or type == 'EPM':
            start_times = self.get_event_times("rising")
            end_times = self.get_event_times("falling")
        else:
            print "Experiment type not implemented. use --exp-type flag with 'sucrose', 'homecagenovel', or 'homecagesocial'."
            sys.exit(0)


        before_start_scores = []
        after_start_scores = []

        #initialize result arrays



        if metric=="area":
            before_start_scores = self.get_areas_under_curve( start_times, window=[window[0], 0], baseline_window=window, normalize=False)
            after_start_scores = self.get_areas_under_curve( start_times, window=[0, window[1]], baseline_window=window, normalize=False)

            num_entries = min(len(before_start_scores), len(after_start_scores)) -1 #in case the event is right at the end of the time series 
            before_start_scores = before_start_scores[:num_entries]
            after_start_scores = after_start_scores[:num_entries]



        elif metric=="slice":
            for i in range(len(start_times)):
                t = start_times[i]
                ind = self.convert_seconds_to_index(t)
                if ind + window_indices[1] < len(self.fluor_data)-1 and ind - window_indices[0] > 0:
                
                    baseline = np.min(self.fluor_data[max(0, ind - window_indices[0]):min(len(self.fluor_data), ind + window_indices[1])])
                    slice_ind = self.convert_seconds_to_index(slice_time)
                    before_score = self.fluor_data[max(0, ind - slice_ind)] - baseline
                    after_score = self.fluor_data[min(len(self.fluor_data), ind + slice_ind)] - baseline
                    before_start_scores.append(before_score)
                    after_start_scores.append(after_score)

        elif metric=="peak":
            for i in range(len(start_times)):
                t = start_times[i]
                ind = self.convert_seconds_to_index(t)
                if ind + window_indices[1] < len(self.fluor_data)-1 and ind - window_indices[0] > 0:

                    baseline = np.min(self.fluor_data[max(0, ind - window_indices[0]):min(len(self.fluor_data), ind + window_indices[1])])
                    before_score = np.max(self.fluor_data[max(0, ind - window_indices[0]):ind]) - baseline
                    after_score = np.max(self.fluor_data[ind:min(len(self.fluor_data), ind + window_indices[1])]) - baseline
                   #  pl.plot(self.fluor_data[max(0, ind - window_indices[0]):min(len(self.fluor_data), ind + window_indices[1])])
                   #  pl.title('peak')
                   # # pl.show()
                    before_start_scores.append(before_score)
                    after_start_scores.append(after_score)  
        else:
            print "metric ' " + metric + " ' has not yet been implemented. Please try 'area', 'slice', or 'peak'."



        #---------------------------begin plot----------------------------------------------------------------------------
        pl.figure()
        before_plot, = pl.plot(range(len(before_start_scores)), before_start_scores, '-or')
        after_plot, = pl.plot(range(len(after_start_scores)), after_start_scores, '-og')
        pl.legend([before_plot, after_plot], [ "Before bout start", "After bout start"])
        pl.title("Window: [" + str(window[0]) + "s, " + str(window[1]) + "s]")
        pl.xlabel("Bout number")
        if metric=="area":
            pl.ylabel("Area under fluorescence curve [" + self.fluor_normalization + "]")
        elif metric=="slice":
            pl.ylabel("Fluorescence value at " + str(slice_time) + "s from bout start [" + self.fluor_normalization + "]")
        elif metric=="peak":
            pl.ylabel("Maximum fluorescence value [" + self.fluor_normalization + "]")
        

        if out_path is None:
            pl.title("No output path given")
            pl.show()
        else:
            save_path = out_path + "_before_and_after_window_" + str(window[0]) + "s_" + metric
            if metric == "slice":
                save_path += "_time_" + str(slice_time) + "s"
            pdf_save_path = save_path + ".pdf"
            png_save_path = save_path + ".png"
            npz_save_path = save_path + ".npz"
            txt_save_path = save_path + ".txt"


            pl.savefig(pdf_save_path)
            pl.savefig(png_save_path)
            np.savez(npz_save_path, before=before_start_scores, after=after_start_scores)
            np.savetxt(txt_save_path, (before_start_scores, after_start_scores))


       #---------------------------end plot----------------------------------------------------------------------------

        return (before_start_scores, after_start_scores)





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

    if FA.plot_type == 'tseries':
        if FA.time_range is None:
            FA.plot_basic_tseries(out_path = options.output_path + "_full_")
        else:
            # start = int(FA.time_range.split(':')[0])
            # end = int(FA.time_range.split(':')[1])
            # if end == -1:
            #     end = max(FA.time_stamps)
            # if start == -1:
            #     start = min(FA.time_stamps)

            # if end - start < 100:
            #     res = 1
            # elif end - start < 500 and end - start > 100:
            #     res = 10
            # elif end - start > 500 and end - start < 1000:
            #     res = 30
            # else:
            #     res = 40

            FA.plot_basic_tseries(out_path = options.output_path + "_" + str(int(FA.time_range.split(':')[0])) + 
                                    "_" + str(int(FA.time_range.split(':')[1])) +"_" ) #,  resolution=res)



#    FA.plot_next_event_vs_intensity(intensity_measure="peak", next_event_measure="onset", window=[0, 1], out_path=None)

#    FA.wavelet_plot()
#    FA.notch_filter(10.0, 10.3)
  #  FA.plot_periodogram(plot_type="log",out_path = options.output_path)

  
  #  FA.plot_lick_density_vs_time(1, out_path = options.output_path)
   # FA.get_sucrose_event_times(5, "falling")
#    FA.event_vs_baseline_barplot(out_path = options.output_path)

    

    #FA.plot_area_under_curve_wrapper( window=[0, 1], edge="rising", normalize=False, out_path = options.output_path)
    #FA.plot_peaks_vs_time(out_path = options.output_path)
 
  ####  FA.plot_peritrigger_edge(window=[1, 3], out_path = options.output_path + '_1_3_', edge="falling")
  ####  FA.plot_peritrigger_edge(window=[5, 5], out_path = options.output_path + '_5_5_', edge="falling")
  ####  FA.plot_peritrigger_edge(window=[30, 30], out_path = options.output_path + '_30_30_', edge="falling")
  ####  FA.plot_peritrigger_edge(window=[10, 30], out_path = options.output_path + '_10_30_', edge="falling")
  

  #  FA.plot_area_under_curve_wrapper( window=[0, 3], edge="rising", normalize=False, out_path = options.output_path)
  #  FA.plot_peaks_vs_time(out_path = options.output_path)
    #Dont use these anymore, use compare_before_and_after_event?

  #  FA.plot_peaks_vs_time(type="sucrose", out_path = options.output_path)
  #  FA.debleach(out_path = options.output_path) #you want to use --fluor-normalization = 'raw' when debleaching!!!
   # peak_inds, peak_vals, peak_times = FA.get_peaks(window_in_seconds)
  #  FA.plot_peak_data()

  #  FA.get_peaks_convolve([3, 3], out_path = options.output_path)
  ##  FA.fit_peak( type="sucrose", out_path=options.output_path)
###    FA.analyze_spectrum(type="sucrose", peak_index=None, out_path=None )

    # FA.compare_before_and_after_event(window=[5, 5], metric="slice", slice_time=1, out_path=options.output_path)
    # FA.compare_before_and_after_event(window=[5, 5], metric="slice", slice_time=2, out_path=options.output_path)

    # FA.compare_before_and_after_event(window=[5, 5], metric="area", out_path=options.output_path)
    # FA.compare_before_and_after_event(window=[5, 5], metric="peak", out_path=options.output_path)

##    FA.compare_before_and_after_event(window=[3, 3], metric="slice", slice_time=1, out_path=options.output_path)
##    FA.compare_before_and_after_event(window=[3, 3], metric="slice", slice_time=2, out_path=options.output_path)

 ##   FA.compare_before_and_after_event(window=[3, 3], metric="area", out_path=options.output_path)
 ##   FA.compare_before_and_after_event(window=[3, 3], metric="peak", out_path=options.output_path)






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
    parser.add_option("", "--time-range", dest="time_range",default='0:-1',
                      help="Specify a time window over which to analyze the time series in format start:end. -1 chooses the appropriate extremum")
    parser.add_option('-p', "--plot-type", default = '', dest="plot_type",
                      help="Type of plot to produce.")
    parser.add_option('', "--fluor-normalization", default = 'deltaF', dest="fluor_normalization",
                      help="Normalization of fluorescence trace. Can be a.u. between [0,1]: 'stardardize' or deltaF/F: 'deltaF' or 'raw'.")
    parser.add_option('-s', "--smoothness", default = 0, dest="smoothness",
                      help="Should the time series be smoothed, and how much.")
    parser.add_option('-x', "--selectfiles", action="store_true", default = False, dest = "selectfiles",
                       help="Should you select filepaths in a pop window instead of in the command line.")
    parser.add_option("", "--save-txt", action="store_true", default=False, dest="save_txt",
                      help="Save data matrix out to a text file.")
    parser.add_option("", "--save-to-h5", default=None, dest="save_to_h5",
                      help="Save data matrix to a dataset in an hdf5 file.")
    parser.add_option("", "--save-and-exit", action="store_true", default=False, dest="save_and_exit",
                      help="Exit immediately after saving data out.")
    parser.add_option("", "--save-debleach", action="store_true", default=False, dest="save_debleach",
                      help="Debleach fluorescence time series by fitting with an exponential curve.")
    parser.add_option("", "--filter-freqs", default=None, dest="filter_freqs",
                      help="Use a notch filter to remove high frequency noise. Format lowfreq:highfreq.")
    parser.add_option("", "--exp-type", dest="exp_type", default=None,
                       help="Specify either 'homecagenovel', 'homecagesocial', or 'sucrose', or 'EPM'.")
    parser.add_option("", "--event-spacing", dest="event_spacing", default=0,
                       help="Specify minimum time (in seconds) between the end of one event and the beginning of the next")
    parser.add_option("", "--mouse-type", dest="mouse_type", default="GC5",
                       help="Specify the type of virus injected in the mouse (GC5, GC3, EYFP)")


    (options, args) = parser.parse_args()
    
    # Test the class
    test_FiberAnalyze(options)
    
# EOF
