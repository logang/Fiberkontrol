import h5py
import numpy as np
import pylab as pl
import matplotlib as mpl
import statsmodels.api as sm
from matplotlib import cm

from fiber_record_analyze import FiberAnalyze

#-------------------------------------------------------------------------------------------

if __name__ == "__main__":
    all_data = h5py.File("/Users/logang/Documents/Results/FiberRecording/Cell/all_data_raw.h5",'r')

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

    # --- Plot data --- #
    exp_type = 'homecagesocial'
    time_window = [0,1] # [before,after] event in seconds 

    fig = pl.figure()
    ax = fig.add_subplot(1,1,1)

    i=0 # color counter
    for animal_id in all_data.keys():
        # load data from hdf5 file by animal-date-exp_type
        animal = all_data[animal_id]
        for dates in animal.keys():
            date = animal[dates]
            FA = FiberAnalyze( options )
            FA.subject_id = animal_id
            FA.exp_date = dates
            FA.exp_type = exp_type
            FA.load(file_type="hdf5")

            # get intensity and next_val values for this animal
            peak_intensity, onset_next_vals = FA.plot_next_event_vs_intensity(intensity_measure="integrated", next_event_measure="onset", window=time_window, out_path=None, plotit=False)\

            # fit a robust regression
            if len(onset_next_vals) > 0:
                X = np.vstack( (np.log(peak_intensity), np.ones((len(onset_next_vals),))) )
#                rlm_model = sm.RLM(np.log(onset_next_vals), X.T, M=sm.robust.norms.TukeyBiweight())
#                lm_results = rlm_model.fit()
                lm_model = sm.OLS(np.log(onset_next_vals), X.T)
                lm_results = lm_model.fit()
                print "-------------------------------------"
                print animal_id, dates
                try:
                    print "\t--> Slope:",lm_results.params[0]
#                    print "\t--> Intercept:",lm_results.params[1]
#                    print "\t--> Confidence Interval for Slope:", lm_results.conf_int()[0,:]
                    print "\t--> P-value for Slope:", lm_results.pvalues[0]
#                    print "\t--> Confidence Interval for Intercept:", lm_results.conf_int()[1,:]
#                    print "\t--> P-value Interval for Intercept:", lm_results.pvalues[1]
                    print "\t--> R-squared", lm_results.rsquared
                except:
                    pass
                try:
                    print "\t--> R-squared Adjusted:", lm_results.rsquared_adj
                except:
                    print "\t--> Could not calculate adjusted R-squared."
                yhat = lm_results.fittedvalues

#                fig = pl.figure()
#                ax = fig.add_subplot(1,1,1)
#                ax.loglog(peak_intensity,onset_next_vals,'o')

#                ax.plot(np.log(peak_intensity), np.log(onset_next_vals),'o',color=cm.jet(float(i)/10.))
                ax.plot(peak_intensity, onset_next_vals,'o',color=cm.jet(float(i)/10.))
#                ax.plot(np.log(peak_intensity), yhat, '-', color=cm.jet(float(i)/10.) )

#                pl.show()
                i+=1 # increment color counter
            else:
                #ax.plot(np.log(peak_intensity), np.log(onset_next_vals),'o')
                print "No values to plot for", animal_id, dates, exp_type

#    pl.xlabel("log peak intensity in first second after interaction onset")
    pl.xlabel("log integrated intensity in first second after interaction onset")
    
    pl.ylabel("log time until next interaction")
#    pl.ylabel("log length of next interaction")
    pl.show()

#----------------------------------------------------------------------------------------   
