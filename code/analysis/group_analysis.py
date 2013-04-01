import h5py
import numpy as np
import pylab as pl
import matplotlib as mpl
#import statsmodels.api as sm
from matplotlib import cm
import matplotlib.pyplot as plt
import os
import sys
import scipy.stats as stats

from fiber_record_analyze import FiberAnalyze

def group_regression_plot(all_data, options, exp_type='homecagesocial', time_window=[-1,1] ):

    """
    Plot fits of regression lines on data from home cage or novel social 
    fiber photometry data, with points corresponding to bout and bout number.

    TODO: write better description of function. Clean up code. Have figure save to output_path. 
    """

    # Create figure
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
            peak_intensity, onset_next_vals = FA.plot_next_event_vs_intensity(intensity_measure="integrated", next_event_measure="onset", window=time_window, out_path=None, plotit=False)

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

def group_bout_heatmaps(all_data, options, exp_type, time_window, 
                        df_max=0.35, event_edge="rising"):
    """
    Save out 'heatmaps' showing time on the x axis, bouts on the y axis, and representing signal
    intensity with color.
    """
    i=0 # color counter
    for animal_id in all_data.keys():
        # load data from hdf5 file by animal-date-exp_type
        animal = all_data[animal_id]
        for dates in animal.keys():
            # Create figure
            fig = pl.figure()
            ax = fig.add_subplot(2,1,1)
            
            date = animal[dates]
            FA = FiberAnalyze( options )
            FA.subject_id = animal_id
            FA.exp_date = dates
            FA.exp_type = exp_type
            print animal_id, " ", dates, " ", exp_type
            if exp_type in animal[dates].keys():
                if(FA.load(file_type="hdf5") != -1):
                    event_times = FA.get_event_times(event_edge, int(options.event_spacing))
                    print "len(event_times)", len(event_times)
                    time_arr = np.asarray( FA.get_time_chunks_around_events(FA.fluor_data, 
                                                                            event_times, 
                                                                            time_window) )

                    # Generate a heatmap of activity by bout, with range set 
                    # between the 5% quantile of the data and the 'df_max' argument 
                    # of the function
                    from scipy.stats.mstats import mquantiles
                    baseline = mquantiles( time_arr.flatten(), prob=[0.05])
                    ax.imshow(time_arr, interpolation="nearest",vmin=baseline,
                              vmax=df_max,cmap=pl.cm.afmhot, 
                              extent=[-time_window[0], time_window[1], 0, time_arr.shape[0]])
                    ax.set_aspect('auto')
                    pl.title("Animal #: "+animal_id+'   Date: '+dates)
                    pl.ylabel('Bout Number')
                    ax.axvline(0,color='white',linewidth=2,linestyle="--")

                    ax = fig.add_subplot(2,1,2)
                    FA.plot_perievent_hist(event_times, time_window, 
                                           out_path=None, plotit=True, subplot=ax )
                    pl.ylim([0,df_max])

                    if options.output_path is not None:
                        import os
                        outdir = os.path.join(options.output_path, options.exp_type)
                        if not os.path.isdir(outdir):
                            os.makedirs(outdir)
                        pl.savefig(outdir+'/'+animal_id+'_'+dates+'.png')
                        print outdir+'/'+animal_id+'_'+dates+'.png'
                    else:
                        pl.show()

def group_bout_ci(all_data, options, exp_type, time_window, 
                       df_max=0.35, event_edge="rising"):
    """
    Save out plots of mean or median activity with confidence intervals. 

    """
    i=0 # color counter
    exp_types = ['homecagesocial', 'homecagenovel']
    for animal_id in all_data.keys():
        # load data from hdf5 file by animal-date-exp_type
        animal = all_data[animal_id]
        for dates in animal.keys():
            # Create figure
            fig = pl.figure()
            ax = fig.add_subplot(1,1,1)
            for exp_type in exp_types:
                date = animal[dates]
                FA = FiberAnalyze( options )
                FA.subject_id = animal_id
                FA.exp_date = dates
                FA.exp_type = exp_type
                print animal_id, " ", dates, " ", exp_type
                median_time_series = []
                if exp_type in animal[dates].keys():
                    if(FA.load(file_type="hdf5") != -1):
                        event_times = FA.get_event_times(event_edge, int(options.event_spacing))
                        print "len(event_times)", len(event_times)
                        time_arr = np.asarray( FA.get_time_chunks_around_events(FA.fluor_data, 
                                                                                event_times, 
                                                                                time_window) )

                        # Generate a heatmap of activity by bout, with range set 
                        # between the 5% quantile of the data and the 'df_max' argument 
                        # of the function
                        median_time_series.append( np.median(time_arr, axis=0) )

            ax.plot(median_time_series)
            ax.set_aspect('auto')
            pl.title("Animal #: "+animal_id+'   Date: '+dates)
            pl.ylabel('Bout Number')
            ax.axvline(0,color='white',linewidth=2,linestyle="--")

            if options.output_path is not None:
                import os
                outdir = options.output_path
                if not os.path.isdir(outdir):
                    os.makedirs(outdir)
                pl.savefig(outdir+'/'+animal_id+'_'+dates+'.png')
                print outdir+'/'+animal_id+'_'+dates+'.png'
            else:
                pl.show()

def group_plot_time_series(all_data, options):
    """
    Save out time series for each trial, overlaid with 
    red lines indicating event epochs
    """
    for animal_id in all_data.keys():
        # load data from hdf5 file by animal-date-exp_type
        animal = all_data[animal_id]
        for date in animal.keys():

            for exp_type in animal[date].keys():
            
                FA = FiberAnalyze( options )
                FA.subject_id = animal_id
                FA.exp_date = date
                FA.exp_type = exp_type
                FA.fluor_normalization = 'deltaF'
                FA.time_range = '0:-1'

                print animal_id, " ", date, " ", exp_type
                if(FA.load(file_type="hdf5") != -1):
                    dir = options.output_path + '/' + FA.exp_type
                    print dir
                    if os.path.isdir(dir) is False:
                        os.makedirs(dir)
                    FA.plot_basic_tseries(out_path = dir + '/' + FA.subject_id + "_" + 
                                          FA.exp_date + "_" + FA.exp_type + "_" + 
                                          str(int(FA.time_range.split(':')[0])) +  "_" + 
                                          str(int(FA.time_range.split(':')[1])) +"_" ) 


def plot_representative_time_series(options, representative_time_series_specs_file):
    """
    Save out plots of time series overlaid with bars indicating event times 
    (i.e. sucrose lick or social interaction) for all trials listed in 
    representative_time_series_specs.txt. This function can be used to provide 
    multiple levels of detail of the same time series (i.e. to 'zoom' in) The 
    format of representative_time_series_specs.txt is:
    animal#     date        start   end exp_type    smoothness      

    """
#    all_data = h5py.File(options.input_path,'r') #just for testing

    print representative_time_series_specs_file
    f = open( representative_time_series_specs_file, "r" )
    f.readline() #skip the first header line

    for line in f:
        specs = line.split()
        if specs != []:
            animal_id = specs[0]
            date = specs[1]
            start = specs[2]
            end = specs[3]
            exp_type = specs[4]
            smoothness = specs[5]

            FA = FiberAnalyze( options )
            FA.subject_id = str(animal_id)
            FA.exp_date = str(date)
            FA.exp_type = str(exp_type)
            FA.smoothness = int(smoothness)
            FA.time_range = str(start) + ':' + str(end)
            FA.fluor_normalization = 'deltaF'

           # print "Test Keys: ", all_data[str(421)][str(20121008)][FA.exp_type].keys()

            print ""
            print "--> Plotting: ", FA.subject_id, FA.exp_date, FA.exp_type, FA.smoothness, FA.time_range
            if(FA.load(file_type="hdf5") != -1):
                dir = options.output_path + '/' + FA.exp_type
                if os.path.isdir(dir) is False:
                    os.makedirs(dir)
                FA.plot_basic_tseries(out_path = dir + '/' + FA.subject_id + "_" +
                                      FA.exp_date + "_" + FA.exp_type + "_" + 
                                      str(int(FA.time_range.split(':')[0])) +  
                                      "_" + str(int(FA.time_range.split(':')[1])) +"_" ) 


def get_novel_social_pairs(all_data, exp1, exp2):
    """
    all_data = an hdf5 file containing all of the time series data
    exp1 and exp2 = the two experiment types to be compared (i.e. homecagesocial and homecagenovel)

    Returns: a dict where the keys are animal_ids
    and each entry is a dict containing for each exp_type (the keys), 
    the entry is the date of the best trial of that behavior
    """
    pairs = dict()

    for animal_id in all_data.keys():
        # load data from hdf5 file by animal-date-exp_type

        animal = all_data[animal_id]
        if animal.attrs['mouse_type'] == 'GC5': #don't use EYFP or GC3
            pairs[animal_id] = dict()
            for date in animal.keys(): 
                #make a list for each experiment type of all dates on which that exp was run
                for exp_type in animal[date].keys(): 
                    if exp_type == str(exp1) or exp_type == str(exp2):
                        if exp_type in pairs[animal_id].keys():
                            pairs[animal_id][exp_type].append(int(date))
                        else:
                            pairs[animal_id][exp_type] = [int(date)]

    #print "Pairs before max filter: ", pairs
                    
    #As a heuristic to choose between multiple trials of the same exp_type,
    # use the one with the latest (largest) date                
    for animal_id in pairs.keys():
        if pairs[animal_id].keys() == []:
            del pairs[animal_id]
        else:
            for exp_type in pairs[animal_id].keys():
                pairs[animal_id][exp_type] = np.max(pairs[animal_id][exp_type])

    print "Pairs after max filter: ", pairs

    return pairs


def score_of_chunks(ts_arr, metric='area'):
    """
    Given an array of time series chunks, return an array
    holding a score for each of these chunks

    metric can be 'area' (area under curve) or 'peak' (peak fluorescence value)
    """

    scores = []
    for ts in ts_arr:
        if metric == 'area':
            scores.append(np.sum(ts))
        elif metric == 'peak':
            scores.append(np.max(ts))

    return scores



def compare_start_and_end_of_epoch(all_data, options, time_window=[0,0.25], metric='area', test='ttest', plot_perievent=False):

    """
    Calculates the difference between the fluorescence in a window at the beginning
    and the end of each epoch.
    Compares these differences between novel object and social behaviors.
    Plots the average difference vs. epoch number for novel and social (ideally on the same plot)
    Returns the t-test score comparing novel and social.

    metric can be 'area' (area under curve) or 'peak' (peak fluorescence value)
    """

    exp1 = 'homecagesocial'
    exp2 = 'homecagenovel'
    pairs = get_novel_social_pairs(all_data, exp1, exp2) #can use any function here that returns pairs of data

    pair_scores = dict() #key: animal_id, entry: a dict storing an array of scores for each trial type (i.e. homecagenovel and homecagesocial)
    pair_avg_scores = dict() #key: animal_id, entry: a dict storing the average score within a trial for each trial type (i.e. homecagenovel and homecagesocial)

    for animal_id in pairs.keys():
        pair_scores[animal_id] = dict()
        pair_avg_scores[animal_id] = dict()
        for exp_type in pairs[animal_id].keys():

            FA = FiberAnalyze( options )
            FA.subject_id = str(animal_id)
            FA.exp_date = str(pairs[animal_id][exp_type])
            FA.exp_type = str(exp_type)
            print FA.subject_id, " ", FA.exp_date, " ", FA.exp_type
            if(FA.load(file_type="hdf5") != -1):

                start_event_times = FA.get_event_times("rising", int(options.event_spacing))
                end_event_times = FA.get_event_times("falling", int(options.event_spacing))

                #--Get an array of time series chunks in a window around each event time
                #reverse_window = [time_window[1], time_window[0]]
                start_time_arr = np.asarray( FA.get_time_chunks_around_events(FA.fluor_data, 
                                                                              start_event_times, 
                                                                              time_window, 
                                                                              baseline_window=-1))

                #before_time_arr = np.asarray( FA.get_time_chunks_around_events(FA.fluor_data, end_event_times, reverse_window, baseline_window=-1 ))

                end_time_arr = np.asarray( FA.get_time_chunks_around_events(FA.fluor_data, 
                                                                            end_event_times, 
                                                                            time_window, 
                                                                            baseline_window=-1 ))

                start_scores = np.array(score_of_chunks(start_time_arr, metric))
                #before_scores = np.array(score_of_chunks(before_time_arr, metric))
                end_scores = np.array(score_of_chunks(end_time_arr, metric))
                scores_diff = end_scores - start_scores
                #scores_diff = end_scores - before_scores

                pair_scores[animal_id][exp_type] = scores_diff
                pair_avg_scores[animal_id][exp_type] = np.mean(scores_diff)

                if(plot_perievent):
                    fig = plt.figure()
                    ax = fig.add_subplot(2,1,1)
                    title = FA.subject_id + ' ' + FA.exp_date + ' ' + FA.exp_type
                    ax.set_title(title)
                    FA.plot_perievent_hist(start_event_times, time_window, 
                                           out_path=options.output_path, plotit=True, 
                                           subplot=ax, baseline_window=-1  )

                    #FA.plot_perievent_hist(end_event_times, reverse_window, out_path=options.output_path, plotit=True, subplot=ax, baseline_window=-1 )

                    ax = fig.add_subplot(2,1,2)
                    FA.plot_perievent_hist(end_event_times, time_window, 
                                           out_path=options.output_path, plotit=True, 
                                           subplot=ax, baseline_window=-1 )
                    if options.output_path is None:
                        pl.show()
                    else:
                        print "Saving peri-event time series..."
                        pl.savefig(options.output_path + FA.subject_id + '_' + FA.exp_date + 
                                   '_' + FA.exp_type + "_perievent_tseries.png")


    #-- Next, plot all of the pair_scores (vs animal, for now)

    #-- Create two matched arrays (one for novel, one for social) with the avg score for each animal
    exp_scores = dict() #key: exp_type, entry: array of avg score for each animal
    for animal_id in pair_avg_scores.keys():
        for exp_type in pair_avg_scores[animal_id].keys():
            print animal_id, exp_type
            if exp_type in exp_scores.keys():
                exp_scores[exp_type].append(pair_avg_scores[animal_id][exp_type])
            else:
                exp_scores[exp_type] = [pair_avg_scores[animal_id][exp_type]]

    print "Exp_scores", exp_scores
    plt.close('all')
    plt.figure()
    p1, = plt.plot(exp_scores[exp1], 'o')
    p2, = plt.plot(exp_scores[exp2], 'o')
    plt.legend([p1, p2], [exp1, exp2])
    plt.xlabel('Mouse (one mouse per column)')
    plt.ylabel( 'End - Start (' + metric +' w/in ' + str(time_window[1]) + 's window, avged across epochs)')
    plt.title('More fluorescence after end than after start of interaction with novel object ')
    pl.savefig(options.output_path + str(time_window[1]) + '_' + metric+ '_end_minus_start.png')


    if test == "ttest":
            [tvalue, pvalue] = stats.ttest_rel(exp_scores[exp1], exp_scores[exp2])
            print "normalized area tvalue: ", tvalue, " normalized area pvalue: ", pvalue
    if test == "wilcoxon":
            [zstatistic, pvalue] = stats.wilcoxon(exp_scores[exp1], exp_scores[exp2])
            print "normalized area zstatistic: ", zstatistic, " normalized area pvalue: ", pvalue

    plt.show()





#For each mouse, open the novel and social files
#then use FA.get_time_chunks_around_events to get 
# the time series around each event (with some spacing parameter enfored?)
# both around the beginning and falling edges
# Now write a separate function that compares some time period (can use area or peak) in each of these
# two sets of chunks (see FA.compare_before_and_after_event())
# output two matched arrays which you can run through a t test, one entry for each mouse



#-------------------------------------------------------------------------------------------

if __name__ == "__main__":

    # Parse command line options
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-o", "--output-path", dest="output_path", default=None,
                      help="Specify the ouput path.")

    parser.add_option("-t", "--trigger-path", dest="trigger_path", default=None,
                      help=("Specify path to files with trigger times, minus the '_s.npz' "
                            "and '_e.npz' suffixes."))

    parser.add_option("-i", "--input-path", dest="input_path",
                      help="Specify the input path.")

    parser.add_option("", "--time-range", dest="time_range",default=None,
                      help=("Specify a time window over which to analyze the time series "
                            "in format start:end. -1 chooses the appropriate extremum"))

    parser.add_option('-p', "--plot-type", default = 'tseries', dest="plot_type",
                      help="Type of plot to produce.")

    parser.add_option('', "--fluor_normalization", default = 'deltaF', dest="fluor_normalization",
                      help=("Normalization of fluorescence trace. Can be a.u. between [0,1]: "
                            "'stardardize' or deltaF/F: 'deltaF'."))

    parser.add_option('-s', "--smoothness", default = 0, dest="smoothness",
                      help="Should the time series be smoothed, and how much.")

    parser.add_option('-x', "--selectfiles", default = False, dest = "selectfiles",
                       help=("Should you select filepaths in a pop window instead of in the "
                             "command line."))

    parser.add_option("", "--save-txt", action="store_true", default=False, dest="save_txt",
                      help="Save data matrix out to a text file.")

    parser.add_option("", "--save-to-h5", default=None, dest="save_to_h5",
                      help="Save data matrix to a dataset in an hdf5 file.")

    parser.add_option("", "--save-and-exit", action="store_true", default=False, 
                      dest="save_and_exit", help="Exit immediately after saving data out.")

    parser.add_option("", "--filter-freqs", default=None, dest="filter_freqs",
                      help=("Use a notch filter to remove high frequency noise. Format "
                            "lowfreq:highfreq."))

    parser.add_option("", "--save-debleach", action="store_true", default=False, 
                      dest="save_debleach", help=("Debleach fluorescence time series by "
                                                  "fitting with an exponential curve."))

    parser.add_option('', "--exp-type", default = 'homecagesocial', dest="exp_type",
                      help=("Which type of experiment. Current options are 'homecagesocial' "
                            "and 'homecagenovel'"))

    parser.add_option("", "--time-window", dest="time_window",default='3:3',
                      help="Specify a time window for peri-event plots in format before:after.")

    parser.add_option("", "--event-spacing", dest="event_spacing", default=0,
                       help=("Specify minimum time (in seconds) between the end of one event "
                             "and the beginning of the next"))

    parser.add_option("", "--mouse-type", dest="mouse_type", default="GC5",
                       help="Specify the type of virus injected in the mouse (GC5, GC3, EYFP)")


    parser.add_option("", "--representative-time-series-specs-file", 
                      dest="representative_time_series_specs_file", 
                      default='representative_time_series_specs.txt',
                      help=("Specify file of representative trials to plot. File in format: "
                            "animal# date start_in_secs end_in_secs exp_type smoothness"))

    
    (options, args) = parser.parse_args()

    # --- Plot data --- #

#    all_data = h5py.File("/Users/logang/Documents/Results/FiberRecording/Cell/all_data_raw.h5",'r')
    all_data = h5py.File(options.input_path,'r')

    time_window = np.array(options.time_window.split(':'), dtype='float32') # [before,after] event in seconds 
#    group_regression_plot(all_data, options, exp_type=exp_type, time_window=time_window)
#    group_bout_heatmaps(all_data, options, exp_type=options.exp_type, time_window=time_window)
    group_bout_heatmaps(all_data, options, time_window=time_window)
##    group_plot_time_series(all_data, options)
##    plot_representative_time_series(options, options.representative_time_series_specs_file)
    compare_start_and_end_of_epoch(all_data, options, time_window=[0, .1], metric='peak', test='ttest')


#----------------------------------------------------------------------------------------   
