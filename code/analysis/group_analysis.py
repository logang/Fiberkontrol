import h5py
import numpy as np
import pylab as pl
import matplotlib as mpl
import statsmodels.api as sm
from matplotlib import cm
import matplotlib.pyplot as plt
import os
import sys
import scipy as sp
import scipy.stats as stats
from state_space import denoise
import pickle

import fiber_record_analyze as fra 
from fiber_record_analyze import FiberAnalyze

def group_iter_list(all_data, options, 
                    exp_type=None, 
                    animal_id=None, 
                    exp_date=None, 
                    mouse_type=None):
    """
    Returns a list where each
    entry is an array containing
    [animal_id, date, exp_type].
    Only includes experiments that 
    match options.exp_date,
    options.exp_type, and
    options.mouse_type

    One can then iterate over this list
    to analyze a group of mice

    TODO: This has not yet been incorporated into all of the rest of the code
    """

    iter_list = []
    animal_id_list = []
    date_list = []
    exp_type_list = []

    if exp_date is None:
        exp_date = options.exp_date
    if exp_type is None:
        exp_type = options.exp_type
    if mouse_type is None:
        mouse_type = options.mouse_type

    for animal_id_key in all_data.keys():
        if animal_id is None or animal_id_key == animal_id:
            animal = all_data[animal_id_key]
            if animal.attrs['mouse_type'] == mouse_type:
                for date_key in animal.keys():
                    if exp_date is None or exp_date == date_key:
                        for exp_type_key in animal[date_key].keys():
                            if exp_type is None or exp_type_key == exp_type:
                                iter_list.append({'animal_id':animal_id_key, 
                                                  'date':date_key, 
                                                  'exp_type':exp_type_key})
                                if animal_id_key not in animal_id_list:
                                    animal_id_list.append(animal_id_key)
                                if date_key not in date_list:
                                    date_list.append(date_key)
                                if exp_type_key not in exp_type_list:
                                    exp_type_list.append(exp_type_key)

    print "iter_list", [iter_list, animal_id_list, date_list, exp_type_list]
    return [iter_list, animal_id_list, date_list, exp_type_list]


def loadFiberAnalyze(FA, options, animal_id, exp_date, exp_type):
    """
    Load an instance of the fiberAnalyze class, initialized
    to an experimental trial identified by the id# of the animal,
    the date of the experiment, and the type of the experiment (i.e. 
        homecagesocial or homecagenovel)
    """

    FA.subject_id = str(animal_id)
    FA.exp_date = str(exp_date)
    FA.exp_type = str(exp_type)
    print FA.subject_id, " ", FA.exp_date, " ", FA.exp_type
    try:
        success = FA.load(file_type="hdf5") 
    except:
        success = -1
    if(success != -1):
        print "denoise"
        FA.fluor_data = np.asarray(denoise(FA.fluor_data))

    return [FA, success]


def group_regression_plot(all_data, 
                          options, 
                          exp_type, 
                          time_window,
                          metric,
                          ):
    """
    Plot fits of regression lines on data from home cage or novel social 
    fiber photometry data, with points corresponding to bout and bout number.

    TODO: write better description of function. Clean up code. Have figure save to output_path. 
    """

    # Create figure
    fig = pl.figure()
    ax = fig.add_subplot(1,1,1)

    i=0 # color counter
    num_animals = 12.
    num_bouts = 10
    slopes = []
    next_event_measure = "onset"

    [iter_list, animal_id_list, date_list, exp_type_list] = group_iter_list(all_data, options)

    for exp in iter_list:
        animal_id = exp['animal_id']
        dates = exp['date']
        exp_type = exp['exp_type']
        print "animal_id", animal_id


        FA = FiberAnalyze(options)
        [FA, success] = loadFiberAnalyze(FA,
                                         options, 
                                         animal_id, 
                                         dates, 
                                         exp_type)


        # get intensity and next_val values for this animal
        if success != -1:
            print "metric", metric
            print "time_window", time_window
            peak_intensity, onset_next_vals = FA.plot_next_event_vs_intensity(intensity_measure=metric, 
                                                                            next_event_measure=next_event_measure, 
                                                                            window=time_window, 
                                                                            out_path=None, 
                                                                            plotit=False,
                                                                            baseline_window=-1)
            # fit a robust regression
            if len(onset_next_vals) > 0:
                X = np.nan_to_num( np.vstack( ( np.log(peak_intensity), np.ones((len(onset_next_vals),))) ) )
                print "X", X.T
                if X.shape[1] > 2:
                

                    X = X[0:min(num_bouts,X.shape[1]),:]
                    y = np.log(onset_next_vals)
                    lm_model = sm.OLS(y, X.T)
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
                    print "yhat", yhat
                    slopes.append( lm_results.params[0] )
    #                fig = pl.figure()
    #                ax = fig.add_subplot(1,1,1)
    #                ax.loglog(peak_intensity,onset_next_vals,'o')

                    ax.plot(np.log(peak_intensity), np.log(onset_next_vals),'o',color=cm.jet(float(i+1)/num_animals))
    #                ax.plot(peak_intensity, onset_next_vals,'o',color=cm.jet(float(i)/10.))
                    ax.plot(np.log(peak_intensity), yhat, '-', color=cm.jet(float(i+1)/num_animals) )

    #                pl.show()
                    i+=1 # increment color counter
                else:
                    #ax.plot(np.log(peak_intensity), np.log(onset_next_vals),'o')
                    print "No values to plot for", animal_id, dates, exp_type

    if time_window[0] == 0 and time_window[1] == 0:
        pl.xlabel("log " + str(metric) + " during interaction event")
    else:
        pl.xlabel("log " + str(metric) + " in first " + str(time_window[1]) + " seconds after interaction onset")
    
    if next_event_measure == "onset":
        pl.ylabel("Log time in seconds until next interaction")
    elif next_event_measure == "length":
       pl.ylabel("log length of next interaction")
    pl.title(options.mouse_type + " " + options.exp_type + " " + options.intensity_metric)

    if options.output_path is not None:
        outdir = options.output_path + '/' + options.exp_type
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        outpath = outdir +'/'+str(dates)+'_'+'_' + str(options.exp_type) + '_' + str(options.mouse_type) +'_' + str(metric) + '_window_'+str(time_window[0])+'_'+str(time_window[1])+ options.plot_format
        pl.savefig(outpath)
        print "outpath: ", outpath
    else:
        pl.show()

    print "Slopes:",slopes
    return slopes, peak_intensity, onset_next_vals, lm_results


#------------------------------------------------------------------------------

def print_spike_times(all_data,
                      options,
                      exp_type,
                      max_num_epochs,
                      time_window,
                      event_edge="rising",
                      baseline_window='full',):
    """
    Save a text file that lists, for each mouse trial,
    the time (since the beginning of the trial, in seconds),
    of the peak fluorescence in the time window (i.e. [3,3])
    surrounding the beginning of each interaction event. 

    Return dicts (keys: animalid_date_exptype) where each 
    key corresponds to a trial, and contains an array with 
    the peak times, the peak fluorescence values, the 
    time of the peak with respect to the behavior interaction
    start time, and the behavior interaction end time. 
    """

    [iter_list, animal_id_list, date_list, exp_type_list] = group_iter_list(all_data, options)

    all_peak_times_dict = dict()
    all_peak_vals_dict = dict()
    all_local_times_dict = dict()
    all_interaction_start_times_dict = dict()
    all_interaction_end_times_dict = dict()


    for exp in iter_list:
        animal_id = exp['animal_id']
        date = exp['date']
        exp_type = exp['exp_type']

        dict_key = str(animal_id) + '_' + str(date) + '_' + str(exp_type)
        print "dict_key", dict_key
        all_peak_times_dict[dict_key] = np.array([])
        all_peak_vals_dict[dict_key] = np.array([])
        all_local_times_dict[dict_key] = np.array([])
        all_interaction_start_times_dict[dict_key] = np.array([])
        all_interaction_end_times_dict[dict_key] = np.array([])


        FA = FiberAnalyze(options)
        [FA, success] = loadFiberAnalyze(FA,
                                         options, 
                                         animal_id, 
                                         date, 
                                         exp_type)

        if(success!=-1):
            event_times = FA.get_event_times(edge="rising", 
                                             exp_type=exp_type)

            end_event_times = FA.get_event_times(edge="falling", 
                                     exp_type=exp_type)
            

            if len(event_times) > 0:
                if max_num_epochs > 0:
                    event_times = event_times[0:max_num_epochs]

                print "baseline_window", baseline_window

                end_times = None
                if time_window[0] == 0 and time_window[1] == 0:
                    end_times = end_event_times

                time_arr = np.asarray( FA.get_time_chunks_around_events(
                                            FA.fluor_data, 
                                            event_times, 
                                            time_window, 
                                            baseline_window=baseline_window,
                                            end_event_times = end_times) )

                for i in range(len(time_arr)):
                    chunk = time_arr[i]
                    event_time = event_times[i]
                    end_event_time = end_event_times[i]

                    max_index = np.argmax(chunk)
                    max_val = np.amax(chunk)
                    max_time = FA.time_stamps[FA.convert_seconds_to_index(event_time - time_window[0]) 
                                                + max_index]
                    all_peak_times_dict[dict_key] = np.append(all_peak_times_dict[dict_key], 
                                                              max_time)
                    all_local_times_dict[dict_key] = np.append(all_local_times_dict[dict_key], 
                                                              max_time - event_time)
                    all_peak_vals_dict[dict_key] = np.append(all_peak_vals_dict[dict_key], 
                                                              max_val)
                    all_interaction_start_times_dict[dict_key] = np.append(
                                                    all_interaction_start_times_dict[dict_key],
                                                    event_time)
                    all_interaction_end_times_dict[dict_key] = np.append(
                                                    all_interaction_end_times_dict[dict_key],
                                                    end_event_time)

    if options.output_path is not None:
        import os
        import csv
        outdir = os.path.join(options.output_path, options.exp_type)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        ## Save file as a pickle for easy loading back into python
        pickle_path = options.output_path+'/'+'list_of_event_times_'+options.mouse_type+'_'+ options.exp_type+'.pkl'
        output = open(pickle_path, 'wb')
        pickle.dump(all_peak_times_dict, output)
        pickle.dump(all_peak_vals_dict, output)
        pickle.dump(all_local_times_dict, output)
        pickle.dump(all_interaction_start_times_dict, output)
        pickle.dump(all_interaction_end_times_dict, output)
        output.close()

        ## Save file in a human readable format for easy manual checking
        path = outdir+'/'+'list_of_event_times_'+options.mouse_type+'_'+ options.exp_type+'.txt'
        f = open(path,'w')
        f.write('time,\t fluor_value, \t\ttime_relative_to_event_time\tinteraction_start\tinteraction_end\n\n')
        for key in all_peak_times_dict.keys():
            f.write('\n' + str(key) + ',\n')
            for i in range(len(all_peak_times_dict[key])):
                time = all_peak_times_dict[key][i]
                val = all_peak_vals_dict[key][i]
                local = all_local_times_dict[key][i]
                start_time = all_interaction_start_times_dict[key][i]
                end_time = all_interaction_end_times_dict[key][i]
                f.write("%.2f" % time + ',\t' + "%.2f" % val + ',\t\t' + "%.2f" % local
                        +',\t' + "%.2f" % start_time+',\t' + "%.2f" % end_time+'\n')

        f.close()
        print path





def group_bout_heatmaps(all_data, 
                        options, 
                        exp_type, 
                        time_window, 
                        max_num_epochs,
                        ymax_setting,
                        df_max=0.35, 
                        event_edge="rising",
                        baseline_window='full', ):
    """
    Save out 'heatmaps' showing time on the x axis, 
    bouts on the y axis, and representing signal
    intensity with color.
    Below the heatmap for each mouse, plot the peri-event
    time series.

    ymax_setting = 'large' or 'small' allows you to zoom in on
        weaker responding mice

    """

    i=0 # color counter

    [iter_list, animal_id_list, date_list, exp_type_list] = group_iter_list(all_data, options)

    for exp in iter_list:
        animal_id = exp['animal_id']
        date = exp['date']
        exp_type = exp['exp_type']


        FA = FiberAnalyze(options)
        [FA, success] = loadFiberAnalyze(FA,
                                         options, 
                                         animal_id, 
                                         date, 
                                         exp_type)
        
        [df_max, df_min] = FA.get_plot_ylim(exp_type, 
                                            FA.fluor_normalization,
                                            ymax_setting,
                                            np.max(FA.fluor_data))


        if(success!=-1):
            event_times = FA.get_event_times(edge=event_edge, 
                                             #nseconds=float(options.event_spacing), 
                                             exp_type=exp_type)
            print "len(event_times)", len(event_times)
            if len(event_times) > 0:
                if max_num_epochs > 0:
                    event_times = event_times[0:max_num_epochs]

                print "baseline_window", baseline_window
                time_arr = np.asarray( FA.get_time_chunks_around_events(
                                            FA.fluor_data, 
                                            event_times, 
                                            time_window, 
                                            baseline_window=baseline_window) )

                # Generate a heatmap of activity by bout, with range set between the 5% quantile of
                # the data and the 'df_max' argument of the function
                fig = pl.figure()
                ax = fig.add_subplot(2,1,1)

                from scipy.stats.mstats import mquantiles
                baseline = mquantiles( time_arr.flatten(), prob=[0.05])
                ax.imshow(time_arr, 
                          interpolation="nearest",
                          vmin=baseline,
                          vmax=df_max,
                          cmap=pl.cm.afmhot, 
                          extent=[-time_window[0], time_window[1], 0, time_arr.shape[0]])
                ax.set_aspect('auto')
                pl.title("Animal #: "+animal_id+'   Date: '+date)
                pl.ylabel('Bout Number')
                ax.axvline(0,color='white',linewidth=2,linestyle="--")
                #ax.axvline(np.abs(time_window[0])*time_arr.shape[1]/(time_window[1]-time_window[0]),color='white',linewidth=2,linestyle="--")

                ax = fig.add_subplot(2,1,2)
                FA.plot_perievent_hist(event_times, time_window, out_path=None, plotit=True, subplot=ax )
                pl.ylim([0,df_max])

                if options.output_path is not None:
                    import os
                    outdir = os.path.join(options.output_path, options.exp_type)
                    if not os.path.isdir(outdir):
                        os.makedirs(outdir)
                    pl.savefig(outdir+'/'+animal_id+'_'+date+options.plot_format)
                    print outdir+'/'+animal_id+'_'+date+options.plot_format
                else:
                    pl.show()



    # for animal_id in all_data.keys():
    #     # load data from hdf5 file by animal-date-exp_type
    #     animal = all_data[animal_id]
    #     for date in animal.keys():
    #         if options.exp_date is None or options.exp_date == date:

    #             fig = pl.figure()
    #             ax = fig.add_subplot(2,1,1)
                
    #             FA = FiberAnalyze(options)
    #             [FA, success] = loadFiberAnalyze(FA,
    #                                              options, 
    #                                              animal_id, 
    #                                              date, 
    #                                              exp_type)
    #             [df_max, df_min] = FA.get_plot_ylim(exp_type, 
    #                                                 FA.fluor_normalization,
    #                                                 ymax_setting)

    #             if exp_type in animal[date].keys():
    #                 print "trigger_data", FA.trigger_data

    #                 if(success!=-1):
    #                     event_times = FA.get_event_times(edge=event_edge, 
    #                                                      nseconds=float(options.event_spacing), 
    #                                                      exp_type=exp_type)
    #                     print "len(event_times)", len(event_times)
    #                     if len(event_times) > 0:
    #                         if max_num_epochs > 0:
    #                             event_times = event_times[0:max_num_epochs]

    #                         print "baseline_window", baseline_window
    #                         time_arr = np.asarray( FA.get_time_chunks_around_events(
    #                                                     FA.fluor_data, 
    #                                                     event_times, 
    #                                                     time_window, 
    #                                                     baseline_window=baseline_window) )

    #                         # Generate a heatmap of activity by bout, with range set between the 5% quantile of
    #                         # the data and the 'df_max' argument of the function
    #                         from scipy.stats.mstats import mquantiles
    #                         baseline = mquantiles( time_arr.flatten(), prob=[0.05])
    #                         ax.imshow(time_arr, 
    #                                   interpolation="nearest",
    #                                   vmin=baseline,
    #                                   vmax=df_max,
    #                                   cmap=pl.cm.afmhot, 
    #                                   extent=[-time_window[0], time_window[1], 0, time_arr.shape[0]])
    #                         ax.set_aspect('auto')
    #                         pl.title("Animal #: "+animal_id+'   Date: '+date)
    #                         pl.ylabel('Bout Number')
    #                         ax.axvline(0,color='white',linewidth=2,linestyle="--")
    #                         #ax.axvline(np.abs(time_window[0])*time_arr.shape[1]/(time_window[1]-time_window[0]),color='white',linewidth=2,linestyle="--")

    #                         ax = fig.add_subplot(2,1,2)
    #                         FA.plot_perievent_hist(event_times, time_window, out_path=None, plotit=True, subplot=ax )
    #                         pl.ylim([0,df_max])

    #                         if options.output_path is not None:
    #                             import os
    #                             outdir = os.path.join(options.output_path, options.exp_type)
    #                             if not os.path.isdir(outdir):
    #                                 os.makedirs(outdir)
    #                             pl.savefig(outdir+'/'+animal_id+'_'+date+options.plot_format)
    #                             print outdir+'/'+animal_id+'_'+date+options.plot_format
    #                         else:
    #                             pl.show()

#----------------------------------------------------------------------------------------

def group_bout_ci(all_data, options, 
                  df_max=0.35, event_edge="rising",num_bouts=5, baseline_window='full'):
    """
    Save out plots of mean or median activity with confidence intervals. 
    """
    exp_type = options.exp_type
    time_window = np.asarray(options.time_window.split(':'),dtype=np.int)

    # Create figure
    fig = pl.figure()
    ax = fig.add_subplot(1,1,1)

    i=0 # color counter
    exp_types = ['homecagesocial', 'homecagenovel']
    exp_arrays = []
    for exp_type in exp_types:
        median_time_series = []
        for animal_id in all_data.keys():
            # load data from hdf5 file by animal-date-exp_type
            animal = all_data[animal_id]
            if animal.attrs['mouse_type'] == options.mouse_type:
                for dates in animal.keys():

                    date = animal[dates]               
                    FA, success = loadFiberAnalyze(options, animal_id, dates, exp_type)

                    if exp_type in animal[dates].keys():
    #                    if(FA.load(file_type="hdf5") != -1):
                        if(success != -1):
                            event_times = FA.get_event_times(event_edge, int(options.event_spacing))
                            print "len(event_times)", len(event_times)
                            time_arr = np.asarray( FA.get_time_chunks_around_events(FA.fluor_data, 
                                                                                    event_times, 
                                                                                    time_window,
                                                                                    baseline_window=baseline_window) )

                            if num_bouts is not None:
                                time_arr = time_arr[:num_bouts,:]
                            
                            median_time_series.append( np.median(time_arr, axis=0) )

        # XXX HACK: the time series aren't quite the same length so we truncate them 
        # to the same length in order to stack them
        truncate = 1e10
        for i in xrange(len(median_time_series)):
            truncate = min( truncate, len(median_time_series[i]) )

        exp_array = np.zeros((len(median_time_series), truncate))
        for i in xrange(len(median_time_series)):
            exp_array[i,:] = median_time_series[i][:truncate]

        exp_arrays.append(exp_array)

    fmt = 'r-'
    for a in exp_arrays:
        e = np.std(a,axis=0)
#        for i in xrange(a.shape[0]):
#            ax.plot(a[i,:],fmt)
#        ax.plot(np.median(a,axis=0))
        ax.errorbar(range(a.shape[1]), np.mean(a,axis=0),yerr=e, fmt=fmt)
        ax.set_aspect('auto')
        fmt = 'b-'
    pl.show()

        # THIS CODE WAS INDENTED INCORRECTLY, I WASN'T SURE HOW TO FIX IT
        #         FA = FiberAnalyze(options)
        #         [FA, success] = loadFiberAnalyze(FA,
        #                                          options, 
        #                                          animal_id, 
        #                                          dates, 
        #                                          exp_type)

        #         median_time_series = []
        #         if exp_type in animal[dates].keys():
        # #                    if(FA.load(file_type="hdf5") != -1):
        #             if(success != -1):
        #                 event_times = FA.get_event_times(edge=event_edge, 
        #                                                  #nseconds=int(options.event_spacing), 
        #                                                  exp_type=exp_type)
        #                 print "len(event_times)", len(event_times)
        #                 time_arr = np.asarray( FA.get_time_chunks_around_events(FA.fluor_data, 
        #                                                                         event_times, 
        #                                                                         time_window) )

        #                 # Generate a heatmap of activity by bout, with range set 
        #                 # between the 5% quantile of the data and the 'df_max' argument 
        #                 # of the function
        #                 median_time_series.append( np.median(time_arr, axis=0) )


        #     ax.plot(median_time_series)
        #     ax.set_aspect('auto')
        #     pl.title("Animal #: "+animal_id+'   Date: '+dates)
        #     pl.ylabel('Bout Number')
        #     ax.axvline(0,color='white',linewidth=2,linestyle="--")

        #     if options.output_path is not None:
        #         import os
        #         outdir = options.output_path
        #         if not os.path.isdir(outdir):
        #             os.makedirs(outdir)
        #         pl.savefig(outdir+'/'+animal_id+'_'+dates+options.plot_format)
        #         print outdir+'/'+animal_id+'_'+dates+options.plot_format
        #     else:
        #         pl.show()

#----------------------------------------------------------------------------------------

def group_plot_time_series(all_data, options):
    """
    Save out time series for each trial, overlaid with 
    red lines indicating event epochs
    """
    # load data from hdf5 file by animal-date-exp_type

    [iter_list, animal_id_list, date_list, exp_type_list] = group_iter_list(all_data, options)

    for exp in iter_list:
        animal_id = exp['animal_id']
        date = exp['date']
        exp_type = exp['exp_type']

        FA = FiberAnalyze(options)
        FA.fluor_normalization = 'deltaF'
        FA.time_range = '0:-1'
        [FA, success] = loadFiberAnalyze(FA, options, animal_id, date, exp_type)

        if (success != -1):
            dir = options.output_path + '/' + FA.exp_type
            print dir
            if os.path.isdir(dir) is False:
                os.makedirs(dir)
            FA.plot_basic_tseries(out_path = dir + '/' + FA.subject_id + "_" + 
                                  FA.exp_date + "_" + FA.exp_type + "_" + 
                                  str(int(FA.time_range.split(':')[0])) +  "_" + 
                                  str(int(FA.time_range.split(':')[1])) +"_" ) 

#----------------------------------------------------------------------------------------

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

            FA = FiberAnalyze(options)
            FA.smoothness = int(smoothness)
            FA.time_range = str(start) + ':' + str(end)
            FA.fluor_normalization = 'deltaF'
            [FA, success] = loadFiberAnalyze(FA, options, animal_id, date, exp_type)


           # print "Test Keys: ", all_data[str(421)][str(20121008)][FA.exp_type].keys()

            print ""
            print "\t--> Plotting: ", FA.subject_id, FA.exp_date, FA.exp_type, 
            print "\t              ", FA.smoothness, FA.time_range
           # if(FA.load(file_type="hdf5") != -1):
            if (success != -1):
                dir = options.output_path + '/' + FA.exp_type
                if os.path.isdir(dir) is False:
                    os.makedirs(dir)
                FA.plot_basic_tseries(out_path = dir + '/' + FA.subject_id + "_" +
                                      FA.exp_date + "_" + FA.exp_type + "_" + 
                                      str(int(FA.time_range.split(':')[0])) +  
                                      "_" + str(int(FA.time_range.split(':')[1])) +"_",
                                      )
                                      #resolution=1000 ) 

#----------------------------------------------------------------------------------------

def get_novel_social_pairs(all_data, options, exp1, exp2):
    """
    all_data = an hdf5 file containing all of the time series data
    exp1 and exp2 = the two experiment types to be compared 
    (i.e. homecagesocial and homecagenovel)

    Returns: a dict where the keys are animal_ids
    and each entry is a dict: 
    {keys = exp_type, entry = date of the most recent trial of that exp_type} 
    """
    pairs = dict()

    mouse_type = options.mouse_type

    # load data from hdf5 file by animal-date-exp_type.
    for animal_id in all_data.keys():
        animal = all_data[animal_id]

        if animal.attrs['mouse_type'] == mouse_type: #don't use EYFP or GC3
            pairs[animal_id] = dict()

            # make a list for each experiment type of all dates on which that exp was run.
            for date in animal.keys(): 
                for exp_type in animal[date].keys(): 
                    if exp_type == str(exp1) or exp_type == str(exp2):
                        if exp_type in pairs[animal_id].keys():
                            pairs[animal_id][exp_type].append(int(date))
                        else:
                            pairs[animal_id][exp_type] = [int(date)]
                    
    ##As a heuristic to choose between multiple trials of the same exp_type,
    ##use the one with the latest (largest) date                
    for animal_id in pairs.keys():
        if pairs[animal_id].keys() == []:
            del pairs[animal_id]
        else:
            for exp_type in pairs[animal_id].keys():
                pairs[animal_id][exp_type] = np.max(pairs[animal_id][exp_type]) #choose the most recent experiment date

    print "Pairs after max filter: ", pairs

    return pairs


def compileAnimalScoreDictIntoArray(pair_avg_scores):
    """
    Create two matched arrays (i.e. one for novel, one for social) with the 
    avg score for each animal

    input: a dict (keys: animal_id, entries: dict (keys: exp_type, entries: 
    average score across all epochs))

    output: a dict (key: exp_type, entry: array of avg score for each animal, 
    [avg_score_for_animal_1, avg_score_for_animal_2,...])
    """
    exp_scores = dict() #key: exp_type, entry: array of avg score for each animal
    animal_list = []
    for animal_id in pair_avg_scores.keys():
        animal_list.append(animal_id)
        for exp_type in pair_avg_scores[animal_id].keys():
            print animal_id, exp_type
            if exp_type in exp_scores.keys():
                exp_scores[exp_type].append(pair_avg_scores[animal_id][exp_type])
            else:
                exp_scores[exp_type] = [pair_avg_scores[animal_id][exp_type]]

    return [exp_scores, animal_list]

def plot_compare_start_and_end(options,
                               exp_scores, 
                               exp1, 
                               exp2, 
                               compare_before_after_end, 
                               time_window, 
                               metric, 
                               output_path,
                               pvalue = None,
                               ):
    
    plt.figure()
    p1, = plt.plot(exp_scores[exp1], 'o')
    p2, = plt.plot(exp_scores[exp2], 'o')
    plt.legend([p1, p2], [exp1, exp2])
    plt.xlabel('Mouse (one mouse per column)')
    if compare_before_after_end:
        plt.ylabel( 'After End - Before End (' + metric +' w/in ' + 
                     str(time_window[1]) + 's window, avged across epochs)')
    else:
        plt.ylabel( 'End - Start (' + metric +' w/in ' + str(time_window[1]) + 
                    's window, avged across epochs)')

    if pvalue is not None:
        plt.title('Comparison of peak fluorescence after start and end of interaction, p<' + "{0:.4f}".format(pvalue))
    else:
        plt.title('Comparison of peak fluorescence after start and end of interaction')
    if compare_before_after_end:
        pl.savefig(output_path + '/' + options.mouse_type + '_'+ str(time_window[1]) + '_' + metric+ 
                    '_after_minus_before_end'+options.plot_format)
    else:
        pl.savefig(output_path + '/' + options.mouse_type + '_' + str(time_window[1]) + '_' + metric+ 
                    '_end_minus_start'+options.plot_format)


def compare_start_and_end_of_epoch(all_data, options, 
                                   exp1='homecagesocial', 
                                   exp2='homecagenovel', 
                                   time_window=[0,0.25], 
                                   metric='average', 
                                   test='ttest', 
                                   plot_perievent=False, 
                                   compare_before_after_end=False,
                                   show_plot=True,
                                   perivent_window = [3, 3],
                                   max_bout_number=0, 
                                    ):

    """
    Calculates the difference between the fluorescence in a window at the beginning
    and the end of each epoch.
    Compares these differences between novel object and social behaviors.
    Plots the average difference vs. epoch number for novel and social (ideally on the same plot)
    Returns the t-test score comparing novel and social.

    Metric can be 'average' (mean value of curve) or 'peak' (peak fluorescence value).
    """

    pairs = get_novel_social_pairs(all_data, options, exp1, exp2) 
                ##can use any function here that returns pairs of data

    pair_scores = dict() #key: animal_id, entry: a dict storing an array of scores for each trial type (i.e. homecagenovel and homecagesocial)
    pair_avg_scores = dict() #key: animal_id, entry: a dict storing the average score within a trial for each trial type (i.e. homecagenovel and homecagesocial)

    for animal_id in pairs.keys():
        pair_scores[animal_id] = dict()
        pair_avg_scores[animal_id] = dict()
        for exp_type in pairs[animal_id].keys():
            FA = FiberAnalyze(options)
            [FA, success] = loadFiberAnalyze(FA, options, animal_id, pairs[animal_id][exp_type], exp_type)
            if(success != -1):

                start_event_times = FA.get_event_times(edge="rising", 
                                                       #nseconds=float(options.event_spacing), 
                                                       exp_type=exp_type)
                end_event_times = FA.get_event_times(edge="falling", 
                                                     #nseconds=float(options.event_spacing), 
                                                     exp_type=exp_type)

                #--Get an array of time series chunks in a window around each event time
                reverse_window = [time_window[1], time_window[0]]
                start_time_arr = np.asarray( FA.get_time_chunks_around_events(FA.fluor_data, 
                                                                              start_event_times, 
                                                                              time_window, 
                                                                              baseline_window=-1 ) )
                before_time_arr = np.asarray( FA.get_time_chunks_around_events(FA.fluor_data, 
                                                                               end_event_times, 
                                                                               reverse_window, 
                                                                               baseline_window=-1 ))
                end_time_arr = np.asarray( FA.get_time_chunks_around_events(FA.fluor_data, 
                                                                            end_event_times, 
                                                                            time_window, 
                                                                            baseline_window=-1 ))


                start_scores = np.array(FA.score_of_chunks(start_time_arr, metric))
                before_scores = np.array(FA.score_of_chunks(before_time_arr, metric))
                end_scores = np.array(FA.score_of_chunks(end_time_arr, metric))
                scores_diff = end_scores - start_scores
                if compare_before_after_end:
                    scores_diff = end_scores - before_scores

                pair_scores[animal_id][exp_type] = scores_diff
                #pair_avg_scores[animal_id][exp_type] = np.mean(scores_diff)

                if max_bout_number>0:
                    pair_avg_scores[animal_id][exp_type] = np.mean(scores_diff[0:max_bout_number])
                else:
                    pair_avg_scores[animal_id][exp_type] = np.mean(scores_diff)

                if(plot_perievent):
                    fig = plt.figure()
                    ax = fig.add_subplot(2,1,1)
                    title = FA.subject_id + ' ' + FA.exp_date + ' ' + FA.exp_type
                    ax.set_title(title)
                    FA.plot_perievent_hist(start_event_times, perivent_window, 
                                           out_path=options.output_path, plotit=True, 
                                           subplot=ax, baseline_window=-1  )
                    plt.title('Top: aligned with start; Bottom: aligned with end')


                    ax = fig.add_subplot(2,1,2)
                    FA.plot_perievent_hist(end_event_times, perivent_window, 
                                           out_path=options.output_path, plotit=True, 
                                           subplot=ax, baseline_window=-1 )
                    if options.output_path is None:
                        pl.show()
                    else:
                        print "Saving peri-event time series..."
                        pl.savefig(options.output_path + FA.subject_id + '_' + FA.exp_date + 
                                   '_' + FA.exp_type + "_perievent_tseries"+options.plot_format)
                        pl.savefig(options.output_path + FA.subject_id + '_' + FA.exp_date + 
                                   '_' + FA.exp_type + "_perievent_tseries"+options.plot_format)


    [exp_scores, animal_list] = compileAnimalScoreDictIntoArray(pair_avg_scores)
    [zstatistic, pvalue] = statisticalTestOfComparison(exp_scores, exp1, exp2, test)
    print "Exp_scores", exp_scores

    plt.close('all')
    plot_compare_start_and_end(options=options,
                               exp_scores=exp_scores, 
                               exp1=exp1, 
                               exp2=exp2, 
                               compare_before_after_end=compare_before_after_end, 
                               time_window=time_window, 
                               metric=metric, 
                               output_path=options.output_path,
                               pvalue = pvalue,
                                )

    #Write out actual values to a text file to be used in making bar plot and
    #calculating significance of difference
    filename = options.output_path + '/' + options.mouse_type + '_' + 'window_' + \
            str(time_window[1]).replace(".","-") + '_numbouts_' + \
            str(max_bout_number) + '_minspace_' + \
            str(options.event_spacing).replace(".", "-") + '_' + metric
    f = open(filename+'.txt', 'w')
    f.write(str(exp1) + '\n')
    f.write(str(exp_scores[exp1]) + '\n')
    f.write(str(exp2) + '\n')
    f.write(str(exp_scores[exp2]) + '\n')
    f.write("animal ordering: " + str(animal_list))
    f.close()

    if show_plot:
        plt.show()



def statisticalTestOfComparison(exp_scores, exp1, exp2, test):
    """
    perform a statistical test to determine the significance of the difference
    between two arrays representing the score of each animal's trial under
    two different behavioral experiment conditions
    """

    if test == "ttest":
            [tvalue, pvalue] = stats.ttest_rel(exp_scores[exp1], exp_scores[exp2])
            print "tvalue: ", tvalue, " pvalue: ", pvalue
            return [tvalue, pvalue]
    if test == "wilcoxon":
            [zstatistic, pvalue] = stats.wilcoxon(exp_scores[exp1], exp_scores[exp2])
            print "zstatistic: ", zstatistic, " pvalue: ", pvalue
            return [zstatistic, pvalue]


def plotEpochComparison(options,
                        pair_scores, 
                        pair_avg_scores, 
                        exp_scores, 
                        exp1, 
                        exp2, 
                        time_window, 
                        metric, 
                        max_bout_number, 
                        pvalue, 
                        min_spacing,
                        animal_list,
                        ):

    #    colors = ['g', 'b']
    colors = ['#96C681', '#3d7a23'] #dark green and light green

    plt.close('all')
    plt.figure()
    p1, = plt.plot(exp_scores[exp1], 'o', markerfacecolor=colors[0])
    p2, = plt.plot(exp_scores[exp2], 'o', markerfacecolor=colors[1])
    plt.legend([p1, p2], [exp1, exp2])
    plt.xlabel('Mouse (one mouse per column)')
    if time_window[0] == 0 and time_window[1] == 0:
        plt.ylabel( metric + ' w/in entire epoch, avged across epochs')
    else:
        plt.ylabel( metric + ' w/in ' + 
                    str(time_window[1]) + 's window, avged across epochs')
    plt.title('Comparison of ' + metric + 
              ' between ' + exp1 + ' and ' + 
              exp2 + '. p < ' + str(pvalue)) 

    print "window_" + str(time_window[1]) + "_numbouts_" + str(max_bout_number) + "_minspace_" + str(min_spacing) + "_" + metric
    print exp1, ": ", exp_scores[exp1]
    print exp2, ": ", exp_scores[exp2]

    outdir = options.output_path
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    #Write out actual values to a text file to be used in making bar plot and
    #calculating significance of difference
    filename = options.output_path + '/'+options.mouse_type+'_window_' + \
                str(time_window[1]) + '_numbouts_' + \
                str(max_bout_number) + '_minspace_' + \
                str(min_spacing).replace(".", "-") + '_' + metric
    f = open(filename+'.txt', 'w')
    f.write(str(exp1) + '\n')
    f.write(str(exp_scores[exp1]) + '\n')
    f.write(str(exp2) + '\n')
    f.write(str(exp_scores[exp2]) + '\n')
    f.write("animal ordering: " + str(animal_list))
    f.close()

    plt.savefig(filename+ options.plot_format)


def compare_epochs(all_data, 
                   options, 
                   exp1, 
                   exp2, 
                   time_window, 
                   metric, 
                   max_bout_number, 
                   test, 
                   make_plot=True, 
                   plot_perievent=False,
                   show_plot=True,
                   ):
    """
    Compares the fluorescence during epochs for each mouse undergoing 
    two behavioral experiments (exp1 and exp2). Fluorescence can be 
    quantified (or, scored) using metrics such as 
    'peak' (maximum fluorescence value during epoch),
    'average'  (average fluorescence during epoch),
    'spacing', (time from end of current epoch to beginning of the next)

    test represents the statistical test used for comparsion:
     can be 'wilcoxon' or 'ttest'

    max_bout_number is the number of bouts to include in calculating
    the score for a trial. Set to 0 to include all bouts of a trial. 

    Time window is an array of two values [time before onset, time after onset]
    In order to use the full length of each epoch as opposed to a fixed window,
    set time_window = [0, 0].

    Plots the average score for each mouse under each 
    behavioral condition. Using a statistical test, determines 
    whether there is a significant difference in the average 
    score between behavioral conditions across all mice.



    Returns: 1) a dict (keys: animal_id, 
                        entries: dict (keys: exp_type, 
                                       entries: array with score for each epoch in trial))
            2) a dict (keys: animal_id, 
                       entries: dict (keys: exp_type, 
                                      entries: average score across all epochs))
    """

    pairs = get_novel_social_pairs(all_data, 
                                   options,
                                   exp1, 
                                   exp2) #can use any function here that returns pairs of data

    pair_scores = dict() #key: animal_id, 
                            #entry: a dict storing an array of scores for 
                            #       each trial type (i.e. homecagenovel and homecagesocial)
    pair_avg_scores = dict() #key: animal_id, 
                                #entry: a dict storing the average score within a trial 
                                #       for each trial type (i.e. homecagenovel and homecagesocial)

    for animal_id in pairs.keys():
        pair_scores[animal_id] = dict() #initialize entry for each animal_id
        pair_avg_scores[animal_id] = dict()

        for exp_type in pairs[animal_id].keys():
            date = pairs[animal_id][exp_type]

            FA = FiberAnalyze(options)
            [FA, success] = loadFiberAnalyze(FA,
                                             options, 
                                             animal_id, 
                                             date, 
                                             exp_type)

            if (success != -1):
                start_event_times = FA.get_event_times(edge="rising", 
                                                       exp_type=exp_type)

                end_event_times = FA.get_event_times(edge="falling", 
                                                     exp_type=exp_type)

                #--Get an array of time series chunks in a window around each event time
                if time_window[0] == 0 and time_window[1] == 0:
                    start_time_arr = np.asarray( 
                                        FA.get_time_chunks_around_events(
                                            FA.fluor_data, 
                                            event_times = start_event_times, 
                                            window = time_window, 
                                            baseline_window=-1, 
                                            end_event_times = end_event_times))
                else:
                    start_time_arr = np.asarray( 
                                        FA.get_time_chunks_around_events(
                                            FA.fluor_data,                              
                                            event_times = start_event_times, 
                                            window = time_window, 
                                            baseline_window=-1))

                scores = np.array(FA.score_of_chunks(start_time_arr, metric, 
                                                    start_event_times, end_event_times))
                pair_scores[animal_id][exp_type] = scores
                if max_bout_number>0:
                    pair_avg_scores[animal_id][exp_type] = np.mean(scores[0:max_bout_number])
                else:
                    pair_avg_scores[animal_id][exp_type] = np.mean(scores)

                if(plot_perievent):
                    fig = plt.figure()
                    ax = fig.add_subplot(1,1,1)
                    title = FA.subject_id + ' ' + FA.exp_date + ' ' + FA.exp_type
                    ax.set_title(title)
                    FA.plot_perievent_hist(start_event_times, 
                                          [0, 10], 
                                          out_path=options.output_path, 
                                          plotit=True, 
                                          subplot=ax, 
                                          baseline_window=-1)
                    if options.output_path is None:
                        pl.show()
                    else:
                        print "Saving peri-event time series..."
                        pl.savefig(options.output_path + 
                                   FA.subject_id + '_' + 
                                   FA.exp_date + '_' + 
                                   FA.exp_type + 
                                   "_perievent_tseries"+options.plot_format)    

    [exp_scores, animal_list] = compileAnimalScoreDictIntoArray(pair_avg_scores)

    print "Exp_scores ", exp_scores
    [score, pvalue] = statisticalTestOfComparison(exp_scores, 
                                                  exp1, 
                                                  exp2, 
                                                  test)

    print 'time_window ', time_window
    plotEpochComparison(options,
                        pair_scores, 
                        pair_avg_scores, 
                        exp_scores, 
                        exp1, 
                        exp2, 
                        time_window, 
                        metric, 
                        max_bout_number, 
                        pvalue, 
                        options.event_spacing,
                        animal_list)

    return [pair_scores, pair_avg_scores]


def get_bout_averages(pair_scores):
    """
    input: a dict (keys: animal_id, entries: 
                            dict (keys: exp_type, 
                                  entries: array with score for each epoch in trial))

    returns: 1) bout_dict = a dict (keys: exp_type, 
                                    entries: a dict (keys: bout number,
                                                     entries: array with score from 
                                                              each mice for that bout number)
             2) bout_avg_dict = a dict (keys: exp_type, 
                                        entries: an array with the average score 
                                                 for each bout number)
             3) bout_count_dict = a dict (keys: exp_type, 
                                          entries: an array with the number of trials 
                                                   that had at least as many bouts
                                                   as the index of the array)

            4) bout_std_err = a dict (keys: exp_type, 
                                          entries: an array with the standard
                                                   error for each bout number)
    """

    bout_dict = dict()
    bout_avg_dict = dict()
    bout_count_dict = dict()
    bout_std_err = dict()

    for animal_id in pair_scores.keys():
        for exp_type in pair_scores[animal_id].keys():
            if exp_type not in bout_dict.keys(): #initialize an exp_type key
                bout_dict[exp_type] = dict()

            scores = pair_scores[animal_id][exp_type] 
            for i in range(len(scores)):
                score = scores[i]
                if i not in bout_dict[exp_type].keys():
                    bout_dict[exp_type][i] = [score]
                else:
                    bout_dict[exp_type][i].append(score)

    for exp_type in bout_dict.keys():
        bout_avg_dict[exp_type] = []
        bout_count_dict[exp_type] = []
        bout_std_err[exp_type] = []

        for i in bout_dict[exp_type].keys():
            bout_avg_dict[exp_type].append(np.mean(bout_dict[exp_type][i])) #mean of scores of bout number i
            bout_count_dict[exp_type].append(len(bout_dict[exp_type][i]))
            std_err = stats.sem(bout_dict[exp_type][i])
            bout_std_err[exp_type].append(std_err)

    return [bout_dict, bout_avg_dict, bout_count_dict, bout_std_err]


########################################################################

def eNegX(p, x):
        x0, y0, c, k=p
        #Set c=1 to normalize all of the trials, since we
        # are only interested in the rate of decay
        y = (1 * np.exp(-k*(x-x0))) + y0
        return y

def eNegX_residuals(p, x, y):
    return y - eNegX(p, x)

def fit_exponential(x, y, num_points=100):
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
                eNegX_residuals, p_guess, args=(x, y), full_output=1)

            x0,y0,c,k=p 

            numPoints = np.floor((np.max(x) - np.min(x))*num_points)
            xp = np.linspace(np.min(x), np.max(x), numPoints)
            pxp = eNegX(p, xp)
            yxp = eNegX(p, x)

            sstot = np.sum(np.multiply(y - np.mean(y), y - np.mean(y)))
            sserr = np.sum(np.multiply(y - yxp, y - yxp))
            r2 = 1 - sserr/sstot
            if max_r2 == -1:
                maxvalues = (xp, pxp, x0, y0, c, k, r2, yxp)
            if r2 > max_r2:
                max_r2 = r2
                maxvalues = (xp, pxp, x0, y0, c, k, r2, yxp)

    print "maxvalues", maxvalues

    return maxvalues

def plot_decay(options,
               bout_dict, 
               bout_avg_dict, 
               bout_count_dict, 
               bout_std_err,
               time_window=[0,1], 
               metric='peak', 
               output_path=None, 
               max_bout_number=0, 
               min_spacing=0,
               show_plot=False,
               ):

    """
    Plot the change in fluorescence vs. bout number 
    using data in the format of output from get_bout_averages()],
    
    Plots:
    1) The average bout score across mice, comparing experiment types.
    2) Number of mice that had at least n bouts, for each 
       n < max_bout_number, again comparing experiment types.
    3) The score of every bout of every mouse (as opposed to the average
        of mice within an experiment type)
    """


    if max_bout_number == 0:
        max_bout_number = len(bout_avg_dict[bout_avg_dict.keys()[0]])

#    colors = ['g', 'b']
    colors = ['#96C681', '#3d7a23'] #dark green and light green

    fig = plt.figure()  
    ax = fig.add_subplot(111)
    x = np.array(range(max_bout_number))
    y0 = bout_avg_dict[bout_avg_dict.keys()[0]][0:max_bout_number]
    y1 = bout_avg_dict[bout_avg_dict.keys()[1]][0:max_bout_number]


    yvalues = y0
    try:
        xp, pxp, xo, yo, c, k, r2, yxp = fit_exponential(x, np.array(yvalues)+1.0)
        ax.plot(xp, pxp-1, color=colors[0])
        legend0 = bout_avg_dict.keys()[0] + ": decay rate = " + "{0:.2f}".format(k) + ", r^2 = " + "{0:.2f}".format(r2)
    except:
        legend0 = bout_avg_dict.keys()[0]
        print "Exponential Curve fit did not work"

    plot0 = ax.errorbar(x, y0, 
                        yerr=1.96*np.array(bout_std_err[bout_avg_dict.keys()[0]][0:max_bout_number]), 
                        fmt='o', color=colors[0])

    yvalues = y1
    try:
        xp, pxp, xo, yo, c, k, r2, yxp = fit_exponential(x, np.array(yvalues)+1.0)
        ax.plot(xp, pxp-1, color=colors[1])
        legend1 = bout_avg_dict.keys()[1] + ": decay rate = " + "{0:.2f}".format(k) + ", r^2 = " + "{0:.2f}".format(r2)
    except:
        legend1 = bout_avg_dict.keys()[1]
        print "Exponential Curve fit did not work"

    plot1 = ax.errorbar(x, y1, 
                        yerr=1.96*np.array(bout_std_err[bout_avg_dict.keys()[1]][0:max_bout_number]), 
                        fmt='o', color=colors[1])




    plt.legend([plot0, plot1], [legend0, legend1])
    plt.xlabel('Bout number')
    plt.title('Average decay over time')
    plt.ylabel('Average ' + metric + ' per bout [dF/F]')
    if metric == 'spacing':
        plt.title('Average time interval between bouts')
        plt.ylabel('Average time from end of bout to beginning of next bout [s]')

    plt.savefig(options.output_path + '/'+ options.mouse_type + '_'+ 'decay_window_' + 
                str(time_window[1]) + '_minspace_' + str(min_spacing) + '_mousetype_' + 
                options.mouse_type + '_' + metric+ options.plot_format)

    plt.figure()
    plot0, = plt.plot(bout_count_dict[bout_count_dict.keys()[0]][0:max_bout_number],  
                        color=colors[0])
    plot1, = plt.plot(bout_count_dict[bout_count_dict.keys()[1]][0:max_bout_number],  
                        color=colors[1])
    plt.legend([plot0, plot1], bout_count_dict.keys())
    plt.title('Counts of bout number')

    plt.figure()
    exps = bout_dict.keys()
    plots = [0, 0]
    for i in range(len(exps)):
        exp = bout_dict[exps[i]]
        for j in exp:
            plots[i], = plt.plot(j*np.ones(len(exp[j])), exp[j], 'o', color=colors[i])
    plt.title('Individual decays over time')
    plt.legend(plots, bout_count_dict.keys())

    if show_plot:
        plt.show()


def compare_decay(all_data, 
                  options, 
                  exp1, 
                  exp2, 
                  time_window, 
                  metric, 
                  test, 
                  max_bout_number,
                  make_plot=True, 
                  just_first=False, 
                  show_plot=False):
    """
    Using 'metric' to score the fluorescent response in each bout,
    plot the decay in the response vs. bout number, and fits
    with an exponential curve.

    Metric can be:
    'peak' (maximum fluorescence value during epoch),
    'average'  (average fluorescence during epoch),
    'spacing', (time from end of current epoch to beginning of the next)

    test represents the statistical test used for comparsion:
    can be 'wilcoxon' or 'ttest'

    max_bout_number is the number of bouts to include in calculating
    the score for a trial. Set to 0 to include all bouts of a trial. 

    Time window is an array of two values [time before onset, time after onset]
    In order to use the full length of each epoch as opposed to a fixed window,
    set time_window = [0, 0].

    """

    [pair_scores, pair_avg_scores] =  compare_epochs(all_data, options, 
                                                     exp1=exp1, 
                                                     exp2=exp2, 
                                                     time_window=time_window, 
                                                     metric=metric, 
                                                     test=test, 
                                                     make_plot=False, 
                                                     max_bout_number = max_bout_number)   

    [bout_dict, bout_avg_dict, bout_count_dict, bout_std_err] = get_bout_averages(pair_scores)

    plot_decay(options=options,
               bout_dict=bout_dict, 
               bout_avg_dict=bout_avg_dict, 
               bout_count_dict=bout_count_dict, 
               bout_std_err=bout_std_err,
               time_window=time_window, 
               metric=metric, 
               output_path=options.output_path, 
               max_bout_number=max_bout_number,
               min_spacing=options.event_spacing,
               show_plot=show_plot,
               )


def event_length_histogram(all_data, 
                           options, 
                           max_bout_number,
                           make_plot=True,
                           ):

    """
    Plots a histogram of the length (in seconds) of each event
    """

    [iter_list, animal_id_list, date_list, exp_type_list] = group_iter_list(all_data, options)

    event_lengths = np.zeros(0)

    for exp in iter_list:
        animal_id = exp['animal_id']
        date = exp['date']
        exp_type = exp['exp_type']

        FA = FiberAnalyze(options)
        FA.fluor_normalization = 'deltaF'
        FA.time_range = '0:-1'
        [FA, success] = loadFiberAnalyze(FA, options, animal_id, date, exp_type)

        if (success != -1):

            start_event_times = np.array(FA.get_event_times(edge="rising", 
                                                   #nseconds=float(options.event_spacing), 
                                                   exp_type=exp_type))
            end_event_times = np.array(FA.get_event_times(edge="falling", 
                                                 #nseconds=float(options.event_spacing), 
                                                 exp_type=exp_type))

            lengths = end_event_times - start_event_times
            event_lengths = np.append(event_lengths, lengths)

    print event_lengths
    x = np.arange(0.,15., 0.1)
    pl.figure()
    pl.hist(event_lengths, x)
    pl.title("Distribution of bout lengths: " + str(options.exp_type))
    pl.xlabel("Bout length [s]")
    pl.ylabel("Number of bouts")
    if options.output_path is not None:
        outdir = os.path.join(options.output_path, options.exp_type)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        pl.savefig(outdir+'/'+exp_type+'_event_length_hist' + options.plot_format)
        print outdir+'/'+exp_type+'_event_length_hist' + options.plot_format
    else:
        pl.show()



#-------------------------------------------------------------------------------------------

def set_and_read_options_parser():
   # Parse command line options
    parser = fra.add_command_line_options() #add all command line options from fiber_record_analyze

    parser.add_option("", "--time-window", dest="time_window",default='0:0',
                      help="Specify a time window for peri-event plots in format before:after.")

    parser.add_option("", "--exp-date", dest="exp_date", default=None,
                       help="Limit group analysis to trials of a specific date ")

    parser.add_option("", "--representative-time-series-specs-file", 
                      dest="representative_time_series_specs_file", 
                      default='representative_time_series_specs.txt',
                      help=("Specify file of representative trials to plot. File in format: "
                            "animal# date start_in_secs end_in_secs exp_type smoothness"))

    parser.add_option("", "--plot-format", dest="plot_format", default='.png',
                      help="Image format for saving plots: '.png', '.pdf', '.svg', '.tiff'")

    #Various plot types
    parser.add_option("", "--intensity-metric", dest="intensity_metric", default='average',
                       help=("Measure of the intensity of a spike in fluorescence during "
                             "a bout. Possible values are: "
                             "'peak', 'average', 'spacing', 'epoch_length', 'event_time', 'event_index'" ))

    parser.add_option("", "--group-regression-plot", dest="group_regression_plot", 
                      action="store_true", default=False, 
                      help=("Plot of linear regression to data from each individual"
                            "trial, with x-axis corresponding to the time until next bout, and y-axis "
                            " corresponding to the measure of spike intensity specified by the"
                            "--intensity-metric tag."))

    parser.add_option("", "--group-bout-heatmaps", dest="group_bout_heatmaps", 
                      action="store_true", default=False, 
                      help=("For each trial, plot perievent heatmaps and time series of individual"
                            " bouts, centered in time around the start of the bout, with the color"
                            " in the heatmap representing a fluorescence intensity."))

    parser.add_option("", "--group-bout-ci", dest="group_bout_ci", 
                      action="store_true", default=False, 
                      help=("For each experiment_type, plot the mean of all time series"
                        " chunks, centered in time around the start of the bout. The mean "
                        " is determined point by point, as the mean of all time chunks."))

    parser.add_option("", "--group-plot-time-series", dest="group_plot_time_series", 
                      action="store_true", default=False, 
                      help=("Save out time series for each trial, overlaid with "
                            " red lines indicating event epochs."))

    parser.add_option("", "--plot-representative-time-series", dest="plot_representative_time_series", 
                      action="store_true", default=False, 
                      help=("For each trial, plot perievent heatmaps and time series of individual"
                        " bouts, centered in time around the start of the bout, with the color"
                        " in the heatmap representing a fluorescence intensity."))

    parser.add_option("", "--compare-start-and-end-of-epoch", dest="compare_start_and_end_of_epoch", 
                      action="store_true", default=False, 
                      help=(" Calculates the difference between the fluorescence in a window at "
                            "the beginning and the end of each epoch. Compares these differences "
                            " between novel object and social behaviors."))

    parser.add_option("", "--compare-epochs", dest="compare_epochs", 
                      action="store_true", default=False, 
                      help=("Compares the fluorescence during epochs for each mouse undergoing "
                            "two behavioral experiments (i.e. homecagenovel and homecagesocial)."
                            "use --max-bout-number to specify how many bouts per trial to use "
                            " (this allows one to compensate for the difference in decays between"
                            " homecagenovel and homecagesocial, for example)."))

    parser.add_option("", "--max-bout-number", dest="max_bout_number", default=0,
                        help=(" The number of bouts to use per trial when comparing"
                             "two behavioral experiments. This allows one to compensate "
                            " for the difference in the number of bouts per trial during"
                            " homecagenovel and homecagesocial experiment types."))

    parser.add_option("", "--compare-decay", dest="compare_decay", 
                      action="store_true", default=False, 
                      help=("Compares the decay in fluorescence "
                            " with increasing bout number within "
                            " a trial for two exp_types (i.e. homecagesocial"
                            " and homecagenovel)."))

    parser.add_option("", "--event-length-histogram", dest="event_length_histogram", 
                      action="store_true", default=False, 
                      help=("Plot a histogram of the event length of all bouts "
                            "across all trial of a given exp_type."))

    parser.add_option("", "--print-spike-times", dest="print_spike_times", 
                      action="store_true", default=False, 
                      help=("Print a list of all peaks in fluorescence "
                            "during event times across all trials of all mice."))

    
    (options, args) = parser.parse_args()
    return (options, args)


if __name__ == "__main__":
    (options, args) = set_and_read_options_parser()
    all_data = h5py.File(options.input_path,'r')
    outdir = options.output_path
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # [before,after] event in seconds.
    time_window = np.array(options.time_window.split(':'), dtype='float32') 


    if options.group_regression_plot:
        group_regression_plot(all_data, options, 
                              exp_type=options.exp_type,
                              time_window=time_window,
                              metric=options.intensity_metric)

    elif options.group_bout_heatmaps:
        group_bout_heatmaps(all_data, options, 
                            exp_type=options.exp_type, time_window=time_window,
                            max_num_epochs=int(options.max_bout_number), ymax_setting = 'small')

    elif options.group_bout_ci:
        group_bout_ci(all_data, options, exp_type=options.exp_type,
                      time_window=time_window)

    elif options.group_plot_time_series:
        group_plot_time_series(all_data, options)

    elif options.plot_representative_time_series:
        plot_representative_time_series(options, options.representative_time_series_specs_file)

    elif options.compare_epochs:
        if options.exp_type == 'sucrose':
            compare_epochs(all_data, options, 
                           exp1='sucrose', exp2='sucrose', 
                           time_window=time_window, metric=options.intensity_metric,
                           test='wilcoxon', make_plot=True, max_bout_number=int(options.max_bout_number), 
                           plot_perievent=False)

        else:
            compare_epochs(all_data, options, 
                           exp1='homecagenovel', exp2='homecagesocial', 
                           time_window=time_window, metric=options.intensity_metric,
                           test='wilcoxon', make_plot=True, max_bout_number=int(options.max_bout_number), 
                           plot_perievent=False)

    elif options.compare_start_and_end_of_epoch:
        compare_start_and_end_of_epoch(all_data, options, 
                                       exp1='homecagesocial', exp2='homecagenovel', 
                                       time_window=[0, .5], metric=options.intensity_metric, test='wilcoxon',
                                       plot_perievent=False, max_bout_number=int(options.max_bout_number),
                                       show_plot=False)

    elif options.compare_decay:
        compare_decay(all_data, options, 
                      exp1='homecagenovel', exp2='homecagesocial', 
                      time_window=time_window, metric=options.intensity_metric, 
                      max_bout_number=int(options.max_bout_number), test='wilcoxon', 
                      make_plot=True, show_plot=False)

    elif options.event_length_histogram:
        event_length_histogram(all_data, options, max_bout_number=int(options.max_bout_number),
                               make_plot=True)

    elif options.print_spike_times:
        print "time_window", time_window
        print_spike_times(all_data, options, exp_type=options.exp_type,
                            max_num_epochs=int(options.max_bout_number), 
                            time_window=time_window, event_edge="rising")

    elif options.make_time_series_animations:
        make_time_series_animations(all_data, options, exp_type=options.exp_type,
                            max_num_epochs=int(options.max_bout_number), 
                            time_window=time_window, event_edge="rising")


# EOF
