from fiber_record_analyze import FiberAnalyze
from R_functions import pcor

from group_analysis import group_iter_list

def group_prediction(all_data, options, exp_type, time_window, metric,
                     num_animals=12.0,
                     num_bouts=10):
    """
    """

    # Create figure
    fig = pl.figure()
    ax = fig.add_subplot(1,1,1)

    next_event_measure = "onset"

    [iter_list, animal_id_list, date_list, exp_type_list] = group_iter_list(all_data, options)

    for exp in iter_list:
        animal_id = exp['animal_id']
        dates = exp['date']
        exp_type = exp['exp_type']
        print "animal_id", animal_id

        FA = FiberAnalyze(options)
        [FA, success] = loadFiberAnalyze(FA, options, animal_id, dates, exp_type)

        # get intensity and next_val values for this animal
        if success != -1:
            print "metric", metric
            print "time_window", time_window
            peak_intensity, onset_next_vals = FA.plot_next_event_vs_intensity(
                intensity_measure=metric, 
                next_event_measure=next_event_measure, 
                window=time_window, 
                out_path=None, 
                plotit=False,
                baseline_window=-1)

            event_time_intensity, _ = FA.plot_next_event_vs_intensity(
                intensity_measure='event_time', 
                next_event_measure=next_event_measure, 
                window=time_window, 
                out_path=None, 
                plotit=False,
                baseline_window=-1)

            # fit a robust regression
            if len(onset_next_vals) > 0:
                X = np.nan_to_num( np.vstack((np.log(peak_intensity), 
                                              np.log(event_time_intensity), 
                                              np.ones((len(onset_next_vals),)))))
                print "X", X.T
                if X.shape[1] > 2:
                
                    X = X[0:min(num_bouts,X.shape[1]),:]
                    y = np.log(onset_next_vals)




if __name__ == '__main__':
    all_data = h5py.File(options.input_path,'r')
