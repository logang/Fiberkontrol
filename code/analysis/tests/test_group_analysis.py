"""
test_group_analysis.py
This module runs unit tests on 
the group_analysis.py module.
"""
import json
import numpy as np
import os
import h5py

import preprocessing as prp
import fiber_record_analyze as fra 
from load_configuration import load_configuration
import group_analysis as ga


def setup_module(module):
    cfg = load_configuration()
    filenames_file = str(cfg['analysis_filenames_file'])
    path_to_raw_data = str(cfg['path_to_raw_data'])
    path_to_hdf5 = str(cfg['path_to_hdf5'])
    path_to_flat_data = str(cfg['path_to_flat_data'])

    analysis_filenames = prp.read_filenames(filenames_file, path_to_filenames=path_to_raw_data)
    prp.generate_hdf5_file(analysis_filenames,path_to_hdf5)
    prp.add_flattened_files_to_hdf5(path_to_flat_data, path_to_hdf5)

def teardown_module(module):
    cfg = load_configuration()
    path_to_hdf5 = str(cfg['path_to_hdf5'])
    os.remove(path_to_hdf5)

class Configure_tests():
    """
    A helper class to maintain the initialization
    of class variables that are set to the same 
    values for each test
    """
    def __init__(self):
        cfg = load_configuration()
        self.filenames_file = str(cfg['analysis_filenames_file'])
        self.path_to_raw_data = str(cfg['path_to_raw_data'])
        self.path_to_hdf5 = str(cfg['path_to_hdf5'])
        self.path_to_flat_data = str(cfg['path_to_flat_data'])
        self.test_output_directory = self.path_to_raw_data + 'test_output/'
        if not os.path.isdir(self.test_output_directory):
            os.mkdir(self.test_output_directory)

        parser = fra.add_command_line_options()
        (self.options, args) = parser.parse_args([]) #override sys.argv with an empty argument list
        self.options.smoothness = 0
        self.options.filter_freqs = None

        self.options.save_txt = False
        self.options.save_to_h5 = None
        self.options.save_and_exit = False
        self.options.save_debleach = False

        self.options.input_path = self.path_to_hdf5
        self.options.output_path = self.test_output_directory

        self.options.exp_date = '20130524'
        self.options.plot_format = '.png'


    def remove_output_directory(self):
        """
        Removes the test_output directory, 
        which should be done after every test.
        """
        for f in os.listdir(self.test_output_directory):
            file_path = os.path.join(self.test_output_directory, f)
            try:
                os.unlink(file_path)
            except Exception, e:
                print e
        os.rmdir(self.test_output_directory)




##MAKE PLOT TESTING FOR ARTIFACT IN group_regression:
##i.e. set the score as the event_index, or event_time

class Test_group_iter_list():
	pass

class Test_group_regression_plot(Configure_tests):
    def setup(self):
        Configure_tests.__init__(self)

        self.options.fluor_normalization = 'deltaF'
        self.options.exp_type = 'homecagesocial'
        self.options.event_spacing = 0
        self.options.mouse_type = 'GC5'

        self.all_data = h5py.File(self.options.input_path,'r')

    def tearDown(self):
        #Configure_tests.remove_output_directory(self)
        pass

    def test_peak_regression(self):

        self.options.intensity_metric='peak'
        self.options.time_window = '0:0'
        time_window = np.array(self.options.time_window.split(':'), dtype='float32') 
        slopes, peak_intensity, onset_next_vals, lm_results = ga.group_regression_plot(self.all_data, self.options, 
                                                                      exp_type=self.options.exp_type, 
                                                                      time_window=time_window,
                                                                      metric=self.options.intensity_metric)
        print "slopes", slopes
        print "peak_intensity", peak_intensity
        print "onset_next_vals", onset_next_vals
        print "lm_results", lm_results

        assert(np.abs(peak_intensity[0] - 1.0) < 0.000001)
        assert(np.abs(peak_intensity[1] - 0.5) < 0.000001)
        assert(np.abs(peak_intensity[2] - 1.0/3.0) < 0.000001)

        assert(np.abs(onset_next_vals[0] - 5.1) < 0.000001)
        assert(np.abs(onset_next_vals[1] - 19.0) < 0.000001)
        assert(np.abs(onset_next_vals[2] - 34.3) < 0.000001)
        assert(np.abs(lm_results.rsquared - 0.99) < 0.01)

    def test_average_regression(self):

        self.options.intensity_metric='average'
        self.options.time_window = '1:1'
        time_window = np.array(self.options.time_window.split(':'), dtype='float32') 
        slopes, peak_intensity, onset_next_vals, lm_results = ga.group_regression_plot(self.all_data, self.options, 
                                                                      exp_type=self.options.exp_type, 
                                                                      time_window=time_window,
                                                                      metric=self.options.intensity_metric)
        print "slopes", slopes
        print "peak_intensity", peak_intensity
        print "onset_next_vals", onset_next_vals
        print "lm_results", lm_results

        assert(np.abs(peak_intensity[0] - 1.0/2.0) < 0.01)
        assert(np.abs(peak_intensity[1] - 0.5/2.0) < 0.01)
        assert(np.abs(peak_intensity[2] - 1.0/3.0/2.0) < 0.01)

        assert(np.abs(onset_next_vals[0] - 5.1) < 0.000001)
        assert(np.abs(onset_next_vals[1] - 19.0) < 0.000001)
        assert(np.abs(onset_next_vals[2] - 34.3) < 0.000001)
        assert(np.abs(lm_results.rsquared - 0.99) < 0.01)


        self.options.intensity_metric='average'
        self.options.time_window = '0:0'
        time_window = np.array(self.options.time_window.split(':'), dtype='float32') 
        slopes, peak_intensity, onset_next_vals, lm_results = ga.group_regression_plot(self.all_data, self.options, 
                                                                      exp_type=self.options.exp_type, 
                                                                      time_window=time_window,
                                                                      metric=self.options.intensity_metric)
        print "slopes", slopes
        print "peak_intensity", peak_intensity
        print "onset_next_vals", onset_next_vals
        print "lm_results", lm_results

        assert(np.abs(peak_intensity[0] - 1.0) < 0.01)
        assert(np.abs(peak_intensity[1] - 0.5) < 0.01)
        assert(np.abs(peak_intensity[2] - 1.0/3.0) < 0.01)

        assert(np.abs(onset_next_vals[0] - 5.1) < 0.000001)
        assert(np.abs(onset_next_vals[1] - 19.0) < 0.000001)
        assert(np.abs(onset_next_vals[2] - 34.3) < 0.000001)
        assert(np.abs(lm_results.rsquared - 0.99) < 0.01)

    def test_time_regression(self):

        self.options.intensity_metric='event_time'
        self.options.time_window = '1:1'
        time_window = np.array(self.options.time_window.split(':'), dtype='float32') 
        slopes, peak_intensity, onset_next_vals, lm_results = ga.group_regression_plot(self.all_data, self.options, 
                                                                      exp_type=self.options.exp_type, 
                                                                      time_window=time_window,
                                                                      metric=self.options.intensity_metric)
        print "slopes", slopes
        print "peak_intensity", peak_intensity
        print "onset_next_vals", onset_next_vals
        print "lm_results", lm_results

        assert(np.abs(peak_intensity[0] - 45.6) < 0.01)
        assert(np.abs(peak_intensity[1] - 51.9) < 0.01)
        assert(np.abs(peak_intensity[2] - 72.1) < 0.01)

        assert(np.abs(onset_next_vals[0] - 5.1) < 0.000001)
        assert(np.abs(onset_next_vals[1] - 19.0) < 0.000001)
        assert(np.abs(onset_next_vals[2] - 34.3) < 0.000001)
        assert(np.abs(lm_results.rsquared - 0.80) < 0.01)


    def test_index_regression(self):

        self.options.intensity_metric='event_index'
        self.options.time_window = '1:1'
        time_window = np.array(self.options.time_window.split(':'), dtype='float32') 
        slopes, peak_intensity, onset_next_vals, lm_results = ga.group_regression_plot(self.all_data, self.options, 
                                                                      exp_type=self.options.exp_type, 
                                                                      time_window=time_window,
                                                                      metric=self.options.intensity_metric)
        print "slopes", slopes
        print "peak_intensity", peak_intensity
        print "onset_next_vals", onset_next_vals
        print "lm_results", lm_results

        assert(np.abs(peak_intensity[0] - 1) < 0.01)
        assert(np.abs(peak_intensity[1] - 2) < 0.01)
        assert(np.abs(peak_intensity[2] - 3) < 0.01)

        assert(np.abs(onset_next_vals[0] - 5.1) < 0.000001)
        assert(np.abs(onset_next_vals[1] - 19.0) < 0.000001)
        assert(np.abs(onset_next_vals[2] - 34.3) < 0.000001)
        assert(lm_results.rsquared > 0.80)


	pass


class Test_group_plot_time_series():
	pass

class Test_compare_epochs():
	pass

class Test_get_novel_social_pairs():
	pass

class Test_plot_representative_time_series():
	pass

class Test_statisticalTestOfComparison():
	pass

class Test_get_bout_averages():
	pass

class Test_plot_decay():
	pass




class Test_group_bout_heatmaps():
	"""
	Read through group_bout_heatmaps,
	it should be fine if the underlying 
	functions have been tested 
	"""
	pass

class Test_compare_start_and_end_of_epoch():
	"""
	Are we using this in the paper?
	"""
	pass

class Test_plot_compare_start_and_end():
	"""
	Are we using this in the paper?
	"""
	pass


class Test_fit_exponential():
	pass


class Test_loadFiberAnalyze():
	pass


class Test_compileAnimalScoreDictIntoArray():
	pass

class Test_group_bout_ci():
	pass