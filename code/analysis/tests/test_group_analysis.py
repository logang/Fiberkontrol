"""
test_group_analysis.py
This module runs unit tests on 
the group_analysis.py module.
"""

import preprocessing as prp
import fiber_record_analyze as fra 
import json
import numpy as np
import os
from load_configuration import load_configuration


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

        self.options.exp_date


class Test_group_bout_heatmaps():
	 def setup(self):
        Configure_tests.__init__(self)

        self.options.fluor_normalization = 'deltaF'
        self.options.event_spacing = 0
        self.options.mouse_type = 'GC5'
        self.options.time_range = '0:-1'
        self.options.exp_type = 'homecagesocial'

    def tearDown(self):
        Configure_tests.remove_output_directory(self)

    def test_windows(self):

	pass


class Test_group_iter_list():
	pass


class Test_group_regression_plot():
	pass


class Test_group_bout_ci():
	pass


class Test_group_plot_time_series():
	pass

class Test_plot_representative_time_series():
	pass


class Test_get_novel_social_pairs():
	pass


class Test_plot_compare_start_and_end():
	pass


class Test_compare_start_and_end_of_epoch():
	pass


class Test_statisticalTestOfComparison():
	pass


class Test_compare_epochs():
	pass


class Test_get_bout_averages():
	pass

class Test_plot_decay():
	pass


class Test_fit_exponential():
	pass


class Test_loadFiberAnalyze():
	pass


class Test_compileAnimalScoreDictIntoArray():
	pass