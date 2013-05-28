"""
test_fiber_record_analyze.py
This module runs unit tests on 
the fiber_record_analyze.py module.
"""

import preprocessing as prp
import fiber_record_analyze as fra 
import json
import numpy as np
import os
from load_configuration import load_configuration
from optparse import OptionParser




def setup_module(module):
    cfg = load_configuration()
    filenames_file = str(cfg['analysis_filenames_file'])
    path_to_raw_data = str(cfg['path_to_raw_data'])
    path_to_hdf5 = str(cfg['path_to_hdf5'])
    path_to_flat_data = str(cfg['path_to_flat_data'])
    print "setup_module"

    analysis_filenames = prp.read_filenames(filenames_file, path_to_filenames=path_to_raw_data)
    prp.generate_hdf5_file(analysis_filenames,path_to_hdf5)
    prp.add_flattened_files_to_hdf5(path_to_flat_data, path_to_hdf5)

def teardown_module(module):
    cfg = load_configuration()
    path_to_hdf5 = str(cfg['path_to_hdf5'])
    os.remove(path_to_hdf5)


class Test_load():
    def setup(self):
        cfg = load_configuration()
        self.filenames_file = str(cfg['analysis_filenames_file'])
        self.path_to_raw_data = str(cfg['path_to_raw_data'])
        self.path_to_hdf5 = str(cfg['path_to_hdf5'])
        self.path_to_flat_data = str(cfg['path_to_flat_data'])

        self.test_output_directory = self.path_to_raw_data + 'test_output/'
        if not os.path.isdir(self.test_output_directory):
            os.mkdir(self.test_output_directory)

    def tearDown(self):
        os.rmdir(self.test_output_directory)

    def test_load_npz_deltaF(self):
        parser = fra.add_command_line_options()
        (options, args) = parser.parse_args([]) #override sys.argv with an empty argument list

        options.smoothness = 0
        options.time_range = '20:120'
        options.fluor_normalization = 'deltaF'
        options.filter_freqs = None
        options.exp_type = 'homecagesocial'
        options.event_spacing = 0
        options.mouse_type = 'GC5'

        options.input_path = self.path_to_raw_data + '20130524/20130524-GC5-homecagesocial-0001-600patch_test.npz'
        options.trigger_path = self.path_to_raw_data + '20130524/GC5_0001_social'
        options.output_path = self.test_output_directory

        options.save_txt = False
        options.save_to_h5 = None
        options.save_and_exit = False
        options.save_debleach = False


        FA = fra.FiberAnalyze( options )
        FA.load(file_type="npz")

        assert(np.max(FA.fluor_data) > 0.9 and np.max(FA.fluor_data) < 1.1) #eyeballed dF/F based on plot of fluorescence
        print 'np.max(FA.time_stamps)', np.max(FA.time_stamps)
        print 'np.min(FA.time_stamps)', np.min(FA.time_stamps)

        assert(np.abs(np.max(FA.time_stamps) - 120) < 0.01)
        assert(np.abs(np.min(FA.time_stamps) - 20) < 0.01)

        fii = np.where(np.array(FA.trigger_data ))
        first_event_index = fii[0][0]
        print 'FA.time_stamps[first_event_index]',FA.time_stamps[first_event_index]
        assert(np.abs(FA.time_stamps[first_event_index] - 49.8) < 0.00001)
        end_last_event_index = fii[0][-1]
        print 'FA.time_stamps[end_last_event_index+1]', FA.time_stamps[end_last_event_index+1]
        assert(np.abs(FA.time_stamps[end_last_event_index + 1] - 101.5) < 0.00001)


    def test_load_hdf5(self):
        parser = fra.add_command_line_options()
        (options, args) = parser.parse_args([]) #override sys.argv with an empty argument list

        options.smoothness = 0
        options.time_range = '10:-1'
        options.fluor_normalization = 'deltaF'
        options.filter_freqs = None
        options.exp_type = 'homecagesocial'
        options.event_spacing = 0
        options.mouse_type = 'GC5'

        options.input_path = self.path_to_hdf5
        options.output_path = self.test_output_directory

        options.save_txt = False
        options.save_to_h5 = None
        options.save_and_exit = False
        options.save_debleach = False

        FA = fra.FiberAnalyze( options )
        FA.subject_id = '0001'
        FA.exp_date = '20130524'
        FA.exp_type = 'homecagesocial'
        FA.load(file_type="hdf5")

        assert(np.max(FA.fluor_data) > 0.9 and np.max(FA.fluor_data) < 1.1) #eyeballed dF/F based on plot of fluorescence
        assert(np.abs(np.max(FA.time_stamps) - 150) < 0.01)
        assert(np.abs(np.min(FA.time_stamps) - 10) < 0.01)

        fii = np.where(np.array(FA.trigger_data ))
        first_event_index = fii[0][0]
        print 'FA.time_stamps[first_event_index]',FA.time_stamps[first_event_index]
        assert(np.abs(FA.time_stamps[first_event_index] - 49.8) < 0.00001)
        end_last_event_index = fii[0][-1]
        print 'FA.time_stamps[end_last_event_index+1]', FA.time_stamps[end_last_event_index+1]
        assert(np.abs(FA.time_stamps[end_last_event_index + 1] - 101.5) < 0.00001)



class Test_crop_data():
    pass


class Test_load_trigger_data():
    pass



class Test_normalize_fluorescence_data():
    pass


class Test_load_event_data():
    pass



class Test_set_resolution():
    pass



class Test_plot_basic_tseries():
    pass


class Test_save_time_series():
    pass


class Test_plot_next_event_vs_intensity():
    pass


class Test_get_time_chunks_around_events():
    pass


class Test_get_peak():
    pass

class Test_plot_peaks_vs_time():
    pass


class Test_plot_perievent_hist():
    pass

class Test_get_event_times():
    pass

class Test_convert_seconds_to_index():
    pass


class Test_get_areas_under_curve():
    pass


class Test_notch_filter():
    #Though are we using this function??
    pass

class Test_get_sucrose_event_times():
    pass




