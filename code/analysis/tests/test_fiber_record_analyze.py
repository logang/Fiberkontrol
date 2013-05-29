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
        for f in os.listdir(self.test_output_directory):
            file_path = os.path.join(self.test_output_directory, f)
            try:
                os.unlink(file_path)
            except Exception, e:
                print e
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

    def test_load_hdf5_deltaF(self):
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


    def test_normalize_fluorescence_data_raw(self):
        parser = fra.add_command_line_options()
        (options, args) = parser.parse_args([]) #override sys.argv with an empty argument list

        options.smoothness = 0
        options.time_range = '0:-1'
        options.fluor_normalization = 'raw'
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

        assert(np.max(FA.fluor_data) > 8.6 and np.max(FA.fluor_data) < 8.7) #eyeballed dF/F based on plot of fluorescence
        assert(np.abs(np.max(FA.time_stamps) - 150) < 0.01)
        assert(np.abs(np.min(FA.time_stamps) - 0) < 0.01)

    def test_normalize_fluorescence_data_deltaF(self):
        parser = fra.add_command_line_options()
        (options, args) = parser.parse_args([]) #override sys.argv with an empty argument list

        options.smoothness = 0
        options.time_range = '20:-1'
        options.fluor_normalization = 'deltaF'
        options.filter_freqs = None
        options.exp_type = 'homecagenovel'
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
        FA.exp_type = 'homecagenovel'
        FA.load(file_type="hdf5")

        assert(np.max(FA.fluor_data) > 0.2 and np.max(FA.fluor_data) < 0.3) #eyeballed dF/F based on plot of fluorescence
        assert(np.abs(np.max(FA.time_stamps) - 150) < 0.01)
        assert(np.abs(np.min(FA.time_stamps) - 20) < 0.01)

    def test_normalize_fluorescence_data_standardize(self):
        parser = fra.add_command_line_options()
        (options, args) = parser.parse_args([]) #override sys.argv with an empty argument list
        options.smoothness = 0
        options.filter_freqs = None

        options.time_range = '20:-1'
        options.fluor_normalization = 'standardize'
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

        assert(np.max(FA.fluor_data) > 0.99 and np.max(FA.fluor_data) < 1.001) #should be standardized to between 0 and 1
        assert(np.min(FA.fluor_data) > -0.001 and np.min(FA.fluor_data) < 0.03) 
        assert(np.abs(np.max(FA.time_stamps) - 150) < 0.01)
        assert(np.abs(np.min(FA.time_stamps) - 20) < 0.01)


class Test_crop_data():
    def setup(self):
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

        self.options.fluor_normalization = 'deltaF'
        self.options.exp_type = 'homecagesocial'
        self.options.event_spacing = 0
        self.options.mouse_type = 'GC5'

        self.options.input_path = self.path_to_hdf5
        self.options.output_path = self.test_output_directory

        self.options.save_txt = False
        self.options.save_to_h5 = None
        self.options.save_and_exit = False
        self.options.save_debleach = False

    def tearDown(self):
        for f in os.listdir(self.test_output_directory):
            file_path = os.path.join(self.test_output_directory, f)
            try:
                os.unlink(file_path)
            except Exception, e:
                print e
        os.rmdir(self.test_output_directory)

    def test_crop_data_start_equal_end(self):
        self.options.time_range = '10:10' #this should be invalid, should lead to no cropping

        FA = fra.FiberAnalyze( self.options )
        FA.subject_id = '0001'
        FA.exp_date = '20130524'
        FA.exp_type = 'homecagesocial'
        FA.load(file_type="hdf5")

        assert(np.abs(np.max(FA.time_stamps) - 150) < 0.01)
        assert(np.abs(np.min(FA.time_stamps) - 0) < 0.01)

    def test_crop_data_start_greater_than_end(self):
        self.options.time_range = '100:10' #this should be invalid, should lead to no cropping

        FA = fra.FiberAnalyze( self.options )
        FA.subject_id = '0001'
        FA.exp_date = '20130524'
        FA.exp_type = 'homecagesocial'
        FA.load(file_type="hdf5")

        assert(np.abs(np.max(FA.time_stamps) - 150) < 0.01)
        assert(np.abs(np.min(FA.time_stamps) - 0) < 0.01)

    def test_crop_data_negative_start(self):
        self.options.time_range = '-10:10' #this should lead to start=0
       
        FA = fra.FiberAnalyze( self.options )
        FA.subject_id = '0001'
        FA.exp_date = '20130524'
        FA.exp_type = 'homecagesocial'
        FA.load(file_type="hdf5")

        assert(np.abs(np.max(FA.time_stamps) - 10) < 0.01)
        assert(np.abs(np.min(FA.time_stamps) - 0) < 0.01)

    def test_crop_data_negative_end(self):
        self.options.time_range = '0:-10' #this should be invalid, should lead to no cropping
       
        FA = fra.FiberAnalyze( self.options )
        FA.subject_id = '0001'
        FA.exp_date = '20130524'
        FA.exp_type = 'homecagesocial'
        FA.load(file_type="hdf5")

        assert(np.abs(np.max(FA.time_stamps) - 150) < 0.01)
        assert(np.abs(np.min(FA.time_stamps) - 0) < 0.01)

    def test_crop_data_negative_end(self):
        self.options.time_range = '0:-10' #this should be invalid, should lead to no cropping
       
        FA = fra.FiberAnalyze( self.options )
        FA.subject_id = '0001'
        FA.exp_date = '20130524'
        FA.exp_type = 'homecagesocial'
        FA.load(file_type="hdf5")

        assert(np.abs(np.max(FA.time_stamps) - 150) < 0.01)
        assert(np.abs(np.min(FA.time_stamps) - 0) < 0.01)



class Test_plot_basic_tseries():
    def setup(self):
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

        self.options.fluor_normalization = 'deltaF'
        self.options.event_spacing = 0
        self.options.mouse_type = 'GC5'
        self.options.time_range = '0:-1'

        self.options.input_path = self.path_to_hdf5
        self.options.output_path = self.test_output_directory

        self.options.save_txt = False
        self.options.save_to_h5 = None
        self.options.save_and_exit = False
        self.options.save_debleach = False

    def tearDown(self):
        for f in os.listdir(self.test_output_directory):
            file_path = os.path.join(self.test_output_directory, f)
            try:
                os.unlink(file_path)
            except Exception, e:
                print e

        os.rmdir(self.test_output_directory)


    def test_plot_basic_tseries_homecagesocial(self):
        self.options.exp_type = 'homecagesocial'

        FA = fra.FiberAnalyze( self.options )
        FA.subject_id = '0001'
        FA.exp_date = '20130524'
        FA.exp_type = 'homecagesocial'
        FA.load(file_type="hdf5")

        (fluor_data, trigger_high_locations) = FA.plot_basic_tseries(out_path=self.test_output_directory, 
                                                                     window=None, 
                                                                     resolution=None, 
                                                                     plot_all_trigger_data=False )
        assert(np.abs(trigger_high_locations[0] - 49.8) < 0.01)
        print "trigger_high_locations[-1]", trigger_high_locations[-1]
        assert(np.abs(trigger_high_locations[-1] - 101.5) < 0.01)
        assert(np.max(FA.fluor_data) > 0.9 and np.max(FA.fluor_data) < 1.1)


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


class Test_load_trigger_data():
    #should be tested
    #need to add sucrose data
    pass





