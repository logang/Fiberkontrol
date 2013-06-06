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


class Test_load(Configure_tests):
    def setup(self):
        Configure_tests.__init__(self)

    def tearDown(self):
        Configure_tests.remove_output_directory(self)

    def test_load_npz_deltaF(self):
        self.options.time_range = '20:120'
        self.options.fluor_normalization = 'deltaF'
        self.options.exp_type = 'homecagesocial'
        self.options.event_spacing = 0
        self.options.mouse_type = 'GC5'

        self.options.input_path = self.path_to_raw_data + '20130524/20130524-GC5-homecagesocial-0001-600patch_test.npz'
        self.options.trigger_path = self.path_to_raw_data + '20130524/GC5_0001_social'

        FA = fra.FiberAnalyze( self.options )
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
        self.options.time_range = '10:-1'
        self.options.fluor_normalization = 'deltaF'
        self.options.exp_type = 'homecagesocial'
        self.options.event_spacing = 0
        self.options.mouse_type = 'GC5'


        FA = fra.FiberAnalyze( self.options )
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
        self.options.time_range = '0:-1'
        self.options.fluor_normalization = 'raw'
        self.options.exp_type = 'homecagesocial'
        self.options.event_spacing = 0
        self.options.mouse_type = 'GC5'

        FA = fra.FiberAnalyze( self.options )
        FA.subject_id = '0001'
        FA.exp_date = '20130524'
        FA.exp_type = 'homecagesocial'
        FA.load(file_type="hdf5")

        assert(np.max(FA.fluor_data) > 8.6 and np.max(FA.fluor_data) < 8.7) #eyeballed dF/F based on plot of fluorescence
        assert(np.abs(np.max(FA.time_stamps) - 150) < 0.01)
        assert(np.abs(np.min(FA.time_stamps) - 0) < 0.01)

    def test_normalize_fluorescence_data_deltaF(self):
    
        self.options.time_range = '20:-1'
        self.options.fluor_normalization = 'deltaF'
        self.options.exp_type = 'homecagenovel'
        self.options.event_spacing = 0
        self.options.mouse_type = 'GC5'

        FA = fra.FiberAnalyze( self.options )
        FA.subject_id = '0001'
        FA.exp_date = '20130524'
        FA.exp_type = 'homecagenovel'
        FA.load(file_type="hdf5")

        assert(np.max(FA.fluor_data) > 0.2 and np.max(FA.fluor_data) < 0.3) #eyeballed dF/F based on plot of fluorescence
        assert(np.abs(np.max(FA.time_stamps) - 150) < 0.01)
        assert(np.abs(np.min(FA.time_stamps) - 20) < 0.01)

    def test_normalize_fluorescence_data_standardize(self):
        self.options.time_range = '20:-1'
        self.options.fluor_normalization = 'standardize'
        self.options.exp_type = 'homecagesocial'
        self.options.event_spacing = 0
        self.options.mouse_type = 'GC5'

        FA = fra.FiberAnalyze( self.options )
        FA.subject_id = '0001'
        FA.exp_date = '20130524'
        FA.exp_type = 'homecagesocial'
        FA.load(file_type="hdf5")

        assert(np.max(FA.fluor_data) > 0.99 and np.max(FA.fluor_data) < 1.001) #should be standardized to between 0 and 1
        assert(np.min(FA.fluor_data) > -0.001 and np.min(FA.fluor_data) < 0.03) 
        assert(np.abs(np.max(FA.time_stamps) - 150) < 0.01)
        assert(np.abs(np.min(FA.time_stamps) - 20) < 0.01)


class Test_crop_data(Configure_tests):
    def setup(self):
        Configure_tests.__init__(self)

        self.options.fluor_normalization = 'deltaF'
        self.options.exp_type = 'homecagesocial'
        self.options.event_spacing = 0
        self.options.mouse_type = 'GC5'

    def tearDown(self):
        Configure_tests.remove_output_directory(self)

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



class Test_plot_basic_tseries(Configure_tests):
    def setup(self):
        Configure_tests.__init__(self)

        self.options.fluor_normalization = 'deltaF'
        self.options.event_spacing = 0
        self.options.mouse_type = 'GC5'
        self.options.time_range = '0:-1'

    def tearDown(self):
        Configure_tests.remove_output_directory(self)


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


class Test_get_event_times(Configure_tests):
    #Still need to test sucrose!
    def setup(self):
        Configure_tests.__init__(self)

        self.options.fluor_normalization = 'deltaF'
        self.options.event_spacing = 0
        self.options.mouse_type = 'GC5'
        self.options.time_range = '0:-1'
        self.options.exp_type = 'homecagesocial'

    def tearDown(self):
        Configure_tests.remove_output_directory(self)


    def test_get_event_times_no_spacing(self):
        FA = fra.FiberAnalyze( self.options )
        FA.subject_id = '0001'
        FA.exp_date = '20130524'
        FA.exp_type = 'homecagesocial'
        FA.load(file_type="hdf5")

        event_times = FA.get_event_times( edge="rising", exp_type='homecagesocial')
        assert(np.abs(event_times[0] - 49.8) < 0.01)
        assert(np.abs(event_times[1] - 100) < 0.01)
        event_times = FA.get_event_times( edge="falling", exp_type='homecagesocial')
        assert(np.abs(event_times[0] - 50.7) < 0.01)
        assert(np.abs(event_times[1] - 101.5) < 0.01)


        FA = fra.FiberAnalyze( self.options )
        FA.subject_id = '0002'
        FA.exp_date = '20130524'
        FA.exp_type = 'homecagesocial'
        FA.load(file_type="hdf5")

        event_times = FA.get_event_times( edge='rising', exp_type='homecagesocial')
        assert(np.abs(event_times[0] - 40.8) < 0.01)
        assert(np.abs(event_times[1] - 44.6) < 0.01)
        assert(np.abs(event_times[2] - 110) < 0.01)
        event_times = FA.get_event_times( edge='falling', exp_type='homecagesocial')
        assert(np.abs(event_times[0] - 42.3) < 0.01)
        assert(np.abs(event_times[1] - 45.1) < 0.01)
        assert(np.abs(event_times[2] - 111.5) < 0.01)


    def test_get_event_times_with_spacing(self):
        self.options.event_spacing = 2.4

        FA = fra.FiberAnalyze( self.options )
        FA.subject_id = '0002'
        FA.exp_date = '20130524'
        FA.exp_type = 'homecagesocial'
        FA.load(file_type="hdf5")

        event_times = FA.get_event_times( edge='rising', exp_type='homecagesocial')
        assert(np.max(np.shape(event_times)) == 2)
        assert(np.abs(event_times[1] - 110) < 0.01)


    def test_get_event_times_sucrose(self):
        #TO DO
        pass

    

class Test_plot_next_event_vs_intensity(Configure_tests):
    def setup(self):
        Configure_tests.__init__(self)

        self.options.fluor_normalization = 'deltaF'
        self.options.event_spacing = 0
        self.options.mouse_type = 'GC5'
        self.options.time_range = '0:-1'
        self.options.exp_type = 'homecagesocial'

    def tearDown(self):
        Configure_tests.remove_output_directory(self)

    def test_deterministic_time_series(self):
        """
        Test various combinations of intensity_measure and 
        next_event_measure as inputs to 
        FA.plot_next_event_vs_intensity.
        This also serves to test score_of_chunks()
        Use test data '0005', which is completely deterministic
        (albeit with non-realistic looking gcamp spikes).
        """

        FA = fra.FiberAnalyze( self.options )
        FA.subject_id = '0005'
        FA.exp_date = '20130524'
        FA.exp_type = 'homecagesocial'
        FA.load(file_type="hdf5")

        (intensity, next_vals) = FA.plot_next_event_vs_intensity(
                                        intensity_measure="peak", 
                                        next_event_measure="onset", 
                                        window=[0, 1], 
                                        out_path=None, 
                                        plotit=False)

        assert(np.abs(intensity[0] - 1.0) < 0.000001 )
        assert(np.abs(intensity[1] - 0.5) < 0.000001 )
        assert(np.abs(intensity[2] - 1.0/3.0) < 0.000001 )

        assert(np.abs(next_vals[0] - 5.1) < 0.000001 )
        assert(np.abs(next_vals[1] - 19) < 0.000001 )
        assert(np.abs(next_vals[2] - 34.3) < 0.000001 )


        (intensity, next_vals) = FA.plot_next_event_vs_intensity(
                                        intensity_measure="integrated", 
                                        next_event_measure="onset", 
                                        window=[0, 1], 
                                        out_path=None, 
                                        plotit=False)

        print "intensity", intensity
        assert(np.abs(intensity[0] - 1.0) < 0.000001 )
        assert(np.abs(intensity[1] - 0.5) < 0.000001 )
        assert(np.abs(intensity[2] - 1.0/3.0) < 0.000001 )

        assert(np.abs(next_vals[0] - 5.1) < 0.000001 )
        assert(np.abs(next_vals[1] - 19) < 0.000001 )
        assert(np.abs(next_vals[2] - 34.3) < 0.000001 )


        (intensity, next_vals) = FA.plot_next_event_vs_intensity(
                                intensity_measure="integrated", 
                                next_event_measure="onset", 
                                window=[1, 1], 
                                out_path=None, 
                                plotit=False)

        assert(np.abs(intensity[0] - 1.0/2.0) < 0.01 )
        assert(np.abs(intensity[1] - 0.5/2.0) < 0.01 )
        assert(np.abs(intensity[2] - 1.0/3.0/2.0) < 0.01 )

        assert(np.abs(next_vals[0] - 5.1) < 0.000001 )
        assert(np.abs(next_vals[1] - 19) < 0.000001 )
        assert(np.abs(next_vals[2] - 34.3) < 0.000001 )


        (intensity, next_vals) = FA.plot_next_event_vs_intensity(
                                intensity_measure="integrated", 
                                next_event_measure="length", 
                                window=[0, 1], 
                                out_path=None, 
                                plotit=False)

        assert(np.abs(intensity[0] - 1.0) < 0.000001 )
        assert(np.abs(intensity[1] - 0.5) < 0.000001 )
        assert(np.abs(intensity[2] - 1.0/3.0) < 0.000001 )

        assert(np.abs(next_vals[0] - 1.2) < 0.000001 )
        assert(np.abs(next_vals[1] - 1.6) < 0.000001 )
        assert(np.abs(next_vals[2] - 1.5) < 0.000001 )


        (intensity, next_vals) = FA.plot_next_event_vs_intensity(
                                intensity_measure="event_time", 
                                next_event_measure="length", 
                                window=[0, 1], 
                                out_path=None, 
                                plotit=False)

        assert(np.abs(intensity[0] - 45.6) < 0.000001 )
        assert(np.abs(intensity[1] - 51.9) < 0.000001 )
        assert(np.abs(intensity[2] - 72.1) < 0.000001 )

        assert(np.abs(next_vals[0] - 1.2) < 0.000001 )
        assert(np.abs(next_vals[1] - 1.6) < 0.000001 )
        assert(np.abs(next_vals[2] - 1.5) < 0.000001 )

        (intensity, next_vals) = FA.plot_next_event_vs_intensity(
                                intensity_measure="event_index", 
                                next_event_measure="length", 
                                window=[0, 1], 
                                out_path=None, 
                                plotit=False)

        assert(np.abs(intensity[0] - 1) < 0.000001 )
        assert(np.abs(intensity[1] - 2) < 0.000001 )
        assert(np.abs(intensity[2] - 3) < 0.000001 )

        assert(np.abs(next_vals[0] - 1.2) < 0.000001 )
        assert(np.abs(next_vals[1] - 1.6) < 0.000001 )
        assert(np.abs(next_vals[2] - 1.5) < 0.000001 )

        (intensity, next_vals) = FA.plot_next_event_vs_intensity(
                        intensity_measure="spacing", 
                        next_event_measure="length", 
                        window=[0, 1], 
                        out_path=None, 
                        plotit=False)

        assert(np.abs(intensity[0] - 5.1) < 0.000001 )
        assert(np.abs(intensity[1] - 19) < 0.000001 )
        assert(np.abs(intensity[2] - 34.3) < 0.000001 )

        assert(np.abs(next_vals[0] - 1.2) < 0.000001 )
        assert(np.abs(next_vals[1] - 1.6) < 0.000001 )
        assert(np.abs(next_vals[2] - 1.5) < 0.000001 )



        (intensity, next_vals) = FA.plot_next_event_vs_intensity(
                intensity_measure="epoch_length", 
                next_event_measure="length", 
                window=[0, 1], 
                out_path=None, 
                plotit=False)

        assert(np.abs(intensity[0] - 1.2) < 0.000001 )
        assert(np.abs(intensity[1] - 1.2) < 0.000001 )
        assert(np.abs(intensity[2] - 1.6) < 0.000001 )


class Test_get_time_chunks_around_events(Configure_tests):
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
        """
        Test various windows, baseline_windows as
        inputs to FA.get_time_chunks_around_events().
        Use test data '0005', which is completely deterministic.
        """

        FA = fra.FiberAnalyze( self.options )
        FA.subject_id = '0005'
        FA.exp_date = '20130524'
        FA.exp_type = 'homecagesocial'
        FA.load(file_type="hdf5")

        event_times = FA.get_event_times(edge="rising", exp_type=FA.exp_type)
        assert(np.abs(event_times[0] - 45.6) < 0.01)
        assert(np.abs(event_times[1] - 51.9) < 0.01)
        assert(np.abs(event_times[2] - 72.1) < 0.01)
        assert(np.abs(event_times[3] - 108.0) < 0.01)


        test_window = [0,1]
        baseline_window = -1
        time_chunks = FA.get_time_chunks_around_events(
                                        data=FA.fluor_data,
                                        event_times=event_times,
                                        window=test_window,
                                        baseline_window=baseline_window)
        window_length = FA.convert_seconds_to_index(test_window[1])
        assert(np.abs(time_chunks[0][0] - 1.0) < 0.000001 )
        assert(np.abs(time_chunks[0][-1] - 1.0) < 0.000001 )
        assert(np.abs(time_chunks[1][0] - 0.5) < 0.000001 )
        assert(np.abs(time_chunks[2][0] - 1.0/3.0) < 0.000001 )
        assert(np.abs(time_chunks[3][0] - 1.0/4.0) < 0.000001 )
        assert(np.abs(np.max(np.shape(time_chunks[0])) - window_length) < 0.000001)
        assert(np.abs(np.max(np.shape(time_chunks[1])) - window_length) < 0.000001)
        assert(np.abs(np.max(np.shape(time_chunks[2])) - window_length) < 0.000001)



        test_window = [0,1]
        baseline_window=[0, 1]
        time_chunks = FA.get_time_chunks_around_events(
                                        data=FA.fluor_data,
                                        event_times=event_times,
                                        window=test_window,
                                        baseline_window=baseline_window)
        window_length = FA.convert_seconds_to_index(test_window[1])
        assert(np.abs(time_chunks[0][0] - 0.0) < 0.000001 )
        assert(np.abs(time_chunks[1][0] - 0.0) < 0.000001 )
        assert(np.abs(time_chunks[2][0] - 0.0) < 0.000001 )
        assert(np.abs(time_chunks[3][0] - 0.0) < 0.000001 )
        assert(np.abs(np.max(np.shape(time_chunks[0])) - window_length) < 0.000001)


        test_window = [0,1]
        baseline_window=[1, 1]
        time_chunks = FA.get_time_chunks_around_events(
                                        data=FA.fluor_data,
                                        event_times=event_times,
                                        window=test_window,
                                        baseline_window=baseline_window)
        window_length = FA.convert_seconds_to_index(test_window[1])
        assert(np.abs(time_chunks[0][0] - 1.0) < 0.000001 )
        assert(np.abs(time_chunks[1][0] - 0.5) < 0.000001 )
        assert(np.abs(time_chunks[2][0] - 1.0/3.0) < 0.000001 )
        assert(np.abs(time_chunks[3][0] - 1.0/4.0) < 0.000001 )
        assert(np.abs(np.max(np.shape(time_chunks[0])) - window_length) < 0.000001)
        assert(np.abs(np.max(np.shape(time_chunks[1])) - window_length) < 0.000001)
        assert(np.abs(np.max(np.shape(time_chunks[2])) - window_length) < 0.000001)


        test_window = [1,1]
        baseline_window=[1, 1]
        time_chunks = FA.get_time_chunks_around_events(
                                        data=FA.fluor_data,
                                        event_times=event_times,
                                        window=test_window,
                                        baseline_window=baseline_window)
        window_length = FA.convert_seconds_to_index(sum(test_window))
        assert(np.abs(time_chunks[0][0] - 0.0) < 0.000001 )
        assert(np.abs(time_chunks[0][-1] - 1.0) < 0.000001 )
        assert(np.abs(time_chunks[1][0] - 0.0) < 0.000001 )
        assert(np.abs(time_chunks[2][0] - 0.0) < 0.000001 )
        assert(np.abs(time_chunks[3][0] - 0.0) < 0.000001 )
        assert(np.abs(np.max(np.shape(time_chunks[0])) - window_length) < 0.000001)


        test_window = [0,1]
        baseline_window='full'
        time_chunks = FA.get_time_chunks_around_events(
                                        data=FA.fluor_data,
                                        event_times=event_times,
                                        window=test_window,
                                        baseline_window=baseline_window)
        window_length = FA.convert_seconds_to_index(sum(test_window))
        print "time_chunks[0]", time_chunks[0]
        assert(np.abs(time_chunks[0][0] - 0.0) < 0.000001 )
        assert(np.abs(time_chunks[0][-1] - 0.0) < 0.000001 )
        assert(np.abs(time_chunks[1][0] - 0.0) < 0.000001 )
        assert(np.abs(np.max(np.shape(time_chunks[0])) - window_length) < 0.000001)


        test_window = [1,1]
        baseline_window='full'
        time_chunks = FA.get_time_chunks_around_events(
                                        data=FA.fluor_data,
                                        event_times=event_times,
                                        window=test_window,
                                        baseline_window=baseline_window)
        window_length = FA.convert_seconds_to_index(sum(test_window))
        assert(np.abs(time_chunks[0][0] - 0.0) < 0.000001 )
        assert(np.abs(time_chunks[0][window_length/2 + 1] - 1.0) < 0.000001 )
        assert(np.abs(time_chunks[0][-1] - 1.0) < 0.000001 )
        assert(np.abs(time_chunks[1][-1] - 0.5) < 0.000001 )
        assert(np.abs(np.max(np.shape(time_chunks[0])) - window_length) < 0.000001)


        test_window = [0, 3.5] #There is a tail of 2 seconds of activity past the end of the stimulus
        baseline_window=-1
        time_chunks = FA.get_time_chunks_around_events(
                                        data=FA.fluor_data,
                                        event_times=event_times,
                                        window=test_window,
                                        baseline_window=baseline_window)
        window_length = FA.convert_seconds_to_index(test_window[1])
        assert(np.abs(time_chunks[0][-1] - 0.0) < 0.000001 )
        assert(np.abs(time_chunks[1][-1] - 0.0) < 0.000001 )
        assert(np.abs(time_chunks[2][-1] - 1.0/3.0) < 0.000001 )
        assert(np.abs(time_chunks[3][-1] - 0.0) < 0.000001 )
        assert(np.abs(np.max(np.shape(time_chunks[0])) - window_length) < 0.000001)
        assert(np.abs(np.max(np.shape(time_chunks[1])) - window_length) < 0.000001)
        assert(np.abs(np.max(np.shape(time_chunks[2])) - window_length) < 0.000001)
        assert(np.abs(np.max(np.shape(time_chunks[3])) - window_length) < 0.000001)


class Test_plot_perievent_hist(Configure_tests):
    def setup(self):
        Configure_tests.__init__(self)

        self.options.fluor_normalization = 'deltaF'
        self.options.event_spacing = 0
        self.options.mouse_type = 'GC5'
        self.options.time_range = '0:-1'
        self.options.exp_type = 'homecagesocial'

    def tearDown(self):
        Configure_tests.remove_output_directory(self)

    def test_perievent(self):

        FA = fra.FiberAnalyze( self.options )
        FA.subject_id = '0005'
        FA.exp_date = '20130524'
        FA.exp_type = 'homecagesocial'
        FA.load(file_type="hdf5")


        event_times = FA.get_event_times(edge="rising", 
                                         exp_type=FA.exp_type)

        assert(np.abs(event_times[0] - 45.6) < 0.01)
        assert(np.abs(event_times[1] - 51.9) < 0.01)
        assert(np.abs(event_times[2] - 72.1) < 0.01)
        assert(np.abs(event_times[3] - 108.0) < 0.01)
        #Remember that there is a tail of 2 seconds of activity
        #that extends past the end of the event

        time_window = [4, 4]
        baseline_window = 'full'
        (time_arr, x) = FA.plot_perievent_hist(event_times, 
                                             window=time_window, 
                                             out_path=None,
                                             plotit=False, 
                                             subplot=None, 
                                             baseline_window=baseline_window )

        print "time_arr", time_arr[:][:]
        print "x", x
        assert(np.abs(x[0] - (-4.0))<0.00001)
        assert(np.abs(x[-1] - 4.0)<0.01)
        assert(np.abs(time_arr[0][0] - 0.0)<0.00001) #beginning of first time chunk
        assert(np.abs(time_arr[0][-1] - 0.0)<0.00001)#end of first time chunk
        assert(np.abs(time_arr[0][1] - 1.0)<0.00001) #Second time chunk

        assert(np.abs(time_arr[time_arr.shape[0]/2+1][0] - 1.0)<0.00001) #middle of first chunk (at event onset)
        assert(np.abs(time_arr[time_arr.shape[0]/2+1][1] - 0.5)<0.00001) #middle of second chunk (at event onset)
        assert(np.abs(time_arr[time_arr.shape[0]/2+1][2] - 1.0/3.0)<0.00001)#middle of third chunk (at event onset)











    pass


class Test_plot_peaks_vs_time():
    pass









class Test_convert_seconds_to_index():
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

class Test_get_areas_under_curve():
    pass

class Test_save_time_series():
    #This function is tested in test_preprocessing
    pass




