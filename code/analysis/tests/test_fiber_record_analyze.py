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



def setup_module(test_fiber_record_analyze):
    cfg = load_configuration()
    filenames_file = str(cfg['analysis_filenames_file'])
    path_to_raw_data = str(cfg['path_to_raw_data'])
    path_to_hdf5 = str(cfg['path_to_hdf5'])
    path_to_flat_data = str(cfg['path_to_flat_data'])

    analysis_filenames = prp.read_filenames(filenames_file)
    prp.generate_hdf5_file(analysis_filenames,path_to_hdf5)
    prp.add_flattened_files_to_hdf5(path_to_flat_data, path_to_hdf5)


class Test_load():
    def setup(self):
        cfg = load_configuration()
        self.filenames_file = str(cfg['analysis_filenames_file'])
        self.path_to_raw_data = str(cfg['path_to_raw_data'])
        self.path_to_hdf5 = str(cfg['path_to_hdf5'])
        self.path_to_flat_data = str(cfg['path_to_flat_data'])

    def test_load_npz(self):
        assert('a' == 'b')

    def test_load_hdf5(self):
        pass



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




