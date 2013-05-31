"""
test_preprocessing.py
This module runs unit tests on 
the preprocessing.py module.
"""

import preprocessing as prp
import json
import h5py
import numpy as np
import os
from load_configuration import load_configuration

            
class Test_read_filenames():
    def setup(self):
        cfg = load_configuration()
        self.filenames_file = str(cfg['analysis_filenames_file'])
        self.path_to_raw_data = str(cfg['path_to_raw_data'])
        self.actual_filenames = ['20130524/20130524-GC5-homecagesocial-0001-600patch_test.npz',
                                 '20130524/20130524-GC5-homecagenovel-0001-600patch_test.npz',
                                 '20130524/20130524-GC5-homecagesocial-0002-600patch_test.npz',
                                 '20130524/20130524-GC5-homecagenovel-0002-600patch_test.npz',
                                 '20130523/20130523-GC5_NAcprojection-homecagesocial-0003-600patch_test.npz',
                                 '20130523/20130523-GC5_NAcprojection-homecagenovel-0003-600patch_test.npz',
                                 '20130523/20130523-GC5_NAcprojection-homecagesocial-0004-600patch_test.npz',
                                 '20130523/20130523-GC5_NAcprojection-homecagenovel-0004-600patch_test.npz',
                                 '20130524/20130524-GC5-homecagesocial-0005-600patch_test.npz',
                                 '20130524/20130524-GC5-homecagenovel-0005-600patch_test.npz',
                                ]

    def tearDown(self):
        pass

    def test_no_path_just_files(self):
        analysis_filenames = prp.read_filenames(self.filenames_file)
        assert analysis_filenames == self.actual_filenames

    def test_with_path(self):
        analysis_filenames = prp.read_filenames(self.filenames_file, self.path_to_raw_data)

        for i in range(len(analysis_filenames)):
            assert analysis_filenames[i] == str(self.path_to_raw_data) + self.actual_filenames[i]
        # assert analysis_filenames == ['/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20130524/20130524-GC5-homecagesocial-0001-600patch_test.npz',
        #                               '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20130524/20130524-GC5-homecagenovel-0001-600patch_test.npz']


class Test_generate_hdf5_file():
    def setup(self):
        self.epsilon = 0.000001 #used to check equality of two floats

        cfg = load_configuration()
        self.filenames_file = str(cfg['analysis_filenames_file'])
        self.path_to_raw_data = str(cfg['path_to_raw_data'])
        self.path_to_hdf5 = str(cfg['path_to_hdf5'])

        self.analysis_filenames = prp.read_filenames(self.filenames_file, self.path_to_raw_data)
        if os.path.isfile(self.path_to_hdf5):
            os.remove(self.path_to_hdf5)


    def tearDown(self):
        #self.f.close()
        os.remove(self.path_to_hdf5)

    def test_add_multiple_animals(self):
        prp.generate_hdf5_file(self.analysis_filenames,self.path_to_hdf5)
        # success = 0;
        # try: #this test takes too long and is redundant
        #     prp.generate_hdf5_file(self.analysis_filenames,self.path_to_hdf5) #ensure that you can rewrite over the same file
        #     success = 1;
        # except:
        #     success = 0;
        # assert success == 1

        print "analysis_filenames", self.analysis_filenames
        self.f = h5py.File(self.path_to_hdf5)
        f = self.f
        assert(unicode('0001') in f.keys())
        assert(unicode('0002') in f.keys())
        assert(unicode('0003') in f.keys())
        assert(unicode('0004') in f.keys())
        assert(unicode('0005') in f.keys())

        assert(f['0005'].keys() == [unicode('20130524')])
        assert(f['0005'].attrs['mouse_type'] == 'GC5')
        assert(f['0005']['20130524']['homecagenovel'].keys() == 
                        [unicode('event_tuples'), unicode('time_series_arr')])

        assert(f['0003'].keys() == [unicode('20130523')])
        assert(f['0003'].attrs['mouse_type'] == 'GC5_NAcprojection')
        assert(f['0003']['20130523']['homecagenovel'].keys() == 
                        [unicode('event_tuples'), unicode('time_series_arr')])

        assert(f['0001'].keys() == [unicode('20130524')])
        assert(f['0001']['20130524'].keys() == [unicode('homecagenovel'), unicode('homecagesocial')])
        assert(f['0001']['20130524']['homecagenovel'].keys() == 
                        [unicode('event_tuples'), unicode('time_series_arr')])
        assert(f['0001']['20130524']['homecagesocial'].keys() == [unicode('event_tuples'), unicode('time_series_arr')])
        assert(f['0001'].attrs['mouse_type'] == 'GC5')

        event_tuple_diffs = np.abs(np.array(f['0001']['20130524']['homecagesocial']['event_tuples']) - np.array([[49.8, 50.7], [100.0, 101.5]]))
        assert((event_tuple_diffs.all() < self.epsilon) == True)

        time_series_arr = np.array(f['0001']['20130524']['homecagesocial']['time_series_arr'])
        print "shape time_series_arr", time_series_arr, np.shape(time_series_arr)
        assert(np.shape(time_series_arr) == (37501, 3))
        assert(np.abs(np.max(time_series_arr[:,0]) - 150.0) < self.epsilon) #max time of simulated data
        assert(np.abs(np.max(time_series_arr[:,1]) - 3.0) < self.epsilon) #trigger data after processing in fiber_record_analyze
        assert(np.abs(np.max(time_series_arr[:,2]) - 8.67676616) < 0.1) #max fluorescence of simulated data (changes slightly with each run of generate_test_data)

        time_series_arr = np.array(f['0001']['20130524']['homecagenovel']['time_series_arr'])
        assert(np.shape(time_series_arr) == (37501, 3))
        assert(np.abs(np.max(time_series_arr[:,2]) - 5.44846108) < 0.1) #max fluorescence of simulated data (changes slightly with each run)

        time_series_arr = np.array(f['0003']['20130523']['homecagenovel']['time_series_arr'])
        assert(np.shape(time_series_arr) == (37501, 3))

        time_series_arr = np.array(f['0005']['20130524']['homecagesocial']['time_series_arr'])
        print "shape time_series_arr", time_series_arr, np.shape(time_series_arr)
        assert(np.shape(time_series_arr) == (37501, 3))
        assert(np.abs(np.max(time_series_arr[:,0]) - 150.0) < self.epsilon) #max time of simulated data
        assert(np.abs(np.max(time_series_arr[:,1]) - 3.0) < self.epsilon) #trigger data after processing in fiber_record_analyze
        assert(np.abs(np.max(time_series_arr[:,2]) - 2.0) < 0.1)
     

class Test_add_flattened_files_to_hdf5():
    def setup(self):
        self.epsilon = 0.000001 #used to check equality of two floats

        cfg = load_configuration()
        self.filenames_file = str(cfg['analysis_filenames_file'])
        self.path_to_raw_data = str(cfg['path_to_raw_data'])
        self.path_to_hdf5 = str(cfg['path_to_hdf5'])
        self.path_to_flat_data = str(cfg['path_to_flat_data'])

        self.analysis_filenames = prp.read_filenames(self.filenames_file, self.path_to_raw_data)
        if os.path.isfile(self.path_to_hdf5):
            os.remove(self.path_to_hdf5)

        prp.generate_hdf5_file(self.analysis_filenames,self.path_to_hdf5)

    def tearDown(self):
        #self.f.close()
        os.remove(self.path_to_hdf5)

    def test_add_flat_files(self):
#        flat_directories = str(self.path_to_flat_data)
        prp.add_flattened_files_to_hdf5(self.path_to_flat_data, self.path_to_hdf5)

        self.f = h5py.File(self.path_to_hdf5)
        f = self.f
        assert(unicode('0001') in f.keys())
        assert(unicode('0002') in f.keys())
        assert(unicode('0003') in f.keys())
        assert(unicode('0004') in f.keys())

        assert(f['0001'].keys() == [unicode('20130524')])
        assert(f['0001'].attrs['mouse_type'] == 'GC5')

        assert(f['0003'].keys() == [unicode('20130523')])
        assert(f['0003'].attrs['mouse_type'] == 'GC5_NAcprojection')

        assert(f['0003']['20130523']['homecagenovel'].keys() == 
                        [unicode('event_tuples'), unicode('flat'), unicode('time_series_arr')])
        flat_arr = np.array(f['0003']['20130523']['homecagenovel']['flat'])
        assert(np.shape(flat_arr) == (37501, 1));

        assert(f['0002']['20130524']['homecagenovel'].keys() == 
                [unicode('event_tuples'), unicode('flat'), unicode('time_series_arr')])

        flat_arr = np.array(f['0001']['20130524']['homecagesocial']['flat'])
        assert(np.abs(np.max(flat_arr) - 8.686864357) < 0.1)



# class Test_debleach():
#     pass
    #TODO
    #TODO
    #TODO
    #Need to implement a more robust debleaching function





