import numpy as np
import os
import sys

class AllData:
    list_of_TrialDatas = []             # a list of type TrialData
    list_of_StimChunks = []             # a list of type StimChunk
    list_of_trial_names = []         


class TrialData:
    all_y_data = None           # an array representing the entire time series
    all_unflat_y_data = None    # before debleaching
    all_shutter_data = None
    all_time_data = None
    all_input1_data = None
    all_input2_data = None
    all_input3_data = None

    low_pass_y = None           # data filtered by a low pass filter
    cutoff = None               # the cutoff for the low pass filter

    frequency_list = None
    intensity_list = None

    mouse_name = None
    trial_number = None
    day_of_trial = None    
    file_name = None

    list_of_StimChunks = []
    chunk_counter = 0

class StimChunk:
    intensity = -1             # as on optical density
    frequency = -1             # in Hz
    start_time = -1            # in s
    start_stim_time = -1
    end_stim_time = -1
    end_time = -1              # in s
    position_in_trial = -1     # i.e. n in "this is the nth chunk in the trial
    tag = None                 # 'stim', 'sugar', 'juvenile', 'blow'
    mouse_name = None
    trial_number = None
    day_of_trial = None

    chunk_data = None          # an array representing the time series
    before_data = None         # before stim
    during_data = None         # during stim
    after_data = None          # after stim

    before_data_line = None
    during_data_line = None
    after_data_line = None
    full_data_line = None

    chunk_shutter_data = None
    chunk_time_data = None
    chunk_input1_data = None
    chunk_input2_data = None
    chunk_input3_data = None

    prev_chunk = None          # the StimChunk that came prior (in time)
                               ## remains None if position_in_trial == 1

    trial = None               # the trial of which this chunk is part of




  
    
    
