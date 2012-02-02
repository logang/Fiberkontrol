import numpy as np
import os
import sys

import data_classes as dc
import unpack_data




def ConvertSecondsToDataPoints(tStartIndex, seconds, t):
    t0 = t[tStartIndex]

    for i in range(len(t)):
        elapsedTime = t[i] - t0

        if elapsedTime >= seconds:
            return i

def FindTimeIndex(desired_t, t_data):
    for i in range(len(t_data)):
        if t_data[i] > desired_t:
            return i-1
    return len(t_data)

def FindNextOnIndex(currentIndex, s):
    if currentIndex >= len(s):
        print "Reached END of trial"
        return -1
    else:
        while s[currentIndex] == 0:
            currentIndex += 1
            if currentIndex >= len(s):
                return -1

    print "next On Index", currentIndex
    return currentIndex


#Split a time series into pulses __|--|__ with tOff seconds before and after, and tOn se\
#conds in the middle 
def SplitTrialIntoChunks(trial):
    print "SplitTrial", trial
    tOff = 9.9     # Set these according to the dataset
    tOn = 9.9

#    nOff = ConvertSecondsToDataPoints(tStart, tOff, trial.all_time_data)
#    nOn  = ConvertSecondsToDataPoints(tOn, trial.all_time_data)

    currentIndex = 0

    while True:
        onIndex = FindNextOnIndex(currentIndex, trial.all_shutter_data)
        print "onTime", trial.all_time_data[onIndex]
        if onIndex == -1:
            break

#       nOff = ConvertSecondsToDataPoints(tStartIndex, tOff, trial.all_time_data)
#       nOn  = ConvertSecondsToDataPoints(tOn, trial.all_time_data)
#        pulseStart = onIndex - nOff
#        pulseEnd = onIndex + nOn + nOff


        pulseStartTime = trial.all_time_data[onIndex] - tOff
        pulseStart = FindTimeIndex(pulseStartTime, trial.all_time_data)
        pulseEndTime = trial.all_time_data[onIndex] + tOn + tOff
        pulseEnd = FindTimeIndex(pulseEndTime, trial.all_time_data)
        print "pulseEnd", pulseEnd
        print "pulseEndTime", pulseEndTime

        stimEndTime = trial.all_time_data[onIndex] + tOn
        stimEnd = FindTimeIndex(stimEndTime, trial.all_time_data)
        

        if(pulseEnd >= len(trial.all_y_data)):
            pulseEnd = len(trial.all_y_data)-1

        # Make a StimChunk

        chunk = dc.StimChunk()
        print "Trial to append to", trial
        print "Len(trial.list_of_StimChunks append", len(trial.list_of_StimChunks)
        trial.list_of_StimChunks.append(chunk)

        if (len(trial.intensity_list) > trial.chunk_counter):
            chunk.intensity  = trial.intensity_list[trial.chunk_counter]
        else:
            chunk.intensity  = trial.intensity_list[0]

        print "trial.intensity", trial.intensity_list

        if (len(trial.frequency_list) > trial.chunk_counter):
            chunk.frequency = trial.frequency_list[trial.chunk_counter]
        else:
            chunk.frequency = trial.frequency_list[0]

        

        chunk.start_time        = trial.all_time_data[pulseStart]
        chunk.start_stim_time   = trial.all_time_data[onIndex]
        chunk.end_time          = trial.all_time_data[pulseEnd]
        chunk.position_in_trial = trial.chunk_counter
        
        ### ------ CHANGE THE TAG depending on the trial type ('stim', 'sugar', 'juvenile')
        chunk.tag          = 'stim'
        chunk.mouse_name   = trial.mouse_name
        chunk.trial_number = trial.trial_number
        chunk.day_of_trial = trial.day_of_trial


        chunk.chunk_data  = trial.all_y_data[pulseStart:pulseEnd]

        chunk.before_data = trial.all_y_data[pulseStart:onIndex - 1]


        chunk.during_data = trial.all_y_data[onIndex:stimEnd]
        chunk.after_data  = trial.all_y_data[stimEnd + 1: pulseEnd]
        

        chunk.chunk_shutter_data = trial.all_shutter_data[pulseStart:pulseEnd]
        chunk.chunk_time_data    = trial.all_time_data[pulseStart:pulseEnd]
        chunk.chunk_input1_data  = trial.all_input1_data[pulseStart:pulseEnd]
        chunk.chunk_input2_data  = trial.all_input2_data[pulseStart:pulseEnd]
        chunk.chunk_input3_data  = trial.all_input3_data[pulseStart:pulseEnd]

        if (chunk.position_in_trial > 1):
            chunk.prev_chunk = trial.list_of_StimChunks[chunk.position_in_trial - 1]
        chunk.trial = trial

        currentIndex = pulseEnd
        trial.chunk_counter += 1

    print "len(trial.list_of_StimChunks)", len(trial.list_of_StimChunks)
        




def ReadFileList(f, filenames, frequency_strings, intensity_strings):
    i = 0
    for line in f:
        if i%4 == 0:
            filenames.append(line[:-1])
        elif i%4 == 1:
            intensity_strings.append(line[:-1])
        elif i%4 == 2:
            frequency_strings.append(line[:-1])
        else:
            if (line[:-1] != '#'):
                print 'file format error'
        i += 1

    print 'frequency_strings', frequency_strings
    print 'intensity_strings', intensity_strings


def ReadData(list_of_files):
    f = open(list_of_files, 'r')

    filenames = []
    frequency_strings = []
    intensity_strings = []
    ReadFileList(f, filenames, frequency_strings, intensity_strings)
    
    all = dc.AllData()  # instantiate AllData class

    # Fill the data classes

    trialCounter = 0
    for file in filenames:
       all.list_of_trial_names.append(file[9:])

       directory = '/Users/kellyz/Documents/Data/Fiberkontrol/'
       dataFilename = directory + file + '.npz'
       sFilename = directory + file + '_s.npz'
       tFilename = directory + file + '_t.npz'
       i1Filename = directory + file + '_i1.npz'
       i2Filename = directory + file + '_i2.npz'
       i3Filename = directory + file + '_i3.npz'

       try:
            d = np.load(dataFilename)['x']
            s = np.load(sFilename)['x']
            t = np.load(tFilename)['x']

            d = unpack_data.unpackArduino(d)
            s = unpack_data.unpackArduino(s)
            t = unpack_data.unpackArduino(t)
            i1 = []
            i2 = []
            i3 = []

       except:
            d = np.load(dataFilename)['arr_0']
            s = np.load(sFilename)['arr_0']
            t = np.load(tFilename)['arr_0']
            i1 = np.load(i1Filename)['arr_0']
            i2 = np.load(i2Filename)['arr_0']
            i3 = np.load(i3Filename)['arr_0']

            d  = unpack_data.unpackLabjack(d)
            s  = unpack_data.unpackLabjack(s)
            t  = unpack_data.unpackLabjack(t)
            i1 = unpack_data.unpackLabjack(i1)
            i2 = unpack_data.unpackLabjack(i2)
            i3 = unpack_data.unpackLabjack(i3)

           
       # Initialize a new TrialData instance

       trial = dc.TrialData() 
       all.list_of_TrialDatas.append(trial)
       print all.list_of_TrialDatas


       ## remove padding 
       length_trial = 0
       for i in range(len(t)-2):
           length_trial = i
           if(t[i] == 0 and t[i+1] == 0):
               break
           
       min_length = min(len(s), len(d), length_trial)
       min_length = min_length - 2  # remove padding at the end
       print "min_length", min_length

       dUnFlat = d
       trial.all_unflat_y_data = dUnFlat[0:min_length-1]

       # Flatten data

       dFlat = []
       fitCurves = []
       unpack_data.FlattenData(dFlat, trial.all_unflat_y_data, t, fitCurves)
       d = dFlat[0]


       trial.all_y_data = d[0:min_length-1]
       trial.all_shutter_data = s[0:min_length-1]
       trial.all_time_data = t[0:min_length-1]
       if (len(i1)>0):
           trial.all_input1_data = i1[0:min_length-1]
       else:
           trial.all_input1_data = i1

       if (len(i2)>0):
           trial.all_input2_data = i2[0:min_length-1]
       else:
           trial.all_input2_data = i2

       if (len(i3)>0):
           trial.all_input3_data = i3[0:min_length-1]
       else:
           trial.all_input3_data = i3


       trial.frequency_list = unpack_data.ConvertStringToFloatList(
           frequency_strings[trialCounter])
       trial.intensity_list = unpack_data.ConvertStringToIntList(
           intensity_strings[trialCounter])
       trial.mouse_name = file[-7:-3]
       trial.trial_number = file[-2:]
       trial.day_of_trial = file[-16:-8]
       trial.file_name = file

       trial.list_of_StimChunks = []

       SplitTrialIntoChunks(trial)
       trialCounter += 1
       print "trialCounter", trialCounter


#    print all.list_of_trial_names
#    print all.list_of_TrialDatas

    return all

#if __name__=="__main__":

#    list_of_files = sys.argv[1]
#    allData = ReadData(list_of_files)

#    print allData.list_of_trial_names
    
#    print "     "

#    for trial in allData.list_of_TrialDatas:
#        print "TRIAL"
#        print trial.file_name
#        print trial.frequency_list
#        print trial.intensity_list

#        print trial.list_of_StimChunks

#        for chunk in trial.list_of_StimChunks:
            
#            print chunk.mouse_name
#            print chunk.position_in_trial

