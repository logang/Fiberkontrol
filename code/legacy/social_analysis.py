import numpy as np
import scipy as sp
import os
import sys

import matplotlib.pyplot as plt

import read_data as rd
import fit_data as fd
import data_classes as dc

from scipy.optimize import leastsq


def CalculateMedianPulse2(chunk_list):
    medianPulse = []

    for i in range(len(chunk_list[0].chunk_data)):
        valuesAtTimePoint = []
        
        for j in range(len(chunk_list)):
            
            if i < len(chunk_list[j].chunk_data):
                valuesAtTimePoint.append(chunk_list[j].chunk_data[i])

        medianPulse.append(np.median(valuesAtTimePoint))
    
    return medianPulse

def CalculateMedianPulse3(trial_list):
    medianPulse = []

    for i in range(len(trial_list[0].low_pass_y)):
        valuesAtTimePoint = []
        
        for j in range(len(trial_list)):
            
            if i < len(trial_list[j].low_pass_y):
                valuesAtTimePoint.append(trial_list[j].low_pass_y[i])

        medianPulse.append(np.median(valuesAtTimePoint))
    
    return medianPulse

def CalculateMedianPulse(chunk_list):
    medianPulse = []

    for i in range(len(chunk_list[0].chunk_data)):
        valuesAtTimePoint = []
        
        for j in range(len(chunk_list)):
            
            if i < len(chunk_list[j].chunk_data):
                valuesAtTimePoint.append(chunk_list[j].chunk_data[i])

        medianPulse.append(np.median(valuesAtTimePoint))

    median_chunk = dc.StimChunk()
    example_chunk = chunk_list[0]
    median_chunk.start_time = 0
    median_chunk.start_stim_time = example_chunk.start_stim_time - example_chunk.start_time
    median_chunk.end_stim_time = example_chunk.end_stim_time - example_chunk.start_time
    median_chunk.end_time = example_chunk.end_time - example_chunk.start_time

    median_chunk.chunk_time_data = np.array(example_chunk.chunk_time_data) - example_chunk.start_time
    median_chunk.chunk_time_data = median_chunk.chunk_time_data.tolist()

    median_chunk.position_in_trial = 1
    median_chunk.mouse_name = 'none'
    median_chunk.trial_number = 'none'
    median_chunk.day_of_trial = 'none'

    median_chunk.chunk_data = medianPulse
    start_index = rd.FindTimeIndex(median_chunk.start_time, median_chunk.chunk_time_data)
    start_stim_index = rd.FindTimeIndex(median_chunk.start_stim_time, median_chunk.chunk_time_data)
    end_stim_index = rd.FindTimeIndex(median_chunk.end_stim_time, median_chunk.chunk_time_data)
    end_index = rd.FindTimeIndex(median_chunk.start_stim_time, median_chunk.chunk_time_data)

    median_chunk.before_data = median_chunk.chunk_data[start_index:start_stim_index-1]
    median_chunk.during_data = median_chunk.chunk_data[start_stim_index:end_stim_index - 1]
    median_chunk.after_data = median_chunk.chunk_data[end_stim_index:end_index]

    median_chunk.chunk_shutter_data = example_chunk.chunk_shutter_data
    median_chunk.chunk_input1_data = example_chunk.chunk_input1_data
    median_chunk.chunk_input2_data = example_chunk.chunk_input2_data
    median_chunk.chunk_input3_data = example_chunk.chunk_input3_data

    median_chunk.prev_chunk = 'none'

#    return medianPulse
    return median_chunk
                                         

## Plots all of the chunks in the given list overlaid in light gray, plots the median chunk on top
def PlotChunkOverlay(given_chunk_list):

    for f in range(len(given_chunk_list)):
        chunk_list = given_chunk_list[f]
#        print 'chunk_list', chunk_list

        if (chunk_list != []):
            fig = plt.figure()
            for chunk in chunk_list:
                plt.plot(np.array(chunk.chunk_time_data) - chunk.chunk_time_data[0], np.array(chunk.chunk_data - chunk.chunk_data[100]), alpha=0.3, color='0.75')
            median_chunk= CalculateMedianPulse(chunk_list)
            medianPulse = median_chunk.chunk_data
                
#            medianPulse = CalculateMedianPulse(chunk_list)
                

            firstChunk = chunk_list[0]
            timeData = np.array(firstChunk.chunk_time_data) - firstChunk.chunk_time_data[0]
            minLen = min(len(timeData), len(medianPulse))
            plt.plot(timeData[0:minLen], np.array(medianPulse[0:minLen]) - medianPulse[0], linewidth=5, color='k')
            plt.title(str(f/10))
#             FitLineToChunk(median_chunk)

def LowPassFilter(trial, cutoff):
    data = trial.all_y_data
    rawsignal = np.array(data)
    fft = sp.fft(rawsignal)

    fig = plt.figure()
    plt.plot(np.abs(fft[5:200]))

    bp = fft[:]
    for i in range(len(bp)):
        if i>= cutoff: bp[i] = 0
    ibp = sp.ifft(bp)


    print 'ibp[0]', ibp[0]
    print 'data[0]', data[0]
    trial.low_pass_y = np.abs(ibp) - np.median(np.abs(ibp))
    trial.low_pass_y = ibp*np.abs(data[100])/np.abs(ibp[100])
    
    
    trial.cutoff = cutoff

#also look up autoregression model

def FitLineToChunk(chunk):
    fp = lambda a, x: a[0]*x + a[1]
    error = lambda a, x, y: (fp(a, x) - y)


    start_index = rd.FindTimeIndex(chunk.start_time, chunk.chunk_time_data)
    start_stim_index = rd.FindTimeIndex(chunk.start_stim_time, chunk.chunk_time_data)
    end_stim_index = rd.FindTimeIndex(chunk.end_stim_time, chunk.chunk_time_data)
    end_index = rd.FindTimeIndex(chunk.start_stim_time, chunk.chunk_time_data)

    y = np.array(chunk.before_data)
    x = np.arange(len(chunk.before_data))
    a0 = np.array([0, 0])
    a, success = leastsq(error, a0, args=(x, y), maxfev=100)
    yfit = fp(a, x)
    chunk.before_data_line = yfit.tolist()

    y = np.array(chunk.during_data)
    x = np.arange(len(chunk.during_data))
    a0 = np.array([0, 0])
    a, success = leastsq(error, a0, args=(x, y), maxfev=100)
    yfit = fp(a, x)
    chunk.during_data_line = yfit.tolist()

    y = np.array(chunk.after_data)
    x = np.arange(len(chunk.after_data))
    a0 = np.array([0, 0])
    a, success = leastsq(error, a0, args=(x, y), maxfev=100)
    yfit = fp(a, x)
    chunk.after_data_line = yfit.tolist()

    data_line = chunk.before_data_line
    data_line.extend(chunk.during_data_line)
    data_line.extend(chunk.after_data_line)

    fig = plt.figure()
#    plt.plot(chunk.chunk_time_data[start_stim_index:end_stim_index], chunk.during_data_line)
#    plt.plot(chunk.chunk_time_data[start_stim_index:end_stim_index], chunk.during_data)

    chunk.full_data_line = data_line[0:len(chunk.chunk_time_data)-2]
    print 'len(chunk.chunk_time_data)', 
    print 'len(chunk.full_data_line)', 


    if len(chunk.chunk_time_data) > len(chunk.full_data_line):
        minLength = len(chunk.full_data_line)
    else:
        minLength = len(chunk.chunk_time_data)

    plt.plot(chunk.chunk_time_data[0:minLength], chunk.full_data_line[0:minLength])
    plt.plot(chunk.chunk_time_data[0:minLength], chunk.chunk_data[0:minLength])



def CompareEpochsInChunk(frequency_chunk_list):
    
    for f in range(len(frequency_chunk_list)):
        chunk_list = frequency_chunk_list[f]
        if chunk_list != []:
            before_medians = []
            during_medians = []
            after_medians = []
            
  #          print 'before_medians', before_medians

            prev_before_medians = []
            prev_during_medians = []
            prev_after_medians = []

            for chunk in chunk_list:
            
                if (chunk.prev_chunk != None):
                    if (f != 10):
 #                       print 'before_medians', before_medians
 #                       print 'np.median(np.array(chunk.before_data)))', np.median(np.array(chunk.before_data))

                        before_medians.append(np.median(np.array(chunk.before_data)))
                        during_medians.append(np.median(np.array(chunk.during_data)))
                        after_medians.append(np.median(np.array(chunk.after_data)))
                        

#                        print 'before_medians', before_medians


            
                        prev_before_medians.append(np.median(np.array(chunk.prev_chunk.before_data)))
                        prev_during_medians.append(np.median(np.array(chunk.prev_chunk.during_data)))
                        prev_after_medians.append(np.median(np.array(chunk.prev_chunk.after_data)))

            
            weightedSumOfMedians_before = 0
            weightedSumOfMedians_during = 0
            weightedSumOfMedians_after = 0
            if (f!= 10):
                for i in range(len(before_medians)):
                    weightedSumOfMedians_before += before_medians[i]*prev_during_medians[i]
                    weightedSumOfMedians_during += during_medians[i]*prev_during_medians[i]
                    weightedSumOfMedians_after += after_medians[i]*prev_during_medians[i]

                average_before = weightedSumOfMedians_before/sum(prev_during_medians)
                average_during = weightedSumOfMedians_during/sum(prev_during_medians)
                average_after = weightedSumOfMedians_after/sum(prev_during_medians)

#                print 'f:', f/10, 'before: ', average_before, 'during: ', average_during, 'after: ', average_after

#                print 'WEIGHTED before diff: ', average_during - average_before, 'after diff: ', average_during - average_after
            


 #               print 'f:', f/10, 'before: ', np.median(before_medians), 'during: ', np.median(during_medians), 'after: ', np.median(after_medians)

                sign_during = np.median(during_medians)/np.abs(np.median(during_medians))
  

                print 'f:' , f/10, 'NON_WEIGHTED. before diff: ', (np.median(during_medians)
                                                                  - np.median(before_medians)), 'after diff: ', (np.median(during_medians) - np.median(after_medians))
                
                    





if __name__=="__main__":    
#    list_of_files = sys.argv[1]
    list_of_files = '/Users/kellyz/Documents/Data/Fiberkontrol/Social_Trials.txt'
    allData = rd.ReadData(list_of_files)
    for trial in allData.list_of_TrialDatas:
        rd.SplitTrialIntoChunks(trial)

    MAX_FREQ = 1000  #Fill this list with the index = frequency*10
    frequency_chunk_list = []
    for i in range(MAX_FREQ):
        frequency_chunk_list.append([])

    MAX_INTENSITY = 10
    intensity_chunk_list = []
    for i in range(MAX_INTENSITY):
        intensity_chunk_list.append([])

    for trial in allData.list_of_TrialDatas:
        print "TRIAL"
        print trial
#        fig = plt.figure()
        print "len(time)", len(trial.all_time_data)
        print "len(y)", len(trial.all_y_data)
        print "len(s)", len(trial.all_shutter_data)
        print "len(i1)", len(trial.all_input1_data)


        fig = plt.figure()
        plt.plot(trial.all_time_data, trial.all_y_data)
        plt.plot(trial.all_time_data, trial.low_pass_y)

    fig = plt.figure()
    


    fig = plt.figure()
    for trial in allData.list_of_TrialDatas:
        startIndex = rd.FindTimeIndex(10, trial.all_time_data)
        print 'startIndex', startIndex
        endIndex = rd.FindTimeIndex(80, trial.all_time_data)
        print 'endIndex', endIndex

        print 'len(trial.all_time_data)', len(trial.all_time_data)

        normalized_y_data = np.array(trial.low_pass_y)/np.median(np.array(trial.low_pass_y[startIndex:endIndex]))

        print 'len(normalized_y_data)', len(normalized_y_data)
        plt.plot(np.array(trial.all_time_data[startIndex:endIndex]), 
                 np.array(trial.low_pass_y[startIndex:endIndex] - trial.low_pass_y[100]),  alpha=0.3, color='0.75')


    medianPulse = CalculateMedianPulse3(allData.list_of_TrialDatas)
    print 'len(medianPulse)', len(medianPulse)
    trial = allData.list_of_TrialDatas[0]
    startIndex = rd.FindTimeIndex(10, trial.all_time_data)
    print 'startIndex', startIndex
    endIndex = rd.FindTimeIndex(80, trial.all_time_data)
    print 'endIndex', endIndex
    
    plt.plot(np.array(trial.all_time_data[startIndex:endIndex]), 
             np.array(medianPulse[startIndex:endIndex] - medianPulse[100]), linewidth=5, color='k')


    fig = plt.figure()
    plt.plot(np.array(trial.all_time_data[startIndex:endIndex]), np.array(medianPulse[startIndex:endIndex]))



#        for chunk in trial.list_of_StimChunks:
#            if len(chunk.chunk_data) > 0:
            
#                freq = int(chunk.frequency*10)
#               print "freq", freq
#                frequency_chunk_list[freq].append(chunk)

#                intensity = chunk.intensity
#                intensity_chunk_list[intensity].append(chunk)
                

#                FitLineToChunk(chunk)
    
        
#        PlotChunkOverlay(frequency_chunk_list)

#    CompareEpochsInChunk(frequency_chunk_list)

    

    plt.show()
