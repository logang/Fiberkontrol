import numpy as np
import os
import sys

import matplotlib.pyplot as plt

import read_data as rd



def CalculateMedianPulse(chunk_list):
    medianPulse = []

    for i in range(len(chunk_list[0].chunk_data)):
        valuesAtTimePoint = []
        
        for j in range(len(chunk_list)):
            
            if i < len(chunk_list[j].chunk_data):
                valuesAtTimePoint.append(chunk_list[j].chunk_data[i])

        medianPulse.append(np.median(valuesAtTimePoint))

    return medianPulse
        
                                         




if __name__=="__main__":    
#    list_of_files = sys.argv[1]
    list_of_files = '/Users/kellyz/Documents/Data/Fiberkontrol/Test_Trials.txt'
    allData = rd.ReadData(list_of_files)

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
        fig = plt.figure()
        print "len(time)", len(trial.all_time_data)
        print "len(y)", len(trial.all_y_data)
        print "len(s)", len(trial.all_shutter_data)
        print "len(i1)", len(trial.all_input1_data)

#        plt.plot(trial.all_time_data, trial.all_y_data)
#        plt.plot(trial.all_time_data, trial.all_unflat_y_data)


        
        for chunk in trial.list_of_StimChunks:
            if len(chunk.chunk_data) > 0:
#                fig = plt.figure()
#                print chunk
#                plt.plot(np.array(chunk.chunk_time_data), np.array(chunk.chunk_data))
#                plt.plot(chunk.chunk_time_data, np.array(chunk.chunk_shutter_data)*chunk.chunk_data[1]/chunk.frequency)
#                print "chunk.chunk_data[1]", chunk.chunk_data[1]
#                print "chunk.frequency", chunk.frequency

                freq = int(chunk.frequency*10)
                print "freq", freq
                frequency_chunk_list[freq].append(chunk)

                intensity = chunk.intensity
                intensity_chunk_list[intensity].append(chunk)
                

    print 'frequency_chunk_list', frequency_chunk_list
    print 'intensity_chunk_list', intensity_chunk_list


    for f in range(len(frequency_chunk_list)):
        chunk_list = frequency_chunk_list[f]
        print 'chunk_list', chunk_list

        if (chunk_list != []):
            fig = plt.figure()
            for chunk in chunk_list:
                plt.plot(np.array(chunk.chunk_time_data) - chunk.chunk_time_data[0], np.array(chunk.chunk_data - chunk.chunk_data[100]), alpha=0.3, color='0.75')
            medianPulse = CalculateMedianPulse(chunk_list)
        
            firstChunk = chunk_list[0]
            timeData = np.array(firstChunk.chunk_time_data) - firstChunk.chunk_time_data[0]
            minLen = min(len(timeData), len(medianPulse))
            plt.plot(timeData[0:minLen], np.array(medianPulse[0:minLen]) - medianPulse[0], linewidth=5, color='k')

    plt.show()
