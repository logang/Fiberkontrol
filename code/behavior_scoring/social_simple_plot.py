import tkFileDialog
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


#def load_fluor(directory, fluor_file):
def load_fluor(fluorFilename):
#    fluorFilename = directory + fluor_file
    d = np.load(fluorFilename)
    data = d['data']
    fluor = data[:,0]
    t = d['time_stamps']
    licks = data[:,3]

    len = min(np.size(fluor), np.size(t))
    t = t[0:len]
    fluor = fluor[0:len]
    licks = licks[0:len]

    return [fluor, t, licks]

#def load_social(directory, social_file, t):
def load_social(timestamp_file, t):
#    start_socialFilename = directory + social_file + '_s.npz'
#    end_socialFilename = directory + social_file + '_e.npz'
 
    start_socialFilename = timestamp_file[:-6] + '_s.npz'
    end_socialFilename = timestamp_file[:-6] + '_e.npz'

    s = np.load(start_socialFilename)
    s = s['arr_0']
    e = np.load(end_socialFilename)
    e = e['arr_0']

    print s
    print e

    si = 0
    ei = 0


    social_times = np.zeros(len(t))
    for i in range(len(t)):
        if t[i] < s[si] and t[i] < e[ei]:
            social_times[i] = 0
        if t[i] >= s[si] and t[i] < e[ei]:
            social_times[i] = 1
        if t[i] >= s[si] and t[i] >= e[ei]:
            social_times[i] = 0
            si += 1
            ei += 1
            if si >= len(s) or ei >= len(e):
                break

    return [social_times, s, e]

def check_input():
    if len(sys.argv) < 1:
        print "\nUsage: python social_simple_plot.py fluorescence_filename social_times_filename\n"
        exit(0)

if __name__ == '__main__':

    check_input()

    directory ='/Users/kellyz/Documents/Data/Fiberkontrol/'
#    [fluor, t, licks] = load_fluor(directory, sys.argv[1])

    print "select gcamp file:"
    gcamp_file = tkFileDialog.askopenfilename()
    [fluor, t, licks] = load_fluor(gcamp_file)


    print "select timestamp file (either s or e):"
    timestamp_file = tkFileDialog.askopenfilename()
#    [social_times, s, e] = load_social(directory, sys.argv[2], t)
    [social_times, s, e] = load_social(timestamp_file, t)

    plt.plot(t, fluor, color='b')
    plt.plot( t, 3*social_times, color='r')
    plt.show()

