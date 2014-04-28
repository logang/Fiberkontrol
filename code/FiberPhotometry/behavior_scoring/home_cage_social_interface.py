import tkFileDialog
from Tkinter import *
import numpy as np
import time
import sys
import subprocess

def check_input():
    if len(sys.argv) < 1:
        print "\nUsage: python home_cage_social_interface.py movie_filename save_filename\n"
        exit(0)


class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

def add_time(start_behavs, end_behavs, new_state, old_state,  t):
    """
    This function should be called after a key stroke indicating
    a behavior event. The current time is added to an array
    of start times for that behavior event, and if this new
    event comes directly after another behavior event, the
    current time is added to an array of end times for the 
    previous event.
    """

    if old_state != 'off':
        end_behavs[old_state] = np.append(end_behavs[old_state], t)

    if new_state != 'off':
        start_behavs[new_state] = np.append(start_behavs[new_state], t)


if __name__=='__main__':
    check_input()

    #first_LED = float(raw_input('-->Enter time of first LED: '))

    print "Select video file:"
    root = Tk()
    root.withdraw()
    video_file = tkFileDialog.askopenfilename()
    if video_file == '':
        print "Video_file is None"
        sys.exit(1)

    pre_video = time.time()
    p = subprocess.Popen(["open", "-a", "vlc", "-g", video_file])
    #p = subprocess.Popen(["open", "-g", video_file])
    

    press_play_time = time.time()
    
    # print "pre_video", pre_video
    # print "play", press_play_time - pre_video

    start_behavs = {'1':np.zeros(0), 
                        '2': np.zeros(0),
                        '3': np.zeros(0)}
    end_behavs = {'1': np.zeros(0),
                      '2': np.zeros(0),
                      '3': np.zeros(0)}

    placed_in_chamber = np.zeros(0)

    end_video_time = 0

    print "start_behavior_1=h   start_behavior_2=j"
    print "start_behavior_3=k   end_any_behavior=l"
    print "first_LED=a   end_video=f   conspecific_placed_in_chamber=p"

    state = 'off'
    while (True):
        getch = _GetchUnix()
        s = getch()
        print s

        if s == 'h':
            old_state = state
            state = '1'
            add_time(start_behavs, end_behavs, state, old_state,  
                        time.time() - start_video_time)

        if s == 'j':
            old_state = state
            state = '2'
            add_time(start_behavs, end_behavs, state, old_state,
                        time.time() - start_video_time)

                
        if s == 'k':
            old_state = state
            state = '3'
            add_time(start_behavs, end_behavs, state, old_state,
                        time.time() - start_video_time)

        if s == 'l':
            old_state = state
            state = 'off'
            add_time(start_behavs, end_behavs, state, old_state,
                        time.time() - start_video_time)

        if s == 'a':
            start_video_time = time.time()
            #led_on_time = 


        if s == 'f':
            end_video_time = time.time() - start_video_time
            p.kill()

            break

        if s == 'p':
            placed_in_chamber = np.append(placed_in_chamber, time.time() - start_video_time)
            

    full_save_path = video_file[:-4]
    print full_save_path

    np.savez(full_save_path + '_1s', start_behavs['1'])
    np.savez(full_save_path + '_1e', end_behavs['1'])
    np.savez(full_save_path + '_2s', start_behavs['2'])
    np.savez(full_save_path + '_2e', end_behavs['2'])
    np.savez(full_save_path + '_3s', start_behavs['3'])
    np.savez(full_save_path + '_3e', end_behavs['3'])
    np.savez(full_save_path + '_p', placed_in_chamber)
    np.savez(full_save_path + '_first_LED', start_video_time)

    print 'start_behavs[1]', start_behavs['1']
    print 'end_behavs[1]', end_behavs['1']

    print 'start_behavs[2]', start_behavs['2']
    print 'end_behavs[2]', end_behavs['2']

    print 'start_behavs[3]', start_behavs['3']
    print 'end_behavs[3]', end_behavs['3']

    print 'placed_in_chamber', placed_in_chamber
    print 'start_video_time', start_video_time

    print "the end"
    print "play", press_play_time - pre_video


