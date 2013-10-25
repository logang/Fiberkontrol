import tkFileDialog
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


if __name__=='__main__':
    check_input()
    pre_video = time.time()
    print "Select video file:"
    video_file = tkFileDialog.askopenfilename()
#    subprocess.Popen(["open", "-g", sys.argv[1]])
    subprocess.Popen(["open", "-g", video_file])
    
    start_video_time = time.time()
    print start_video_time - pre_video

    start_social = np.zeros(0)
    end_social = np.zeros(0)
    placed_in_chamber = np.zeros(0)

    end_video_time = 0

    print "start_social=j   end_social=k   start_video=a   end_video=f   conspecific_placed_in_chamber=p"
    
    while (True):
#        s = raw_input('-->')
        getch = _GetchUnix()
        s = getch()
        print s

        if s == 'j':
            start_social = np.append(start_social, time.time() - start_video_time)
                
        if s == 'k':
            end_social = np.append(end_social, time.time() - start_video_time)

        if s == 'a':
            start_video_time = time.time()

        if s == 'f':
            end_video_time = time.time() - start_video_time
            break

        if s == 'p':
            placed_in_chamber = np.append(placed_in_chamber, time.time() - start_video_time)
            
    save_path = '/Users/kellyz/Documents/Data/Fiberkontrol/'
#    filename = sys.argv[2]

#    full_save_path = save_path + filename
    full_save_path = video_file[:-4]
    print full_save_path

    np.savez(full_save_path + '_s', start_social)
    np.savez(full_save_path + '_e', end_social)
    np.savez(full_save_path + '_p', placed_in_chamber)
    np.savez(full_save_path + '_a', start_video_time)
    print start_social
    print end_social
    print placed_in_chamber
    print "the end"

