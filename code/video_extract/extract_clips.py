import subprocess
import numpy as np
from optparse import OptionParser
import sys
import pickle
import os.path
import os
import shlex

def get_mouse_info(key):
    """
    Returns the mouse info
    from the a dict key of the format:
    409_20130327_homecagesocial
    """

    info = key.split('_')
    mouse_number = info[0]
    date = info[1]
    exp_type = info[2]
    if exp_type == 'homecagesocial':
        exp_type = 'social'
    elif exp_type == 'homecagenovel':
        exp_type = 'novel'

    return mouse_number, date, exp_type

## Get list of times and fluor values for each movie
def get_movie_file(key, path_to_data):
    """
    Given a key of the format: 409_20130327_homecagesocial
    and a path to the folder containing the folders for 
    each day of data (i.e. 20130327), returns a full
    path to the .avi file containing the movie for that trial
    (i.e. the homecagesocial trial for mouse 409 on day 20130327)
    """
    mouse_number, date, exp_type = get_mouse_info(key)
    movie_file = path_to_data+date+'/'+mouse_number+'_'+exp_type
    return movie_file

def get_output_file(key, path_to_data, output_dir):
    mouse_number, date, exp_type = get_mouse_info(key)

#    output_dir = path_to_data+date+'/'+output_folder
    check_dir(output_dir)
    output_file = output_dir+'/'+mouse_number+'_'+exp_type
    return output_file


def load_start_times(start_times_file):
    all_start_times_dict = dict()
    f = open(start_times_file, 'r')
    for line in f:
        info = line.split(',')
        all_start_times_dict[info[0]] = float(info[1])
    f.close()
    return all_start_times_dict


def load_clip_times(clip_list_file, path_to_data, start_times_file, output_dir):
    """
    Loads the three dictionaries all_peak_times_dict, all_peak_vals_dict, 
    and all_local_times_dict from the provided .pkl clip_list_file, which
    was produced by print_spike_times in group_analysis.py.
    The key of each dict is a name of the format: 409_20130327_homecagesocial
    and the number of entries for each key is variable.

    Then, makes a new dict with the format:
    dict{409_20130327_homecagesocial: 
            dict{movie_file: path to movie file (will check for avi or mp4), 
                 peak_times: array with times of each peak,
                 peak_vals: array with vals of each peak,
                 local_times: array with local times of each peak
                }
        }
    """
    print "clip_list_file", clip_list_file
    pkl_file = open(clip_list_file, 'rb')
    all_peak_times_dict = pickle.load(pkl_file)
    all_peak_vals_dict = pickle.load(pkl_file)
    all_local_times_dict = pickle.load(pkl_file)
    all_interaction_start_times_dict = pickle.load(pkl_file)
    all_interaction_end_times_dict = pickle.load(pkl_file)
    all_start_times_dict = load_start_times(start_times_file)

    movie_info_dict = dict()
    for key in all_peak_times_dict.keys():
        movie_info = dict()
        movie_info['movie_file'] = get_movie_file(key, path_to_data)
        movie_info['output_file'] = get_output_file(key, path_to_data, output_dir)
        movie_info['peak_times'] = all_peak_times_dict[key]
        movie_info['peak_vals'] = all_peak_vals_dict[key]
        movie_info['local_times'] = all_local_times_dict[key]
        movie_info['name'] = key
        movie_info['start_time'] = all_start_times_dict[key]
        movie_info['interaction_start'] = all_interaction_start_times_dict[key]
        movie_info['interaction_end'] = all_interaction_end_times_dict[key]

        movie_info_dict[key] = movie_info
    return movie_info_dict

def get_time_string(t):
    start = t
    hr = int(np.floor(start/3600.0))
    min = int(np.floor((start - hr*3600)/60.0))
    sec = int(start - hr*3600 - min*60)
    return str(hr)+':'+str(min)+':'+str(sec)

def splice_clips(list_file, output_file):
    """
    Calling ffmpeg in the command line,
    joins the clips (in order) listed in 'list_file', 
    a text file with the format for each line: 
    file file1.mp4
    file file2.mp4
    """
    cmd = ['ffmpeg']
    cmd += ['-f', 'concat']
    cmd += ['-i', list_file]
    cmd += ['-c', 'copy']
    cmd += ['-y']
    cmd += [output_file + '_clips.mp4']
    #cmd += ['> /dev/null 2>&1 < /dev/null'] 

    cmd_string = ''.join(["%s " % el for el in cmd])
    #print '-->Running: ', cmd_string
    p = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()

def delete_clips(clip_list_arr):
    """
    Deletes the clips listed in provided array
    (specified by full path/filename.mp4)
    The example use of this deleting all of the
    clips that were generated for the sake
    of splicing together into a larger video.
    """

    for clip in clip_list_arr:
        os.remove(clip)

def write_list_file(output_file, clip_list_arr):
    """
    Write all clip filenames to a text file for use
    by ffmpeg concat (i.e. by the function splice_clips()).
    """
    list_file = output_file+'_clip_list.txt'
    print "list_file: ", list_file
    f = open(list_file, 'w')
    for clip in clip_list_arr:
        line = 'file '+clip
        f.write("%s\n" % line)
        # Add in a divider movie between clips? (it could go here)
    f.close()
    # print 'list_file', list_file
    # print clip_list_arr

    return list_file

def check_video_timestamps(movie_file, desired_format='.mp4', desired_framerate=30):
    """
    Check whether frame rate is 30fps (the original video files should
    be 10fps). If not, convert framerate, and add timecode stamps
    to the videos.
    """

    check_video_format(movie_file, desired_format='.mp4', original_format='.avi')

    new_movie_file = movie_file+'_tt'+desired_format
    if not os.path.isfile(new_movie_file):
        #Convert file to 30 fps
        cmd = ['ffmpeg', '-i', movie_file+desired_format]
        cmd += ['-r', str(desired_framerate)]
        cmd += ['-y', movie_file+'_t'+desired_format]
        cmd_string = ''.join(["%s " % el for el in cmd])  
        #print '-->Running: ', cmd_string
        p = subprocess.Popen(cmd, shell=False)
        p.wait()

        #Add timecode text to video
        cmd = 'ffmpeg -i '+movie_file+'_t'+desired_format+' -vf drawtext=\"fontfile=/opt/X11/share/fonts/TTF/VeraMoBd.ttf: timecode=\'00\:00\:00\:00\':rate=30: fontcolor=white@0.8: x=7: y=460\" -an -y '+movie_file+'_tt'+desired_format
        args = shlex.split(cmd)
        #print args
        p = subprocess.Popen(args, shell=False)
        p.wait()

        os.remove(movie_file+'_t'+desired_format)

    return new_movie_file


def  check_video_format(movie_file, desired_format='.mp4', original_format='.avi'):
    """
    The original (data) video should be in .avi format. But 
    we may need the video to be in a different format (i.e. mp4)
    for proper processing. This function checks whether the desired_format
    video exists. If not, then it converts (copies) the original_format
    to the desired_format.
    """

    if not os.path.isfile(movie_file+original_format):
        print 'Error. avi file does not exist:'+movie_file+'.avi'
    if not os.path.isfile(movie_file+desired_format):
        cmd = ['ffmpeg']
        cmd += ['-i', movie_file+original_format]
        cmd += [movie_file+desired_format]
        cmd_string = ''.join(["%s " % el for el in cmd])
        #print '-->Running: ', cmd_string
        p = subprocess.Popen(cmd, shell=False)
        p.wait()



def cut_into_clips(movie_info, 
                   peak_thresh, 
                   clip_window, 
                   clip_window_origin, 
                   output_file, 
                   draw_box=True):
    """
    Given movie_info (which contains a 'movie_file', a 
    list of times at which to cut clips ('peak_times'),
    values to judge the quality of the clip against
    a threshold ('peak_vals', vs. 'peak_thresh'), and a 
    'start_time' which indicates the length in seconds
    of buffer at the beginning of the video before 
    the LED flashes to indicate the start of fluorescence
    recording and which must be added to the listed times 
    in 'peak_times', cuts clips and saves them using the output_file
    template (i.e. path/movie_name).

    Returns an array listing the file paths to each
    saved clip.
    These clips should only be temporary, and can be
    deleted with the function delete_clips().

    Set clip_window = [0, 0] to use clips of the entire
    behavior interactions (as opposed to a subclip around
    fluorescence peaks).

    Set clip_window_origin = 'peak', 'interaction_start'
    to choose the origin of the clip window (either at
    the time of the peak fluorescence in an interaction, 
    or the start of the interaction). That is, the
    clip_window [a, b] is a window of 'a' seconds before
    clip_window_origin and 'b' seconds  after
    clip_window_origin.
    """
    movie_file = movie_info['movie_file'] + '_tt'
    peak_times = movie_info['peak_times']
    peak_vals = movie_info['peak_vals']
    start_time = movie_info['start_time']
    interaction_start_times = movie_info['interaction_start']
    interaction_end_times = movie_info['interaction_end']

    if clip_window[0] == 0 and clip_window[1] == 0:
        clip_all_interactions = True
    else:
        clip_all_interactions = False

    if clip_window_origin == 'interaction_start':
        start_clip_times = interaction_start_times
    elif clip_window_origin == 'peak':
        start_clip_times = peak_times

    clip_list_arr = []
    ## Now cut clips (do this the lazy way for now,
    ## saving to individual files, figure out pipes
    ## after you get this working). 
    for i in range(len(start_clip_times)):
        t = start_clip_times[i] + start_time - clip_window[0]
        v = peak_vals[i]
        #start = get_time_string(t)
        start = str(t)
        if clip_all_interactions:
            duration = str(interaction_end_times[i] - start_clip_times[i])
        else:
            duration = str(clip_window[0] + clip_window[1])
        print 'start', start, 'duration', duration

        new_file = output_file+'_'+str(int(t*10))+'_clip.mp4'
        if v<peak_thresh:
            print "PEAK LESS THAN THRESHOLD: ", v, "thresh: ", peak_thresh
        if v>=peak_thresh or clip_all_interactions:
            if len(clip_list_arr) == 0 or clip_list_arr[-1] != new_file:
                clip_list_arr.append(new_file)

                cmd = ['ffmpeg']
                cmd += ['-i', movie_file+'.mp4']
                cmd += ['-ss', start]
                cmd += ['-t', duration]
                if draw_box:
                    cmd += ['-vf']
                    cmd += ['drawbox=0:0:100:100:red'+':t=1']
                    cmd += ['-vf']
                    #cmd += ['drawbox=0:0:100:100:red:t='+str(min(100, int(v*1000)/5.0))]
                    cmd += ['drawbox=640:0:100:100:red:t='+str(min(100, int(v*1000)/5.0))]
                cmd += ['-y']
                cmd += [new_file]
                #cmd += ['> /dev/null 2>&1 < /dev/null'] #not 100% sure what this does
                                                        # it is supposed to not send
                                                        # output to the PIPE buffer

                cmd_string = ''.join(["%s " % el for el in cmd])
                print '-->Running: ', cmd_string
                p = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                p.wait()

    return clip_list_arr


def interleave_lists(before, after):
    """
    Interleave two arrays of equal size, 
    'before' and 'after', yielding an output 
    array of twice the size, with entries
    alternating from the the before and 
    after array.
    """
    if len(before) != len(after):
        print "Error: arrays must of same length in interleave_lists"
        return None
    else:
        output = before + after
        output[::2] = before
        output[1::2] = after
    return output

    
def generate_clips(movie_info, clip_window, clip_window_origin,
                   peak_thresh, divider_clip):
    '''
    A wrapper function for cut_into_clips() which
    handles edge cases in the format of clip_window.
    In particular if clip_window=[0,0], then each clip
    corresponds to an entire interaction epoch.
    If clip_window=[T, 0], then each clip corresponds
    to an entire interaction epoch preceded by
    a clip of time T.

    Outputs:
    'clip_list_arr' is the list of clips to be spliced together.
    'clips_to_delete' lists all clips except the divider_clip file.
    '''

    movie_file = movie_info['movie_file']
    output_file = movie_info['output_file']

    if clip_window[0] != 0 and clip_window[1] != 0:
        before_window = [clip_window[0], 0]
        before_clip_list_arr = cut_into_clips(movie_info, peak_thresh, before_window, 
                                            clip_window_origin, output_file, draw_box=False)
        after_window = [0, clip_window[1]]
        after_clip_list_arr = cut_into_clips(movie_info, peak_thresh, after_window, 
                                             clip_window_origin, output_file, draw_box=True)
        clip_list_arr = interleave_lists(before_clip_list_arr, after_clip_list_arr)

    elif clip_window[0] != 0 and clip_window[1] == 0:
        before_window = [clip_window[0], 0]
        before_clip_list_arr = cut_into_clips(movie_info, peak_thresh, before_window, 
                                              clip_window_origin, output_file, draw_box=False)
        after_window = [0, 0]
        after_clip_list_arr = cut_into_clips(movie_info, peak_thresh, after_window, 
                                             clip_window_origin, output_file, draw_box=True)
        clip_list_arr = interleave_lists(before_clip_list_arr, after_clip_list_arr)
    else:
        clip_list_arr = cut_into_clips(movie_info, peak_thresh, clip_window, 'peak', output_file)

    clips_to_delete = clip_list_arr

    if divider_clip is not None:
        clip_list_arr = interleave_lists(clip_list_arr, 
                                        [divider_clip]*len(clip_list_arr))

    return clip_list_arr, clips_to_delete



def cut_and_splice_clips(movie_info, 
                         clip_window, 
                         clip_window_origin,
                         peak_thresh=0.00, 
                         divider_clip=None):
    """
    Given a dict containing the 'movie_file', 'peak_times', 
    'peak_vals', 'start_time', and 'name', splice a movie clip 
    of clip_window[0] seconds before the peak_time and
    clip_window[1] seconds after the peak_time for all times
    in peak_times where peak_val > peak_thresh.
    If clip_window[1] == 0, then the clip will extend
    until the end of the behavior interaction period.
    """

    check_video_timestamps(movie_info['movie_file'], desired_format='.mp4', desired_framerate=30)

    clip_list_arr, clips_to_delete = generate_clips(movie_info, clip_window, 
                                     clip_window_origin, peak_thresh, divider_clip)

    output_file = movie_info['output_file']
    list_file = write_list_file(output_file, clip_list_arr)
    splice_clips(list_file, output_file)
    delete_clips(clips_to_delete)


def check_key(key, options):
    """
    Limit analysis if only specific animal_id,
    exp_date, or exp_type is input as a command
    line option.
    """
    animal_id, exp_date, exp_type = key.split('_')
    if ((options.animal_id is None or animal_id == options.animal_id)
            and (options.exp_date is None or exp_date == options.exp_date)
            and (options.exp_type is None or exp_type == options.exp_type)):
        return True
    else:
        return False

def make_time_series_animation(movie_info, 
                               animation_dir,
                               time_series_data_path,
                               mouse_type, 
                               format):
    """
    Call time_series_data_path() in group_analysis.py
    to create an animation video of the time series
    corresponding to the trial specified in movie_info.
    """
    animal_id, exp_date, exp_type = movie_info['name'].split('_')

    cmd = ['python', '../analysis/group_analysis.py']
    cmd += ['--input-path='+str(time_series_data_path)]
    cmd += ['--output-path='+str(animation_dir)]
    cmd += ['--time-series-animation']

    cmd += ['--mouse-type='+str(mouse_type)]
    cmd += ['--plot-format='+str(format)]
    cmd += ['--exp-type='+str(exp_type)]
    cmd += ['--animal-id='+str(animal_id)]
    cmd += ['--exp-date='+str(exp_date)]

    cmd_string = ''.join(["%s " % el for el in cmd])
    print '-->Running: ', cmd_string
    p = subprocess.Popen(cmd, shell=False)# stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()

def check_dir(direc):
    if not os.path.exists(direc):
        os.makedirs(direc)

def overlay_time_series(movie_info, time_series_data_path, output_dir, mouse_type):
    """
    Generate animation of time series using time_series_animation()
    in group_analysis.py.
    Then, overlay this animation on the full length behavior video,
    clipped to start at the time of the first LED flash (start_time).

    Saves animation to animation/ directory.
    Saves overlay to overlay/ directory.
    """
    if mouse_type is None:
        print 'Error: Must provide --mouse_type when using overlay_time_series.'
        sys.exit(1)

    #Place animation and overlay in two separate folders
    output_dir = '/'.join(output_dir.split('/')[:-1])
    overlay_dir = output_dir + '/Overlay/'
    animation_dir = output_dir + '/Time_Series_Animation/'
    check_dir(overlay_dir)
    check_dir(animation_dir)


    #Generate animation of time series
    format = '.mp4'
    animation_filename = animation_dir + key + format
    if not os.path.isfile(animation_filename):
        make_time_series_animation(movie_info, animation_dir, 
                                   time_series_data_path,
                                   mouse_type, format)
    else:
        print "Animation file already exists: ", key

    #Overlay animation in corner of behavior video
    movie_file_tt = check_video_timestamps(movie_info['movie_file'], desired_format='.mp4', desired_framerate=30)
    overlay_filename = overlay_dir + key + format
    start_time = movie_info['start_time']
    cmd = ['ffmpeg']
    cmd += ['-ss', start_time]
    cmd += ['-i', movie_file_tt]
    cmd += ['filter_complex']

    cmd = 'ffmpeg -ss '+str(start_time)+' -i '+movie_file_tt+' -i '+animation_filename+' -filter_complex \"[0:v] setpts=PTS-STARTPTS, scale=640x480 [background]; [1:v] setpts=PTS-STARTPTS, scale=100x200 [upperleft]; [background][upperleft] overlay=shortest=1\" -c:v libx264 '+overlay_filename
    args = shlex.split(cmd)
    print args
    p = subprocess.Popen(args, shell=False)
    p.wait()


# ## for adding overlay to trimmed video (trim to start time 10.47)
# ffmpeg -ss 10.47 -i 421_social_tt.mp4 -i 20121105_421_homecagesocial.mp4  -filter_complex "[0:v] setpts=PTS-STARTPTS, scale=640x480 [background]; [1:v] setpts=PTS-STARTPTS, scale=100x200 [upperleft]; [background][upperleft] overlay=shortest=1" -c:v libx264 overlay_output.mp4




        # movie_info['movie_file'] = get_movie_file(key, path_to_data)
        # movie_info['output_file'] = get_output_file(key, path_to_data, output_dir)
        # movie_info['peak_times'] = all_peak_times_dict[key]
        # movie_info['peak_vals'] = all_peak_vals_dict[key]
        # movie_info['local_times'] = all_local_times_dict[key]
        # movie_info['name'] = key
        # movie_info['start_time'] = all_start_times_dict[key]
        # movie_info['interaction_start'] = all_interaction_start_times_dict[key]
        # movie_info['interaction_end'] = all_interaction_end_times_dict[key]




#     plot_type=time-series-animation
# exp_type='homecagesocial'
# animal_id='421'
# exp_date='20121105'
# plot_format='.mp4'
# output_path='/Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Results/Cell/Videos'
# python group_analysis.py --input-path=$data_path --output-path=$output_path --$plot_type --mouse-type='GC5' --plot-format=$plot_format --exp-type=$exp_type --animal-id=$animal_id --exp-date=$exp_date







if __name__ == '__main__':
    ## The path_to_data is the folder containing folders corresponding to each trial date.
    ## These folders contain videos for each trial. 
    parser = OptionParser()
    parser.add_option("-o", "--output-dir", dest="output_dir", 
                      default='test', 
                      help=("Specify the full path of the output folder "
                            "to contain the output movies (no final backslash)."))

    parser.add_option("", "--clip-window", dest="clip_window",
                      default='0:1',
                      help="Specify a time window in which to display video clip"
                            " around given time points, in format before:after.")

    parser.add_option("", "--clip-window-origin", dest="clip_window_origin", 
                      default='peak',
                      help="If the clip_window is [a,b], clip_window_origin defines"
                           "the center of the clip_window. Options are 'peak', the time of "
                           "the peak fluorescence value in an interaction period, or "
                           "'interaction_start', the start time of the interaction period.")

    parser.add_option("", "--peak-thresh", dest="peak_thresh",
                      default=0.05,
                      help="Specify the threshold (in dF/F) that the signal"
                           "must cross to be considered a peak.")

    parser.add_option("", "--divider-clip", dest="divider_clip",
                      default=None,
                      help="Provide the path to a divider clip (such as a black frame .mp4)" 
                           "to place between each clip in the final compilation.")

    parser.add_option("", "--mouse-type", dest="mouse_type", 
                      default=None,
                      help="Specify the type of virus injected in the mouse (GC5, GC5_NAcprojection, GC3, EYFP)")

    parser.add_option("", "--animal-id", dest="animal_id", 
                      default=None,
                      help="Limit group analysis to trials of a specific animal.")

    parser.add_option("", "--exp-date", dest="exp_date", 
                      default=None,
                      help="Limit group analysis to trials of a specific date.")

    parser.add_option("", "--exp-type", dest="exp_type", 
                      default=None,
                      help="Limit group analysis to trials of a specific type.")
    (options, args) = parser.parse_args()

    print "options.clip_window", options.clip_window
    clip_window = np.array(options.clip_window.split(':'), dtype='float32') 
    output_dir = options.output_dir
    peak_thresh = float(options.peak_thresh)
    divider_clip = options.divider_clip
    clip_window_origin = options.clip_window_origin
    mouse_type = options.mouse_type

    # Check for required input path
    if len(args) < 4:
        print 'Usage: python extract_clips.py [video_data_path] [start_times_file] [time_series_data_path] [clip_list.pkl] -o [output]'
        print 'Where clip_list.pkl is generated by group_analysis.py and should '
        print ' contain three dicts:'
        print ' all_peak_times_dict, all_peak_vals_dict, all_local_times_dict,'
        print ' where the key of each dict is a name of the format: 409_20130327_homecagesocial .' 
        sys.exit(1)
    video_data_path = args[0]
    start_times_file = args[1]
    time_series_data_path = args[2]
    clip_list_file = args[3]

    movie_info_dict = load_clip_times(clip_list_file, video_data_path, start_times_file, output_dir)

    #print "KEYS: ", movie_info_dict.keys()
    for key in movie_info_dict:
        if check_key(key, options):

            overlay_time_series(movie_info_dict[key], time_series_data_path, output_dir, mouse_type)

            cut_and_splice_clips(movie_info_dict[key], clip_window=clip_window, 
                                 clip_window_origin = clip_window_origin,
                                 peak_thresh=peak_thresh, divider_clip=divider_clip)
            
            #movie_file = movie_info_dict[key]['movie_file']
            #check_video_timestamps(movie_file)
            #check_video_format(movie_file)



"""
#REFERENCE FOR FFMPEG COMMAND LINE COMMANDS
datapath=/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data

ffmpeg -i 407_social.avi 407_social.mp4
ffmpeg -i 407_social.mp4 -ss 00:02:21 -t 00:00:10 407_clip.mp4

ffmpeg -i 401_social.avi 401_social.mp4
ffmpeg -i 401_social.mp4 -ss 00:02:21 -t 00:00:10 401_clip.mp4

ffmpeg -f concat -i mylist.txt -c copy output.mp4

mylist.txt:
file 407_clip.mp4
file 401_clip.mp4

ffmpeg -i 407_clip.mp4 -vf drawbox=10:10:50:50:red,drawbox=100:100:200:200:green 407_box.mp4

ffmpeg -f concat -i <(for f in ./*.wav; do echo "file '$f'"; done) -c copy output.wav

#for adding text - this does not currently work on my installation of ffmpeg
ffmpeg -i 407_clip.mp4 -vf \
   drawtext="fontfile=/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerif.ttf: \
   text='Text to write is this one, overlaid':fontsize=20:fontcolor=red:x=100:y=100" \
   with_text3.mp4

ffmpeg -i 407_clip.mp4 -vf drawtext="fontfile=/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerif.ttf: text='\%T': fontcolor=white@0.8: x=7: y=460" with_text3.mp4

#for adding an overlay video
ffmpeg -i 407_clip.mp4 -i 407_box.mp4  -filter_complex "[0:v] setpts=PTS-STARTPTS, scale=640x480 [background]; [1:v] setpts=PTS-STARTPTS, scale=200x200 [upperleft]; [background][upperleft] overlay=shortest=1" -c:v libx264 overlay_output.mp4


#convert framerate - this is super sketchy to me, but it won't print the timestamp for the 10fps that the video is at.
ffmpeg -i 407_social.mp4 -r 30 -y 407_social_30.mp4

#Adding timecode
ffmpeg -i /Users/isaackauvar/Documents/2012-2013/ZDlab/Test_video_extract/30fps_407_clip.mp4  -vf drawtext="fontfile=/opt/X11/share/fonts/TTF/VeraMoBd.ttf: timecode='00\:00\:00\:00':rate=30: fontcolor=white@0.8: x=7: y=460" -an -y with_text3.mp4

#checking framerate
ffmpeg -i 407_social_30.mp4 2>&1 | sed -n "s/.*, \(.*\) fp.*/\1/p"


#Making divider clip (repeat one image)
# see https://trac.ffmpeg.org/wiki/Create%20a%20video%20slideshow%20from%20images
ffmpeg -loop 1 -i black.png -c:v libx264 -t 30 -pix_fmt yuv420p black1.mp4
ffmpeg -loop 1 -i white.png -c:v libx264 -t 10.47 -pix_fmt yuv420p white.mp4

#How to make time series overlay:
#Run:
plot_type=time-series-animation
exp_type='homecagesocial'
animal_id='421'
exp_date='20121105'
plot_format='.mp4'
output_path='/Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Results/Cell/Videos'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --plot-format=$plot_format --exp-type=$exp_type --animal-id=$animal_id --exp-date=$exp_date

## for adding overlay to trimmed video (trim to start time 10.47)
ffmpeg -ss 10.47 -i 421_social_tt.mp4 -i 20121105_421_homecagesocial.mp4  -filter_complex "[0:v] setpts=PTS-STARTPTS, scale=640x480 [background]; [1:v] setpts=PTS-STARTPTS, scale=100x200 [upperleft]; [background][upperleft] overlay=shortest=1" -c:v libx264 overlay_output.mp4



"""


