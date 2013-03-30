import os
import shlex
import subprocess
import sys
from optparse import OptionParser
import csv




#analysis_filenames = ['/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20121105/20121105-EYFP-homecagenovel-8500-600patch-2012_11_5_17-54_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20121105/20121105-EYFP-homecagesocial-8499-600patch-2012_11_5_17-39_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20121105/20121105-EYFP-homecagesocial-8500-600patch-2012_11_5_19-27_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20121105/20121105-GC5-homecagenovel-421-600patch-2012_11_5_18-52_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20121105/20121105-GC5-homecagenovel-660-600patch-2012_11_5_18-26_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20121105/20121105-GC5-homecagenovel-718-600patch-2012_11_5_16-50_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20121105/20121105-GC5-homecagenovel-8497-600patch-2012_11_5_17-25_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20121105/20121105-GC5-homecagesocial-421-600patch-2012_11_5_17-6_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20121105/20121105-GC5-homecagesocial-660-600patch-2012_11_5_16-33_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20121105/20121105-GC5-homecagesocial-718-600patch-2012_11_5_18-40_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20121105/20121105-GC5-homecagesocial-8497-600patch-2012_11_5_19-6_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagenovel-82-2013_1_8_17-37_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagenovel-401-2013_1_8_18-5_run_number_0.npz','/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagenovel-402-2013_1_8_16-27_run_number_0.npz','/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagenovel-403-2013_1_8_16-58_run_number_0.npz' ,'/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagenovel-2227-2013_1_8_13-5_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagenovel-2229-2013_1_8_14-33_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagenovel-2248-2013_1_8_13-35_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagenovel-2249-2013_1_8_15-2_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagesocial-82-2013_1_8_16-13_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagesocial-401-2013_1_8_16-41_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagesocial-402-2013_1_8_17-50_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagesocial-403-2013_1_8_18-17_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagesocial-2227-2013_1_8_14-46_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagesocial-2229-2013_1_8_12-50_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagesocial-2248-2013_1_8_15-15_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagesocial-2249-2013_1_8_13-21_run_number_0.npz'] 

#analysis_filenames = ['/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121105/20121105-EYFP-homecagesocial-8500-600patch-2012_11_5_19-27_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121105/20121105-GC5-homecagesocial-8497-600patch-2012_11_5_19-6_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121105/20121105-GC5-homecagenovel-421-600patch-2012_11_5_18-52_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121105/20121105-GC5-homecagesocial-718-600patch-2012_11_5_18-40_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121105/20121105-GC5-homecagenovel-660-600patch-2012_11_5_18-26_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121105/20121105-EYFP-homecagenovel-8500-600patch-2012_11_5_17-54_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121105/20121105-EYFP-homecagesocial-8499-600patch-2012_11_5_17-39_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121105/20121105-GC5-homecagenovel-8497-600patch-2012_11_5_17-25_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121105/20121105-GC5-homecagesocial-421-600patch-2012_11_5_17-6_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121105/20121105-GC5-homecagenovel-718-600patch2012_11_5_16-50_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121105/20121105-GC5-homecagesocial-660-600patch-2012_11_5_16-33_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20130108/20130108-GC5-homecagesocial-403-2013_1_8_18-17_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20130108/20130108-GC5-homecagenovel-401-2013_1_8_18-5_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20130108/20130108-GC5-homecagesocial-402-2013_1_8_17-50_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20130108/20130108-GC5-homecagenovel-82-2013_1_8_17-37_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20130108/20130108-GC5-homecagenovel-403-2013_1_8_16-58_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20130108/20130108-GC5-homecagesocial-401-2013_1_8_16-41_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20130108/20130108-GC5-homecagenovel-402-2013_1_8_16-27_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20130108/20130108-GC5-homecagesocial-82-2013_1_8_16-13_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20130108/20130108-GC5-homecagesocial-2248-2013_1_8_15-15_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20130108/20130108-GC5-homecagenovel-2249-2013_1_8_15-2_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20130108/20130108-GC5-homecagesocial-2227-2013_1_8_14-46_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20130108/20130108-GC5-homecagenovel-2229-2013_1_8_14-33_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20130108/20130108-GC5-homecagenovel-2248-2013_1_8_13-35_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20130108/20130108-GC5-homecagesocial-2249-2013_1_8_13-21_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20130108/20130108-GC5-homecagenovel-2227-2013_1_8_13-5_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20130108/20130108-GC5-homecagesocial-2229-2013_1_8_12-50_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121017/20121017-GC5-homecagenovel-8621-600patch-2012_10_18_15-49_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121017/20121017-GC5-homecagenovel-8622-600patch-2012_10_18_15-26_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121017/20121017-GC5-homecagesocial-421-600patch2012_10_17_19-0_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121017/20121017-GC5-homecagesocial-660-600patch2012_10_17_18-20_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121017/20121017-GC5-homecagesocial-718-600patch2012_10_17_17-40_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121017/20121017-GC5-homecagesocial-8622-600patch-2012_10_17_15-41_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121017/20121017-GC5-homecagesocial-8621-600patch-2012_10_17_15-22_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121017/20121017-YFP-homecagesocial-8379-600patch-2012_10_17_15-11_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121017/20121017-YFP-homecagesocial-8381-600patch-2012_10_17_15-0_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121008/20121008-EYFP-sucrose-8498-2012_10_12_17\:0_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121008/20121008-EYFP-sucrose-8499-2012_10_12_15\:21_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121008/20121008-EYFP-sucrose-8500-2012_10_12_13\:56_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121008/20121008-GC5-sucrose-8497-2012_10_12_12\:1_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121008/20121008-EYFP-sucrose-8567-2012_10_11_18\:52_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121008/20121008-GC5-sucrose-8568-2012_10_11_18\:3_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121008/20121008-GC5-sucrose-8569-2012_10_11_16\:54_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121008/20121008-EYFP-sucrose-8379-2012_10_11_15\:42_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121008/20121008-EYFP-sucrose-8250-run2-2012_10_11_14\:45_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121008/20121008-EYFP-sucrose-8381-2012_10_10_19\:1_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121008/20121008-EYFP-sucrose-8250-run1-2012_10_10_18\:11_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121008/20121008-EYFP-sucrose-8246-2012_10_10_16\:45_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121008/20121008-GC5-sucrose-667-200um-2012_10_10_16\:0_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121008/20121008-GC5-sucrose-409-200um-2012_10_10_15\:10_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121008/20121008-GC5-sucrose-625-200um-2012_10_10_14\:31_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121008/20121008-GC5-sucrose-421-run2-2012_10_10_12\:31_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121008/20121008-GC3-sucrose-636-2012_10_9_16\:35_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121008/20121008-GC5-sucrose-718-2012_10_9_15\:39_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121008/20121008-GC3-sucrose-398-2012_10_9_14\:53_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121008/20121008-GC5-sucrose-660-2012_10_9_14\:5_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121008/20121008-GC5-sucrose-8520-2012_10_8_16-26_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121008/20121008-GC5-sucrose-8521-run3-2012_10_8_15-41_run_number_0.npz', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121008/20121008-GC5-sucrose-8522-2012_10_8_14-34_run_number_0.npz']
#flat_directories = ['/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121105/Flat', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121008/Flat', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20130108/Flat', '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/20121017/Flat']

#----------------------------------------------------------------------------------------                                                                                                           
def run_command_wrapper(cmd):
    """                                                                                                  
    Use the subprocess module to run a shell command and collect                                         
    and return any error messages that result.                                                           
    """
    try:
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT,shell=True)
        return [True, result, cmd]
    except subprocess.CalledProcessError,e:
        return [False, str(e) + '\nProgram output was:\n' + e.output, cmd]

#----------------------------------------------------------------------------------------   

def generate_hdf5_file(analysis_filenames,out_path):
    # loop over files in filename list
    for f in analysis_filenames:
        # get file path and info from filename
        path = os.path.abspath(os.path.dirname(f))
        fname = f.split('/')[-1].split('.')[0]
        info = fname.split('-')
        # only process GC5 files, generating a command line command to
        # run fiber_record_analyze.py on the data.
        if info[1] == 'GC5' or info[1] == 'GC3' or info[1] == 'EYFP' or info[1] == 'GC5_NAcprojection':
            cmd = 'python fiber_record_analyze.py -i '
            cmd += f                
            
            
            if info[2] == 'homecagesocial':
                cmd += ' --trigger-path '
                cmd += os.path.join(path, info[1] + '_'+info[3]+'_social ')
                cmd += '--exp-type=homecagesocial '

            elif info[2] == 'homecagenovel':
                cmd += ' --trigger-path '
                cmd += os.path.join(path, info[1] + '_'+info[3]+'_novel ')
                cmd += '--exp-type=homecagenovel '

            elif info[2] == 'sucrose':
                cmd += '--exp-type=sucrose '
            else:
                print info[3], "is not a recognized experiment type. Check filenames."

            cmd += ' -o ' + out_path
            cmd += ' --save-to-h5 ' + out_path
            cmd += ' --fluor-normalization raw '
            cmd += ' --smoothness=0 '
            cmd += ' --mouse-type=' + info[1]
            result = run_command_wrapper(cmd)
            print result


def add_flattened_files_to_hdf5(flat_directories, out_path):
    for d in flat_directories:
        cmd = 'python add_files_to_hdf5.py -f '
        cmd += d 
        cmd += ' -n flat'
        cmd += ' --hdfpath='
        cmd += out_path
        #Note that --add-new flag remains False
        print cmd
        
        #result = run_command_wrapper(cmd)
        #print result
        os.system(cmd)

def debleach_files(analysis_filenames):
    for f in analysis_filenames:
        # get file path and info from filename
        path = os.path.abspath(os.path.dirname(f))
        fname = f.split('/')[-1].split('.')[0]
        info = fname.split('-')

        out_path = path + '/Debleached/' + fname

        cmd = 'python fiber_record_analyze.py -i '
        cmd += f 
        cmd += ' -o '
        cmd += out_path
        cmd += ' --fluor-normalization=raw'
        cmd += ' --save-debleach'
        cmd += ' --save-and-exit'

        os.system(cmd) #this command may become deprecated. I like it, however, because it shows prints in real time
                        # instead of after the command has finished running

    print "Now make a folder Flat/ for each day of trials, and place in this folder "
    print "the 'flat' time series for a each trial. Determine which is the 'flat' "
    print "time series by using the plots comparing original and debleached "
    print "time series that are found in the Debleached/ folder."
    print "Then use add_flattened_files_to_hdf5 to add these flattened files to the hdf5."
    print "Be sure to add the Flat/ directories to the flat_directories global variable "
    print "in preprocessing.py"

def read_filenames(filenames_file):
    
    print "Filename: ", filenames_file
    filenames = []
    with open(filenames_file, 'rb') as f:
        reader = csv.reader(f, delimiter='\n', quoting=csv.QUOTE_NONE)
        for row in reader:
            filenames.append(row[0])

    return filenames

def get_flat_directories(experiment_dates, path_to_npz_data):
    """
    Returns an array with the paths to the directories containing
    the flattened time series for each of the days of experiments
    listed in the experiment_dates file
    These dates should correspond to all files
    to be analyzed in analysis_files
    """

    flat_dirs = []
    path_to_npz_data = '/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/' #change this to whichever directory in which the data is stored
    for d in experiment_dates:
        flat_dirs.append(path_to_npz_data + str(d) + '/Flat')

    return flat_dirs


            
if __name__ == '__main__':

    parser = OptionParser()

    parser.add_option("", "--save-debleach", action="store_true", default=False, dest="save_debleach",
                      help="Debleach fluorescence time series by fitting with an exponential curve.")
    parser.add_option("", "--generate-hdf5", action="store_true", default=False, dest="generate_hdf5",
                      help="Produce and hdf5 file with the raw data stored in the files listed in analysis_filenames.")
    parser.add_option("", "--add-flattened-files", action="store_true", default=False, dest="add_flattened_files",
                      help="Add flattened files to hdf5 for each trial, as dataset labeled 'flat' ")
    parser.add_option("-f", "--analysis-filenames-file", dest="analysis_filenames_file", default='analysis_filenames.txt',
                      help="Specify the path to the list of files to be included in analysis.")
    parser.add_option("-d", "--experiment-dates", dest="experiment_dates_file", default='experiment_dates.txt',
                  help="Specify the path to the list of dates of experiments in format YEARMODA i.e. 20130128.")
    parser.add_option("", "--path-to-npz-data", dest="path_to_npz_data", default='/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/',
                  help="Specify the path to the folder containing folders corresponding to the trials of a certain day")

    (options, args) = parser.parse_args()

    analysis_filenames = read_filenames(options.analysis_filenames_file)
    experiment_dates = read_filenames(options.experiment_dates_file)
    print "This file contains a list of files to be used for batch processing."
    print "The list contains:"
    for f in analysis_filenames:
        print '\t',f

#    generate_hdf5_file(analysis_filenames, '/Users/logang/Documents/Results/FiberRecording/Cell/all_data_raw.h5')
    
    out_path = '/Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Results/Cell/all_data_raw.h5'
    if options.generate_hdf5:
        try:
            os.remove(out_path)
            print "Removed old hdf5 file"
        except OSError:
            pass
        generate_hdf5_file(analysis_filenames, out_path)

    if options.add_flattened_files:
        flat_directories = get_flat_directories(experiment_dates, options.path_to_npz_data)
        add_flattened_files_to_hdf5(flat_directories, out_path)
    
    if options.save_debleach:
        debleach_files(analysis_filenames)
