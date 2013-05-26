# Import major modules
import os
import shlex
import subprocess
import sys
from optparse import OptionParser
import csv

#------------------------------------------------------------------------------

def run_command_wrapper(cmd):
    """
    Use the subprocess module to run a shell command and collect
    and return any error messages that result.
    """
    try:
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT,
                                         shell=True)
        return [True, result, cmd]
    except subprocess.CalledProcessError,e:
        return [False, str(e) + '\nProgram output was:\n' + e.output, cmd]

#------------------------------------------------------------------------------

def generate_hdf5_file(analysis_filenames,out_path):
    """
    Callable function that generates an HDF5 file used in subsequent 
    group analyses. 
    """

    # loop over files in filename list
    for f in analysis_filenames:
        # get file path and info from filename
        path = os.path.abspath(os.path.dirname(f))
        fname = f.split('/')[-1].split('.')[0]
        info = fname.split('-')
        # only process GC5 files, generating a command line command to
        # run fiber_record_analyze.py on the data.
        if ( info[1] == 'GC5' or info[1] == 'GC3' or info[1] == 'EYFP' or 
             info[1] == 'GC5_NAcprojection' ):
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

            elif info[2] == 'EPM':
                cmd += ' --trigger-path '
                cmd += os.path.join(path, info[1] + '_'+info[3]+'_EPM ')
                cmd += '--exp-type=EPM'

            elif info[2] == 'sucrose':
                cmd += ' --exp-type=sucrose '

            else:
                print info[3], "is not a recognized experiment type. Check filenames."

            cmd += ' -o ' + out_path
            cmd += ' --save-to-h5 ' + out_path
            cmd += ' --fluor-normalization raw '
            cmd += ' --smoothness=0 '
            cmd += ' --mouse-type=' + info[1]
            result = run_command_wrapper(cmd)
            print ""
            print cmd
            print result

#------------------------------------------------------------------------------

def debleach_files(analysis_filenames):
    """
    Wrapper for functions that remove  bleaching trends from calcium recordings. 
    """
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

        #this command may become deprecated. I like it, however, because it 
        # shows prints in real time
        # instead of after the command has finished running
        os.system(cmd) 

    print "Now make a folder Flat/ for each day of trials, and place in this folder "
    print "the 'flat' time series for a each trial. Determine which is the 'flat' "
    print "time series by using the plots comparing original and debleached "
    print "time series that are found in the Debleached/ folder."
    print "Then use add_flattened_files_to_hdf5 to add these flattened files to the hdf5."
    print "Be sure to add the Flat/ directories to the flat_directories global variable "
    print "in preprocessing.py"

#------------------------------------------------------------------------------   
# Utility functions for loading and saving data 

def add_flattened_files_to_hdf5(flat_directories, out_path):
    """
    Adds debleached files to the hdf5 file specified.
    """
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

def read_filenames(filenames_file, path_to_filenames=None):
    """
    Read in filenames of data files from 'filenames_file' (a txt file),
    and optionally append a path prefix 'path_to_filenames'.
    """
    print "Filename: ", filenames_file
    filenames = []
    with open(filenames_file, 'rb') as f:
        reader = csv.reader(f, delimiter='\n', quoting=csv.QUOTE_NONE)
        for row in reader:
            if path_to_filenames is not None:
                filenames.append(path_to_filenames + row[0])
            else:
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
    for d in experiment_dates:
        flat_dirs.append(path_to_npz_data + str(d) + '/Flat')

    return flat_dirs

#------------------------------------------------------------------------------
# Main
            
if __name__ == '__main__':

    parser = OptionParser()

    parser.add_option("-o", "--out-path", dest="out_path", default=None, 
                      help="Specify the path to the output HDF5 file. This also"
                            " specifies which HDF5 file to change.")

    parser.add_option("", "--save-debleach", action="store_true", default=False, 
                      dest="save_debleach", 
                      help=("Debleach fluorescence time series by " 
                            "fitting with an exponential curve."))

    parser.add_option("", "--generate-hdf5", action="store_true", default=False, 
                      dest="generate_hdf5", 
                      help=("Produce and hdf5 file with the raw data stored "
                            "in the files listed in analysis_filenames."))

    parser.add_option("", "--add-flattened-files", action="store_true", 
                      default=False, dest="add_flattened_files", 
                      help=("Add flattened files to hdf5 for each "
                            "trial, as dataset labeled 'flat' "))

    parser.add_option("-f", "--analysis-filenames-file", 
                      dest="analysis_filenames_file", 
                      default='analysis_filenames.txt', 
                      help=("Specify the path to the list of "
                            "files to be included in analysis."))

    parser.add_option("-d", "--experiment-dates", dest="experiment_dates_file", 
                      default='experiment_dates.txt',
                      help=("Specify the path to the list of dates "
                            "of experiments in format YEARMODA i.e. 20130128."))

    (options, args) = parser.parse_args()

    # Check for required input path
    if len(args) < 1:
        print 'You must supply at a path to the NPZ files to be analyzed.'
        sys.exit(1)

    path_to_npz_data = args[0]

    analysis_filenames = read_filenames(options.analysis_filenames_file, path_to_npz_data)
    experiment_dates = read_filenames(options.experiment_dates_file)
    print "This file contains a list of files to be used for batch processing."
    print "The list contains:"
    for f in analysis_filenames:
        print '\t',f

    if options.out_path is None:
        out_path = os.path.join( os.path.split(args[0])[0], 'preprocessed_data.h5' )
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            print "Created output directory:", out_path
    else:
        out_path = options.out_path # path to HDF5 output file

    if options.generate_hdf5:
        try:
            os.remove(out_path)
            print "Removed old hdf5 file"
        except OSError:
            pass
        generate_hdf5_file(analysis_filenames, out_path)

    if options.add_flattened_files:
        flat_directories = get_flat_directories(experiment_dates, path_to_npz_data)
        add_flattened_files_to_hdf5(flat_directories, out_path)
    
    if options.save_debleach:
        debleach_files(analysis_filenames)

# EOF
