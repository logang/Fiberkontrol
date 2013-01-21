import os

analysis_filenames = ['/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20121105/20121105-EYFP-homecagenovel-8500-600patch-2012_11_5_17-54_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20121105/20121105-EYFP-homecagesocial-8499-600patch-2012_11_5_17-39_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20121105/20121105-EYFP-homecagesocial-8500-600patch-2012_11_5_19-27_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20121105/20121105-GC5-homecagenovel-421-600patch-2012_11_5_18-52_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20121105/20121105-GC5-homecagenovel-660-600patch-2012_11_5_18-26_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20121105/20121105-GC5-homecagenovel-718-600patch-2012_11_5_16-50_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20121105/20121105-GC5-homecagenovel-8497-600patch-2012_11_5_17-25_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20121105/20121105-GC5-homecagesocial-421-600patch-2012_11_5_17-6_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20121105/20121105-GC5-homecagesocial-660-600patch-2012_11_5_16-33_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20121105/20121105-GC5-homecagesocial-718-600patch-2012_11_5_18-40_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20121105/20121105-GC5-homecagesocial-8497-600patch-2012_11_5_19-6_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagenovel-82-2013_1_8_17-37_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagenovel-401-2013_1_8_18-5_run_number_0.npz','/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagenovel-402-2013_1_8_16-27_run_number_0.npz','/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagenovel-403-2013_1_8_16-58_run_number_0.npz' ,'/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagenovel-2227-2013_1_8_13-5_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagenovel-2229-2013_1_8_14-33_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagenovel-2248-2013_1_8_13-35_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagenovel-2249-2013_1_8_15-2_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagesocial-82-2013_1_8_16-13_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagesocial-401-2013_1_8_16-41_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagesocial-402-2013_1_8_17-50_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagesocial-403-2013_1_8_18-17_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagesocial-2227-2013_1_8_14-46_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagesocial-2229-2013_1_8_12-50_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagesocial-2248-2013_1_8_15-15_run_number_0.npz', '/Users/logang/Dropbox/Fiberkontrol/Fiberkontrol\ Data/Fiberkontrol\ Lisa\ Data/20130108/20130108-GC5-homecagesocial-2249-2013_1_8_13-21_run_number_0.npz'] 

#----------------------------------------------------------------------------------------                                                                                                           
def run_command_wrapper(cmd):
    """                                                                                                  Use the subprocess module to run a shell command and collect                                         and return any error messages that result.                                                           """
    import subprocess
    try:
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT,shell=True)
        return [True, result, cmd]
    except subprocess.CalledProcessError,e:
        return [False, str(e) + '\nProgram output was:\n' + e.output, cmd]

#----------------------------------------------------------------------------------------   

def generate_GC5_hdf5_file(analysis_filenames,out_path):
    # loop over files in filename list
    for f in analysis_filenames:
        # get file path and info from filename
        path = os.path.abspath(os.path.dirname(f))
        fname = f.split('/')[-1].split('.')[0]
        info = fname.split('-')
        # only process GC5 files, generating a command line command to
        # run fiber_record_analyze.py on the data.
        if info[1] == 'GC5':
            cmd = 'python fiber_record_analyze.py -i '
            cmd += f
            cmd += ' --trigger-path '
            if info[2] == 'homecagesocial':
                cmd += os.path.join(path, 'GC5_'+info[3]+'_social ')
            elif info[2] == 'homecagenovel':
                cmd += os.path.join(path, 'GC5_'+info[3]+'_novel ')
            else:
                print info[3], "is not a recognized experiment type. Check filenames."
            cmd += '-o ' + out_path
            cmd += ' --save-to-h5 ' + out_path
            cmd += ' --fluor-normalization raw '
            result = run_command_wrapper(cmd)
            print result
            
if __name__ == '__main__':
    print "This file contains a list of files to be used for batch processing."
    print "The list contains:"
    for f in analysis_filenames:
        print '\t',f
    generate_GC5_hdf5_file(analysis_filenames, '/Users/logang/Documents/Results/FiberRecording/Cell/all_data_raw.h5')
