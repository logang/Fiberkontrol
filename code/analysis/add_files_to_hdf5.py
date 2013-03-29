import h5py
import numpy as np
import os


if __name__ == "__main__":

	from optparse import OptionParser
	parser = OptionParser()


	parser.add_option("-f", "--folder-path", dest="folder_path", default=None,
              help="Specify the path to the input directory.")
	parser.add_option("-n", "--name", dest="name", default=None,
              help="Specify the name of the dataset (to be the name of the dataset in hdf5).")
	parser.add_option("", "--hdfpath", dest="hdf_path", default=None,
              help="Specify the path to the hdf5 file.")

	(options, args) = parser.parse_args()


	files = []
	listing = os.listdir(options.folder_path)
	print "Folder directory: ", options.folder_path
	for infile in listing:
		try:
			fluor_channel = 0
			data_arr = np.load(options.folder_path + '/' + infile)['data'][:,fluor_channel]
			out_arr = np.zeros((len(data_arr), 1))
			out_arr[:, 0] = data_arr
			print options.folder_path + '/' + infile
			print np.shape(out_arr)
		except:
			print "FILE DOESN'T WORK: ", options.folder_path + '/' + infile

		# check if specified h5 file already exists
		h5_exists = os.path.isfile(options.hdf_path)
		try:
			if h5_exists:
				# write to existing h5 file
				h5_file = h5py.File(options.hdf_path)
				print "\t--> Writing to existing  HDF5 file:", options.hdf_path
			else:
				# create new h5 file
				h5_file = h5py.File(hdf_path,'w')
				print "\t--> Created new HDF5 file:", options.hdf_path
		except Exception, e:
			print "Unable to open HDF5 file", options.hdf_path, "due to error:"
			print e

		# save output array to folder in h5 file creating a data set named after the subject number
		# with columns corresponding to time, triggers, and fluorescence data, respectively.

		try:
			prefix=infile.split(".")[0]

			# group by animal number, subgroup by date, subsubgroup by run type

			if prefix.split("-")[3] not in list(h5_file):
				print "\t---> Creating group:", prefix.split("-")[3]
				subject_num= h5_file.create_group(prefix.split("-")[3])
			else:
				print "\t---> Loading group:", prefix.split("-")[3]
				subject_num = h5_file[prefix.split("-")[3]]
	            
			if prefix.split("-")[0] not in list(subject_num):
				print "\t---> Creating subgroup:", prefix.split("-")[0]
				date = subject_num.create_group(prefix.split("-")[0])
			else:
				print "\t---> Loading subgroup:", prefix.split("-")[0]
				date = subject_num[prefix.split("-")[0]]

			if prefix.split("-")[2] not in list(date):
				print "\t---> Creating subsubgroup:", prefix.split("-")[2]
				run_type = date.create_group(prefix.split("-")[2])
			else:
				print "\t---> Loading subsubgroup:", prefix.split("-")[2]
				run_type = date[prefix.split("-")[2]]

			print "\t---> Adding to datasets:", run_type.keys()
			if str(options.name) in run_type:
				del run_type[str(options.name)]
				print "\t---> Deleting datasets:", str(options.name)

			try:
				dset = run_type.create_dataset(str(options.name), data=out_arr)
				#dset.attrs[str(options.name) + "time_series_arr_names"] = options.name
				print "\t---> Writing dataset:", options.name, dset
			except Exception, e:
				print "Unable to write data array due to error:", e
		except:
			print "File " + str(infile) + " did not work"
            
		h5_file.close() # close the file