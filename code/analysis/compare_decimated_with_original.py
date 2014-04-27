"""
compare_decimated_with_original.py


Compare timing of peaks from new (i.e. processed) data
with peaks from original data.
"""
import pickle
import numpy as np 
import matplotlib.pyplot as plt 


#processed_data = open('/Users/isaackauvar/Dropbox/FiberPhotometry/DATA/processed_fixed_labels.pkl', 'rb')
processed_data = open('/Users/isaackauvar/Dropbox/FiberPhotometry/DATA/processed.pkl', 'rb')
original_blind_data = open('/Users/isaackauvar/Dropbox/FiberPhotometry/DATA/blind_time_series.pkl', 'rb')

orig = pickle.load(original_blind_data)
processed = pickle.load(processed_data)

print processed
#print orig


trial = 0 #which trial (i.e. animal_id_date_behavior_type) to plot

new_fluor_data = processed[trial]['fluor_data_decimated']
new_time_stamps = processed[trial]['time_stamps_decimated'] 
orig_fluor_data = orig[trial]['fluor_data']
orig_time_stamps = orig[trial]['time_stamps']

new_time_stamps = new_time_stamps - orig_time_stamps[15] ## account for delay imposed by FIR decimation filter
print "Delay", orig_time_stamps[15]

# new_time_stamps = new_time_stamps * orig_time_stamps[-1]/new_time_stamps[-1]
# new_peak_time = new_peak_times * orig_time_stamps[-1]/new_time_stamps[-1]

new_peak_indices = processed[trial]['peak_indices']
new_peak_times = new_time_stamps[new_peak_indices]
new_peak_vals = new_fluor_data[new_peak_indices]

orig_peak_indices = np.searchsorted(orig_time_stamps, new_peak_times, side='left')
orig_peak_times = orig_time_stamps[orig_peak_indices]
orig_peak_vals = orig_fluor_data[orig_peak_indices]



clip_window = [5, 5]

before_ind_orig = np.where(orig_time_stamps>clip_window[0])[0][0]
after_ind_orig = np.where(orig_time_stamps>clip_window[1])[0][0]

before_ind_new = np.where(new_time_stamps>clip_window[0])[0][0]
after_ind_new = np.where(new_time_stamps>clip_window[1])[0][0]

for i in range(len(new_peak_indices)):
	ind_orig = orig_peak_indices[i]
	ind_new = new_peak_indices[i]

	plt.figure()

	plt.plot(new_time_stamps[ind_new - before_ind_new:ind_new + after_ind_new],
		     new_fluor_data[ind_new - before_ind_new:ind_new + after_ind_new], 'r')
	plt.plot(orig_time_stamps[ind_orig - before_ind_orig:ind_orig + after_ind_orig],
			 orig_fluor_data[ind_orig - before_ind_orig:ind_orig + after_ind_orig], 'b')


plt.plot(new_time_stamps, new_fluor_data, 'r')
plt.plot(orig_time_stamps, orig_fluor_data, 'b')
plt.show()

## Old version, you can delete this...
# fluor_data_decimated = processed[trial]['fluor_data_decimated']
# time_stamps_decimated = processed[trial]['time_stamps_decimated']
# peak_indices = processed[trial]['peak_indices']
# peak_times = time_stamps_decimated[peak_indices]
# peak_vals = fluor_data_decimated[peak_indices]

# fluor_data = orig[trial]['fluor_data']
# time_stamps = orig[trial]['time_stamps']
# orig_peak_indices = np.searchsorted(time_stamps, peak_times, side='left')
# orig_peak_times = time_stamps[orig_peak_indices]
# orig_peak_vals = fluor_data[orig_peak_indices]


# clip_window = [5, 5]

# before_ind_orig = np.where(time_stamps>clip_window[0])[0][0]
# after_ind_orig = np.where(time_stamps>clip_window[1])[0][0]

# before_ind_dec = np.where(time_stamps_decimated>clip_window[0])[0][0]
# after_ind_dec = np.where(time_stamps_decimated>clip_window[1])[0][0]

# for i in range(len(peak_indices)):
# 	ind_orig = orig_peak_indices[i]
# 	ind_dec = peak_indices[i]

# 	plt.figure()
# 	plt.plot(time_stamps_decimated[ind_dec - before_ind_dec:ind_dec + after_ind_dec],
# 			   fluor_data_decimated[ind_dec - before_ind_dec:ind_dec + after_ind_dec], 'r')
# 	plt.plot(time_stamps[ind_orig - before_ind_orig:ind_orig + after_ind_orig],
# 		     fluor_data[ind_orig - before_ind_orig:ind_orig + after_ind_orig], 'b')

# plt.show()


