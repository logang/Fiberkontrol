import pickle
import matplotlib.pyplot as plt


if False:
    fixed_blind_data = open('/Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Results/Cell/Plots/Finalz_post_SfN_000rxn/blind_time_series.pkl', 'rb')
    processed_data = open('/Users/isaackauvar/Dropbox/FiberPhotometry/DATA/processed.pkl', 'rb')


    fb = pickle.load(fixed_blind_data)
    pd = pickle.load(processed_data)
    
    fixed_labels = fb['labels']
    print fixed_labels
    

    for key in pd:
        pd[key]['labels'] = {}

        pd['labels'] = fb['labels']
        
        filename ='/Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Results/Cell/Plots/Finalz_post_SfN_000rxn/'+ 'blind_time_series.pkl'
        filename = '/Users/isaackauvar/Dropbox/FiberPhotometry/DATA/processed_fixed_labels.pkl'
        pickle.dump( pd, open( filename, "wb" ) )


a = open('/Users/isaackauvar/Dropbox/FiberPhotometry/DATA/processed_fixed_labels.pkl', 'rb')
data = pickle.load(a)

plt.figure()
plt.plot(data[10]['time_stamps_decimated'], data[10]['fluor_data_decimated'])
plt.title(data['labels'][10])

plt.figure()
plt.plot(data[11]['time_stamps_decimated'], data[11]['fluor_data_decimated'])
plt.title(data['labels'][11])
plt.figure()
plt.plot(data[0]['time_stamps_decimated'], data[0]['fluor_data_decimated'])
plt.title(data['labels'][0])

print data

#plt.show()




