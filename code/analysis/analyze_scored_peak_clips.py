import numpy as np
from numpy import genfromtxt
import pickle
import matplotlib.pyplot as plt
import os

def load_peak_vals(filepath):
    """
    Returns a social and novel
    dict containing an array with
    the magnitude of each scored peak.
    Each entry is a trial.
    """
    peak_vals_path = open(filepath, 'rb')
    peak_vals = pickle.load(peak_vals_path)

    social_vals = dict()
    novel_vals = dict()

    for key in peak_vals.keys():
        mouse_number, date, behav_type = key.split('_')
        if behav_type == 'homecagesocial':
            social_vals[mouse_number] = peak_vals[key]
        elif behav_type == 'homecagenovel':
            novel_vals[mouse_number] = peak_vals[key]

    return {'social':social_vals, 'novel':novel_vals}

def plot_histogram(behaviors, data, behav_type, mouse_type, 
                   interaction_behaviors=None, solitary_behaviors=None,
                   weighted=False, outpath = None,
                   before_or_after_conspecific=None, normalized=True):
    behavior_tally = dict()
    for key in behaviors: #Initialize behaviors you want to be sure to plot
        behavior_tally[key] = 0

    for key in data.keys():
        if data[key]['mouse_type'] == mouse_type:
            if before_or_after_conspecific == None:
                indices = range(len(data[key]['peak_vals']))
            elif before_or_after_conspecific == 'before':
                indices = range(int(data[key]['conspecific_ind']) + 1)
            elif before_or_after_conspecific == 'after':
                indices = range(int(data[key]['conspecific_ind']) + 1, len(data[key]['peak_vals']))

            for i in indices:
                behav = data[key]['behavs'][i]
                
                if weighted:
                    tally = data[key]['peak_vals'][i]
                else:
                    tally = 1    

                if behavior_tally.has_key(behav):
                    behavior_tally[behav] += tally
                else:
                    behavior_tally[behav] = tally

    for key in behavior_tally.keys(): #Combine dropped 
        if key.rfind('dropped') > 0:
            vals = key.split(' ')
            behavior_tally[vals[0]] += behavior_tally[key]


    if interaction_behaviors is not None:
        behavior_labels = ['interacting', 'solitary']
        interacting_val = sum(np.array([behavior_tally[key] for key in interaction_behaviors]))
        solitary_val = sum(np.array([behavior_tally[key] for key in solitary_behaviors]))
        behavior_hist = [interacting_val, solitary_val]
    else:
        behavior_labels = behaviors
        behavior_hist = [behavior_tally[key] for key in behavior_labels]
    
    if normalized:
        total_counts = sum(behavior_hist)
        behavior_hist = np.array(behavior_hist, dtype=float)/sum(behavior_hist)


    plt.figure()
    plt.bar(np.arange(len(behavior_hist)), behavior_hist, width=0.8)
    plt.xticks(np.arange(len(behavior_hist))+0.4, behavior_labels, rotation=17)
    title = ''
    if interaction_behaviors is None:
        title = 'all__'
    title = title + str(behav_type) 
    if mouse_type == 'GC5':
        title = title + '__cellbodies'
    elif mouse_type == 'GC5_Nacprojection':
        title = title + '__projections'
    if before_or_after_conspecific is not None:
        title = title + '__'+ before_or_after_conspecific + '_conspecific'
    if normalized:
        if weighted:
            plt.ylabel('Normalized weighted counts (total counts = '+str(total_counts)+')')
        else:
            plt.ylabel('Normalized counts (total counts = '+str(total_counts)+')')
    else:
        if weighted:
            plt.ylabel('Total weighted counts')
        else:
            plt.ylabel('Total counts')

    if normalized:
        plt.ylim([0, 1])
    plt.title(title)
    if outpath is not None:
        if not os.path.isdir(outpath):
            os.makedirs(outpath)
        plt.savefig(outpath + title + '.pdf')

        f = open(outpath+title+'.txt', 'w')
        f.write(str(behavior_hist) + '\n')
        f.write(str(behavior_labels) + '\n')
        f.close()

if __name__ == "__main__":
    behaviors = ['ambulation', 'groom', 'sniff', 'rest', 'withdraw', 'burrow', 'rear', 'approach', 'head extension']
    interaction_behaviors = ['sniff', 'approach', 'withdraw']
    solitary_behaviors = ['ambulation', 'groom', 'rest', 'burrow', 'rear', 'head extension']
    path = '/Users/isaackauvar/Dropbox/FiberPhotometry/DATA/behavior/Peak_clips/'

    peak_vals = load_peak_vals(path+'peak_vals.pkl')

    for behav_type in ['social', 'novel']:
        raw_data = genfromtxt(path + 'GC_peak_clips_scored_'+behav_type+'.csv', 
                          delimiter=',', dtype=str)# unpack=True)
        
        data = dict() #keys: name, dict: keys: behavs, conspecific_indices
        behavior_data = np.array(raw_data[3:][:]).T
        for i in range(1, len(raw_data[0])):
            name = raw_data[0][i]
            mouse_type = raw_data[1][i]
            conspecific_ind = raw_data[2][i]
            behavs = np.array(behavior_data[i][:])
            data[name] = {'conspecific_ind':conspecific_ind, 'behavs':behavs, 
                          'peak_vals':peak_vals[behav_type][name], 
                          'mouse_type':mouse_type}

        outpath = path + '/plots/'
        for mouse_type in ['GC5', 'GC5_Nacprojection']:
            interaction_behaviors = None
            solitary_behaviors = None
            weighted = False
           

            plot_histogram(behaviors, data, behav_type, mouse_type, 
                           outpath = outpath+'/unnormalized_unweighted/',
                           normalized=False)
            plot_histogram(behaviors, data, behav_type, mouse_type,  
                            interaction_behaviors=interaction_behaviors, 
                            solitary_behaviors=solitary_behaviors,
                            weighted=weighted, outpath=outpath+'/total/')
            plot_histogram(behaviors, data, behav_type, mouse_type, 
                            interaction_behaviors=interaction_behaviors, 
                            solitary_behaviors=solitary_behaviors,
                            weighted=weighted, outpath=outpath+'/before/', 
                            before_or_after_conspecific='before')
            plot_histogram(behaviors, data, behav_type, mouse_type, 
                            interaction_behaviors=interaction_behaviors, 
                            solitary_behaviors=solitary_behaviors,
                            weighted=weighted, outpath=outpath+'/after/',
                            before_or_after_conspecific='after')

