import numpy as np

# R imports
from rpy2.robjects.packages import importr
DSE = importr("dse")
import rpy2.robjects as ro

# Define and import R functions
ro.r( """
      data_to_R = function(input, output)
      {
        out = TSdata(input=input, output=output)
        out
      }

      sys_id = function(data_object, model_type="ls")
      {
        if(model_type)=='ls' { model = estVARXls(data_object) }
        if(model_type)=='ar' { model = estVARXar(data_object) }
        if(model_type)=='ss' { model = estVARXss(data_object) }        
        if(model_type)=='bft' { model = estVARXbft(data_object) }
        if(model_type)=='mle' { model = estVARXmle(data_object, max.lag=1) }
        model
      }

      plot_model = function(model)
      {
        tfplot(model)
      }

      """)

data_to_R = ro.r['data_to_R']
sys_id = ro.r['sys_id']
plot_model = ro.r['plot_model']

#spmd = ro.r['pmdfit']

# These lines are required before we can covert numpy arrays into R arrays.
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

def convert_numpy_to_R(input, output):
    """
    Converts two numpy arrays, 'input' and 'output' to R time series
    objects suitable for use with the ts and dse packages.
    """
    pass

# def group_bout_heatmaps(all_data, options, exp_type, time_window, df_max=3.5):
#     """
#     Save out 'heatmaps' showing time on the x axis, bouts on the y axis, and representing signal
#     intensity with color.
#     """
#     i=0 # color counter
#     for animal_id in all_data.keys():
#         # load data from hdf5 file by animal-date-exp_type
#         animal = all_data[animal_id]
#         for dates in animal.keys():
#             # Create figure
#             fig = pl.figure()
#             ax = fig.add_subplot(2,1,1)
            
#             date = animal[dates]
#             FA = FiberAnalyze( options )
#             FA.subject_id = animal_id
#             FA.exp_date = dates
#             FA.exp_type = exp_type
#             FA.load(file_type="hdf5")

#             event_times = FA.get_event_times("rising")
