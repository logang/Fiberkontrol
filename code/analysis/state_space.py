import numpy as np
import h5py
import time
import scipy.stats.mstats as mstats
import pylab as pl

# local imports
from fiber_record_analyze import FiberAnalyze

# R imports
from rpy2.robjects.packages import importr
DSE = importr("dse")
RWT = importr("rwt")
import rpy2.robjects as ro

# Define and import R functions
ro.r( """
       data_to_R = function(input, output)
       {
         out = TSdata(input=input, output=output)
       }

      sys_id = function(input, output, model_type="ls", solver="VARX", order=c(1,0,1))
      {
        if( solver == "VARX" )
        {
          data_object = data_to_R(input,output) 
          if(model_type=='ls') { model = estVARXls(data_object,max.lag=10, trend=TRUE) }
          if(model_type=='ar') { model = estVARXar(data_object) }
          if(model_type=='ss') { model = estVARXss(data_object) }        
          if(model_type=='bft') { model = bft(data_object, max.lag=10, standardize=F, subtract.means=T) }
          if(model_type=='mle') { model = estVARXmle(data_object, max.lag=1) }
        }
        else
        {
          model = arima(output, order=order, xreg=input)
        }
        model
      }

      model_coefs = function(model)
      {
        coefs = coefficients(model)
        coefs
      }

      model_summary = function(model, solver="VARX")
      {
        if( solver == "VARX" )
        {
          summary(model)
          print(model)
          tfplot(model)
          checkResiduals(model)
          sum = summary(model)
        }
        else
        {
          print(model)
          sum = model$aic
        }
        sum
      }

      forecast = function(model,n,step=100,horizon=1000, solver='VARX', input=NULL, output=NULL)
      {
        if( solver == 'VARX' )
        {
          z = featherForecasts(model, model, from.periods = seq(1000,n-horizon,by=step), horizon=horizon)
          par(mfrow=c(1,2))
          tfplot(z)
          lines(model$data$input)
          lines(model$data$output,col="red")
        } 
        else
        { 
          pred = predict(model, n.ahead=6, newxreg=input)
          plot(scale(output,scale=F),type='l')
          lines(scale(pred$pred,scale=F),col='red')
        }
      }

      toARMA = function(fit)
      {
        arma_fit = toARMA(fit)
        summary(arma_fit)
      }

      denoise = function(input)
      {
        n = length(input)
        i = 0 
        while(n-2^i>0){i=i+1; y=i}
        n_pad = 2^y - n
        padded = rbind( as.matrix(input), zeros(n_pad,1) )
        h <- daubcqf(6)
        ret.dwt = denoise.dwt(padded, h$h.0)
        out = ret.dwt$xd[1:n]
        out
      }

      """)

# Import functions from R
data_to_R = ro.r['data_to_R']
sys_id = ro.r['sys_id']
model_coefs = ro.r['model_coefs']
model_summary = ro.r['model_summary']
forecast = ro.r['forecast']
denoise = ro.r['denoise']
toARMA = ro.r['toARMA']

# These lines are required before we can covert numpy arrays into R arrays.
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

def fit_armax_homecage(all_data, options, solver='VARX', fit_type="ls", decimate=30, denoise_output=True):
    """
    Fit an ARMAX model to home cage data.
    """
    exp_type = options.exp_type
    for animal_id in all_data.keys():
        # load data from hdf5 file by animal-date-exp_type
        animal = all_data[animal_id]
        for dates in animal.keys():
            # Create figure
#            fig = pl.figure()
#            ax = fig.add_subplot(2,1,1)           
            date = animal[dates]
            FA = FiberAnalyze( options )
            FA.subject_id = animal_id
            FA.exp_date = dates
            FA.exp_type = exp_type
            FA.load(file_type="hdf5")

            # Get fluorescence data (system output) 
            # and optionally denoise
            output = FA.fluor_data[::decimate] 
            n_abs = len(output)
            if denoise_output:
                output = denoise(output)
            n = len(output)
#            output = np.log(output)

            # Get clock times
            times = FA.time_stamps[::decimate]

            # Get event times and convert to input sequence
            # of appropriate magnitude
            input = -1*FA.trigger_data[::decimate] + 1.0 

#            scale_slope = 2
#            scale = scale_slope * np.linspace(1,0,num=len(input))
#            scale = np.ones((len(scale),))
#            slope =  -0.05*np.linspace(0,1,num=len(input))
#            input = scale*input + slope

            # smooth input 
#             window = 'blackman'
#             window_len = 200
#             num_exogenous_lags = 10
#             w=eval('np.'+window+'(window_len)')
#             input=np.convolve(w/w.sum(),input,mode='same')

            if solver == 'arima':
                input = construct_lagged(input, num_exogenous_lags)
                output = output[0:input.shape[0]]

            # Lag input, subtract 0.2 quantile, 
            AICs = []; fits = []

            baseline = mstats.mquantiles(output,[0.05]) 
            output -= baseline
            input -= baseline

            n_lags = 20
            lag = 3
            truncate = n_abs - n_lags*lag

            for i in xrange(n_lags):
                n = len(output)
                output = output[0:(n-lag)] 
                input = input[lag:n]
                print truncate, len(input), len(output)
                
                # Fit model
                try:
                    fits.append(sys_id(input[0:truncate], output[0:truncate], solver=solver))

                    # Plot result
                    sum = model_summary(fits[i],solver=solver)
                    if solver == 'VARX':
                        AIC = sum[0][0][0]
                    else:
                        AIC = sum[0]

                    print "AIC:",AIC
                    AICs.append(AIC)

                except:
                    print "Fit number ", i, "failed."
                    fits.append([None])

            AICs = np.asarray(AICs)
            print "Getting best index..."
            best_idx = np.where(AICs == np.min(AICs))[0]

            print "Forecasting..."
            n = len(input)
            forecast(fits[best_idx[0]], n, solver=solver, input=input, output=output)
            print "Model coefs:", model_coefs(fits[best_idx[0]])
            pl.plot( np.asarray(model_coefs(fits[best_idx[0]]))[5::], 'r-')
            pl.show()
            time.sleep(10)

#            1/0
#            for i in xrange(n):

def construct_lagged(ts, num_lags=10):
    n = len(ts)
    out = np.empty((n-num_lags, num_lags))
    n_out = out.shape[0]
    for i in xrange(num_lags):
        out[:,i] = ts[i:(n_out+i)]
    return out

if __name__ == "__main__":
    # Parse command line options
    from optparse import OptionParser

    parser = OptionParser()

    parser.add_option("-o", "--output-path", dest="output_path", default=None,
                      help="Specify the ouput path.")

    parser.add_option("-t", "--trigger-path", dest="trigger_path", default=None,
                      help="Specify path to files with trigger times, minus the '_s.npz' \
                            and '_e.npz' suffixes.")

    parser.add_option("-i", "--input-path", dest="input_path",
                      help="Specify the input path.")

    parser.add_option("", "--time-range", dest="time_range",default=None,
                      help="Specify a time window over which to analyze the time series in \
                            format start:end. -1 chooses the appropriate extremum")

    parser.add_option('-p', "--plot-type", default = 'tseries', dest="plot_type",
                      help="Type of plot to produce.")

    parser.add_option('', "--fluor_normalization", default = 'deltaF', dest="fluor_normalization",
                      help="Normalization of fluorescence trace. Can be a.u. between [0,1]: \
                            'standardize' or deltaF/F: 'deltaF'.")

    parser.add_option('-s', "--smoothness", default = None, dest="smoothness",
                      help="Should the time series be smoothed, and how much.")

    parser.add_option('-x', "--selectfiles", default = False, dest = "selectfiles",
                       help="Should you select filepaths in a pop window instead of in the \
                             command line.")

    parser.add_option("", "--save-txt", action="store_true", default=False, dest="save_txt",
                      help="Save data matrix out to a text file.")

    parser.add_option("", "--save-to-h5", default=None, dest="save_to_h5",
                      help="Save data matrix to a dataset in an hdf5 file.")

    parser.add_option("", "--save-and-exit", action="store_true", default=False, 
                      dest="save_and_exit", help="Exit immediately after saving data out.")

    parser.add_option("", "--filter-freqs", default=None, dest="filter_freqs",
                      help="Use a notch filter to remove high frequency noise. \
                            Format lowfreq:highfreq.")

    parser.add_option("", "--save-debleach", action="store_true", default=False, 
                      dest="save_debleach", help="Debleach fluorescence time series by fitting \
                            with an exponential curve.")

    parser.add_option('', "--exp-type", default = 'homecagesocial', dest="exp_type",
                      help="Which type of experiment. Current options are 'homecagesocial' and \
                            'homecagenovel'.")

    parser.add_option("", "--time-window", dest="time_window",default='-3:3',
                      help="Specify a time window for peri-event plots in format start:end.")
    
    (options, args) = parser.parse_args()

#    x = range(100)
#    y = construct_lagged(x)
#    1/0

    options.input_path = "/Users/logangrosenick/Dropbox/FiberPhotometry/DATA/all_data_raw.h5"
    all_data = h5py.File(options.input_path,'r')

    fit_armax_homecage(all_data, options)
