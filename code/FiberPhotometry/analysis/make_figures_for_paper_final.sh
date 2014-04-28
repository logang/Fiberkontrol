#!/bin/bash

data_path='/Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Results/Cell/all_data_post_SfN_raw.h5'
# output_path='/Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Results/Cell/Plots/Finalz_post_SfN_000rxn'
output_path='/Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Results/Cell/Plots/Finalz_testing_FiberPhotometryFolder'

plot_type='get-blind-data'
python group_analysis.py --input-path=$data_path --output-path=$output_path/ --$plot_type

if $(true); then

plot_type=time-series-animation
exp_type='homecagesocial'
animal_id='421'
exp_date='20121105'
plot_format='.mp4'
output_path='/Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Results/Cell/Videos'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --plot-format=$plot_format --exp-type=$exp_type --animal-id=$animal_id --exp-date=$exp_date

fi

if $(true); then

plot_type=compare-decay
max_bout_number=13
intensity_metric='peak'
time_window='0:0'
plot_format='.pdf'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --max-bout-number=$max_bout_number --intensity-metric=$intensity_metric --time-window=$time_window --plot-format=$plot_format
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --max-bout-number=$max_bout_number --intensity-metric=$intensity_metric --time-window=$time_window --plot-format=$plot_format

plot_type=compare-decay
max_bout_number=8
intensity_metric='peak'
time_window='0:0'
plot_format='.pdf'
exp_type='sucrose'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/sucrose --$plot_type --exp-type=$exp_type --mouse-type='GC5' --max-bout-number=$max_bout_number --intensity-metric=$intensity_metric --time-window=$time_window --plot-format=$plot_format


plot_type=group-bout-heatmaps
time_window='3:3'
plot_format='.pdf'
max_bout_number=15
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagesocial' --time-window=$time_window --plot-format=$plot_format --max-bout-number=$max_bout_number
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagenovel' --time-window=$time_window --plot-format=$plot_format --max-bout-number=$max_bout_number
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagesocial' --time-window=$time_window --plot-format=$plot_format --max-bout-number=$max_bout_number
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagenovel' --time-window=$time_window --plot-format=$plot_format --max-bout-number=$max_bout_number


plot_type=compare-epochs
max_bout_number=10
intensity_metric='peak'
time_window='0:0'
plot_format='.pdf'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --max-bout-number=$max_bout_number --intensity-metric=$intensity_metric --time-window=$time_window --plot-format=$plot_format
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --max-bout-number=$max_bout_number --intensity-metric=$intensity_metric --time-window=$time_window --plot-format=$plot_format

plot_type=fluorescence-histogram
max_bout_number=0
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagesocial' --max-bout-number=$max_bout_number
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagenovel' --max-bout-number=$max_bout_number
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagesocial' --max-bout-number=$max_bout_number
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagenovel' --max-bout-number=$max_bout_number

plot_type=print-spike-times
time_window='0:0'
max_bout_number=15
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagesocial' --time-window=$time_window --max-bout-number=$max_bout_number
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagenovel' --time-window=$time_window  --max-bout-number=$max_bout_number
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagesocial' --time-window=$time_window --max-bout-number=$max_bout_number
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagenovel' --time-window=$time_window --max-bout-number=$max_bout_number

plot_type=compare-epochs
max_bout_number=10
intensity_metric='peak'
time_window='0:0'
plot_format='.pdf'
exp_type='sucrose'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/$exp_type/GC5 --$plot_type --exp-type=$exp_type --mouse-type='GC5' --max-bout-number=$max_bout_number --intensity-metric=$intensity_metric --time-window=$time_window --plot-format=$plot_format
intensity_metric='average'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/$exp_type/GC5 --$plot_type --exp-type=$exp_type --mouse-type='GC5' --max-bout-number=$max_bout_number --intensity-metric=$intensity_metric --time-window=$time_window --plot-format=$plot_format


plot_type=group-plot-time-series
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='sucrose'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='EYFP' --exp-type='homecagesocial'

plot_type=group-regression-plot
metric=event_index
time_window='0:0'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagesocial' --intensity-metric=$metric --time-window=$time_window
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagenovel'  --intensity-metric=$metric  --time-window=$time_window
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagesocial'  --intensity-metric=$metric  --time-window=$time_window
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagenovel'  --intensity-metric=$metric  --time-window=$time_window

plot_type=group-regression-plot
metric=event_time
time_window='0:0'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagesocial' --intensity-metric=$metric --time-window=$time_window
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagenovel'  --intensity-metric=$metric  --time-window=$time_window
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagesocial'  --intensity-metric=$metric  --time-window=$time_window
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagenovel'  --intensity-metric=$metric  --time-window=$time_window

plot_type=group-regression-plot
metric=average
time_window='0:0'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagesocial' --intensity-metric=$metric --time-window=$time_window
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagenovel'  --intensity-metric=$metric  --time-window=$time_window
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagesocial'  --intensity-metric=$metric  --time-window=$time_window
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagenovel'  --intensity-metric=$metric  --time-window=$time_window


plot_type=group-regression-plot
metric=peak
time_window='0:0'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagesocial' --intensity-metric=$metric --time-window=$time_window
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagenovel'  --intensity-metric=$metric  --time-window=$time_window
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagesocial'  --intensity-metric=$metric  --time-window=$time_window
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagenovel'  --intensity-metric=$metric  --time-window=$time_window


plot_type=event-length-histogram
max_bout_number=0
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagesocial' --max-bout-number=$max_bout_number
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagenovel' --max-bout-number=$max_bout_number
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagesocial' --max-bout-number=$max_bout_number
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagenovel' --max-bout-number=$max_bout_number

plot_type=compare-start-and-end-of-epoch
max_bout_number=0
intensity_metric='peak'
time_window='0:.05'
echo 'The time window used to be 0:.5, but this was not as robust.'
plot_format='.pdf'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --max-bout-number=$max_bout_number --intensity-metric=$intensity_metric --time-window=$time_window --plot-format=$plot_format
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --max-bout-number=$max_bout_number --intensity-metric=$intensity_metric --time-window=$time_window --plot-format=$plot_format

plot_type=plot-representative-time-series
representative_time_series_specs_file='representative_time_series_specs.txt'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagesocial' --representative-time-series-specs-file=$representative_time_series_specs_file
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagenovel'  --representative-time-series-specs-file=$representative_time_series_specs_file
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagesocial' --representative-time-series-specs-file=$representative_time_series_specs_file
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagenovel' --representative-time-series-specs-file=$representative_time_series_specs_file

plot_type=group-plot-time-series
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagesocial'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagenovel' 
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagesocial' 
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagenovel' 

plot_type=plot-representative-time-series
representative_time_series_specs_file='representative_time_series_specs.txt'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagesocial' --representative-time-series-specs-file=$representative_time_series_specs_file
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagenovel'  --representative-time-series-specs-file=$representative_time_series_specs_file
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagesocial' --representative-time-series-specs-file=$representative_time_series_specs_file
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagenovel' --representative-time-series-specs-file=$representative_time_series_specs_file


echo you can use the section inside the if statement to comment out commands
echo done
fi


