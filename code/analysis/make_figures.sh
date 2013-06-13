#!/bin/bash

data_path='/Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Results/Cell/all_data_raw_w20130108.h5'
output_path='/Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Results/Cell/Plots/Testing/20130611Presubmission_test'



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
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagesocial'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagenovel' 
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagesocial' 
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagenovel' 

plot_type=compare-decay
max_bout_number=13
intensity_metric='peak'
time_window='0:0'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --max-bout-number=$max_bout_number --intensity-metric=$intensity_metric --time-window=$time_window
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --max-bout-number=$max_bout_number --intensity-metric=$intensity_metric --time-window=$time_window

plot_type=compare-start-and-end-of-epoch
max_bout_number=0
intensity_metric='peak'
time_window='0:.5'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --max-bout-number=$max_bout_number --intensity-metric=$intensity_metric --time-window=$time_window
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --max-bout-number=$max_bout_number --intensity-metric=$intensity_metric --time-window=$time_window

plot_type=compare-epochs
max_bout_number=10
intensity_metric='peak'
time_window='0:0'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --max-bout-number=$max_bout_number --intensity-metric=$intensity_metric --time-window=$time_window
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --max-bout-number=$max_bout_number --intensity-metric=$intensity_metric --time-window=$time_window



if $(false); then

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

plot_type=group-bout-heatmaps
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagesocial'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagenovel'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagesocial'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagenovel'


fi


