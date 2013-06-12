#!/bin/bash

data_path='/Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Results/Cell/all_data_raw_w20130327.h5'
output_path='/Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Results/Cell/Plots/Testing/20130611Presubmission_test'



plot_type=group-regression-plot
metric=peak
time_window='0:2'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagesocial' --intensity-metric=$metric --time-window=$time_window
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagenovel'  --intensity-metric=$metric  --time-window=$time_window
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagesocial'  --intensity-metric=$metric  --time-window=$time_window
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagenovel'  --intensity-metric=$metric  --time-window=$time_window

if $(false); then

plot_type=group-bout-heatmaps
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagesocial'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5 --$plot_type --mouse-type='GC5' --exp-type='homecagenovel'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagesocial'
python group_analysis.py --input-path=$data_path --output-path=$output_path/$plot_type/GC5_NAcprojection --$plot_type --mouse-type='GC5_NAcprojection' --exp-type='homecagenovel'


fi


