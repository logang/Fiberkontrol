spike_times_path='/Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Results/Cell/Plots/Finalz_post_SfN_000rxn/print-spike-times'
video_data_path='/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/'
start_times_file='/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data/video_start_times_precise.txt'
time_series_data_path='/Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Results/Cell/all_data_post_SfN_raw.h5'
output_directory='/Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Results/Cell/Videos/'
data_paths=$video_data_path' '$start_times_file' '$time_series_data_path' '$spike_times_path


output_folder='clipwindow_3_3_norm_post_sfn'
output=$output_directory$output_folder
clip_window='3:3'
peak_thresh='0.00'
clip_window_origin='interaction_start'
exp_type='homecagesocial'
animal_id='402'
mouse_type='GC5_NAcprojection'
python extract_clips.py $data_paths/$mouse_type/list_of_event_times_$mouse_type'_homecagenovel.pkl' -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin --mouse-type=$mouse_type --exp-type=$exp_type --animal-id=$animal_id
python extract_clips.py $data_paths/$mouse_type/list_of_event_times_$mouse_type'_homecagesocial.pkl' -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin --mouse-type=$mouse_type --exp-type=$exp_type --animal-id=$animal_id
mouse_type='GC5'
python extract_clips.py $data_paths/$mouse_type/list_of_event_times_$mouse_type'_homecagesocial.pkl' -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin --mouse-type=$mouse_type --exp-type=$exp_type --animal-id=$animal_id
python extract_clips.py $data_paths/$mouse_type/list_of_event_times_$mouse_type'_homecagenovel.pkl' -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin --mouse-type=$mouse_type --exp-type=$exp_type --animal-id=$animal_id


if $(false); then

echo 'for running a single trial'
output_folder='clipwindow_3_3'
output=$output_directory$output_folder
clip_window='3:3'
peak_thresh='0.00'
clip_window_origin='interaction_start'
exp_type='homecagesocial'
animal_id='421'
exp_date='20121105'
mouse_type='GC5_NAcprojection'
python extract_clips.py $data_paths/$mouse_type/list_of_event_times_$mouse_type'_homecagesocial.pkl' -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin --mouse-type=$mouse_type --exp-type=$exp_type --animal-id=$animal_id --exp-date=$exp_date
python extract_clips.py $data_paths/$mouse_type/list_of_event_times_$mouse_type'_homecagenovel.pkl' -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin --mouse-type=$mouse_type --exp-type=$exp_type --animal-id=$animal_id --exp-date=$exp_date
mouse_type='GC5'
python extract_clips.py $data_paths/$mouse_type/list_of_event_times_$mouse_type'_homecagesocial.pkl' -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin --mouse-type=$mouse_type --exp-type=$exp_type --animal-id=$animal_id --exp-date=$exp_date
python extract_clips.py $data_paths/$mouse_type/list_of_event_times_$mouse_type'_homecagenovel.pkl' -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin --mouse-type=$mouse_type --exp-type=$exp_type --animal-id=$animal_id --exp-date=$exp_date





output_folder='clipwindow_1_0'
output=$output_directory$output_folder
clip_window='1:0'
peak_thresh='0.00'
clip_window_origin='interaction_start'
python extract_clips.py $data_paths/GC5_NAcprojection/list_of_event_times_GC5_NAcprojection_homecagesocial.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin
python extract_clips.py $data_paths/GC5_NAcprojection/list_of_event_times_GC5_NAcprojection_homecagenovel.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin
python extract_clips.py $data_paths/GC5/list_of_event_times_GC5_homecagesocial.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin
python extract_clips.py $data_paths/GC5/list_of_event_times_GC5_homecagenovel.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin



output_folder='clipwindow_0_1_for_lisa_final'
output=$output_directory$output_folder
clip_window='0:1'
peak_thresh='0.00'
clip_window_origin='peak'
divider_clip='/Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Fiberkontrol/code/video_extract/black.mp4'
python extract_clips.py $data_paths/GC5_NAcprojection/list_of_event_times_GC5_NAcprojection_homecagesocial.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin --divider-clip=$divider_clip
python extract_clips.py $data_paths/GC5_NAcprojection/list_of_event_times_GC5_NAcprojection_homecagenovel.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin --divider-clip=$divider_clip
python extract_clips.py $data_paths/GC5/list_of_event_times_GC5_homecagesocial.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin --divider-clip=$divider_clip
python extract_clips.py $data_paths/GC5/list_of_event_times_GC5_homecagenovel.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin --divider-clip=$divider_clip



output_folder='clipwindow_1_1'
output=$output_directory$output_folder
clip_window='1:1'
peak_thresh='0.00'
clip_window_origin='peak'
python extract_clips.py $data_paths/GC5_NAcprojection/list_of_event_times_GC5_NAcprojection_homecagenovel.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin
python extract_clips.py $data_paths/GC5_NAcprojection/list_of_event_times_GC5_NAcprojection_homecagesocial.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin
python extract_clips.py $data_paths/GC5/list_of_event_times_GC5_homecagesocial.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin
python extract_clips.py $data_paths/GC5/list_of_event_times_GC5_homecagenovel.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin



output_folder='clipwindow_0_0'
output=$output_directory$output_folder
clip_window='0:0'
peak_thresh='0.00'
clip_window_origin='interaction_start'
python extract_clips.py $data_paths/GC5_NAcprojection/list_of_event_times_GC5_NAcprojection_homecagenovel.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin
python extract_clips.py $data_paths/GC5_NAcprojection/list_of_event_times_GC5_NAcprojection_homecagesocial.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin
python extract_clips.py $data_paths/GC5/list_of_event_times_GC5_homecagesocial.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin
python extract_clips.py $data_paths/GC5/list_of_event_times_GC5_homecagenovel.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin


output_folder='window_0_0_tt'
output=$output_directory$output_folder
clip_window = '0:1'
peak_thresh = '0.00'
clip_window_origin='peak'
python extract_clips.py $data_paths/GC5_NAcprojection/list_of_event_times_GC5_NAcprojection_homecagenovel.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin
python extract_clips.py $data_paths/GC5_NAcprojection/list_of_event_times_GC5_NAcprojection_homecagesocial.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin
python extract_clips.py $data_paths/GC5/list_of_event_times_GC5_homecagesocial.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin
python extract_clips.py $data_paths/GC5/list_of_event_times_GC5_homecagenovel.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin



echo you can use the section inside the if statement to comment out commands
echo done
fi