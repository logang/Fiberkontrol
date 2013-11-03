directory='/Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Results/Cell/Plots/Finalz_including_20130920_reaction_time_370ms/print-spike-times'

output='clipwindow_1_0'
clip_window='1:0'
peak_thresh='0.00'
clip_window_origin='interaction_start'
python extract_clips.py $directory/GC5_NAcprojection/list_of_event_times_GC5_NAcprojection_homecagesocial.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin
python extract_clips.py $directory/GC5_NAcprojection/list_of_event_times_GC5_NAcprojection_homecagenovel.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin
python extract_clips.py $directory/GC5/list_of_event_times_GC5_homecagesocial.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin
python extract_clips.py $directory/GC5/list_of_event_times_GC5_homecagenovel.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin

if $(false); then

output='clipwindow_1_1'
clip_window='1:1'
peak_thresh='0.00'
clip_window_origin='peak'
python extract_clips.py $directory/GC5_NAcprojection/list_of_event_times_GC5_NAcprojection_homecagenovel.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin
python extract_clips.py $directory/GC5_NAcprojection/list_of_event_times_GC5_NAcprojection_homecagesocial.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin
python extract_clips.py $directory/GC5/list_of_event_times_GC5_homecagesocial.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin
python extract_clips.py $directory/GC5/list_of_event_times_GC5_homecagenovel.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin



output='clipwindow_0_0'
clip_window='0:0'
peak_thresh='0.00'
clip_window_origin='interaction_start'
python extract_clips.py $directory/GC5_NAcprojection/list_of_event_times_GC5_NAcprojection_homecagenovel.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin
python extract_clips.py $directory/GC5_NAcprojection/list_of_event_times_GC5_NAcprojection_homecagesocial.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin
python extract_clips.py $directory/GC5/list_of_event_times_GC5_homecagesocial.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin
python extract_clips.py $directory/GC5/list_of_event_times_GC5_homecagenovel.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin


output='window_0_0_tt'
clip_window = '0:1'
peak_thresh = '0.00'
clip_window_origin='peak'
python extract_clips.py $directory/GC5_NAcprojection/list_of_event_times_GC5_NAcprojection_homecagenovel.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin
python extract_clips.py $directory/GC5_NAcprojection/list_of_event_times_GC5_NAcprojection_homecagesocial.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin
python extract_clips.py $directory/GC5/list_of_event_times_GC5_homecagesocial.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin
python extract_clips.py $directory/GC5/list_of_event_times_GC5_homecagenovel.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh --clip-window-origin=$clip_window_origin






echo you can use the section inside the if statement to comment out commands
echo done
fi