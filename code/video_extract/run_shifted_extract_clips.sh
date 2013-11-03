directory='/Users/isaackauvar/Documents/2012-2013/ZDlab/FiberKontrol/Results/Cell/Plots/Finalz_including_20130920_reaction_time_370ms/print-spike-times'


output='clipwindow_0_0'
clip_window='0:0'
peak_thresh='0.00'
python extract_clips.py $directory/GC5_NAcprojection/list_of_event_times_GC5_NAcprojection_homecagenovel.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh
python extract_clips.py $directory/GC5_NAcprojection/list_of_event_times_GC5_NAcprojection_homecagesocial.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh
python extract_clips.py $directory/GC5/list_of_event_times_GC5_homecagesocial.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh
python extract_clips.py $directory/GC5/list_of_event_times_GC5_homecagenovel.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh


if $(false); then

output='window_0_0_tt'
clip_window = '0:1'
peak_thresh = '0.00'
python extract_clips.py $directory/GC5_NAcprojection/list_of_event_times_GC5_NAcprojection_homecagenovel.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh
python extract_clips.py $directory/GC5_NAcprojection/list_of_event_times_GC5_NAcprojection_homecagesocial.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh
python extract_clips.py $directory/GC5/list_of_event_times_GC5_homecagesocial.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh
python extract_clips.py $directory/GC5/list_of_event_times_GC5_homecagenovel.pkl -o $output --clip-window=$clip_window --peak-thresh=$peak_thresh






echo you can use the section inside the if statement to comment out commands
echo done
fi