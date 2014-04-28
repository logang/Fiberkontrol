



datapath=/Users/isaackauvar/Dropbox/Fiberkontrol/Fiberkontrol_Data/Lisa_Data

ffmpeg -i 407_social.avi 407_social.mp4
ffmpeg -i 407_social.mp4 -ss 00:02:21 -t 00:00:10 407_clip.mp4

ffmpeg -i 401_social.avi 401_social.mp4
ffmpeg -i 401_social.mp4 -ss 00:02:21 -t 00:00:10 401_clip.mp4

ffmpeg -f concat -i mylist.txt -c copy output.mp4

mylist.txt:
file 407_clip.mp4
file 401_clip.mp4



ffmpeg -f concat -i <(for f in ./*.wav; do echo "file '$f'"; done) -c copy output.wav