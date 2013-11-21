#/bin/bash

divider_image=black.png
num_seconds=2
output=black_2.mp4

ffmpeg -loop 1 -i $divider_image -c:v libx264 -t $num_seconds -pix_fmt yuv420p -y $output
