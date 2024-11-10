# gstream_rtsp
Test gstreamer pipeline for rtsp with Nvidia GPU

# pipelines
Attempt | Pipeline | function | CPU% 
--- | --- | --- | --- 
1 | ffplay rtsp://168.168.11.23/media.amp?streamprofile=Profile3 | ffplay as rtsp base line | 14.5%
2 |gst-launch-1.0 -v rtspsrc location="rtsp://168.168.11.23/media.amp?streamprofile=Profile3" name=src protocols=tcp latency=0 src. ! application/x-rtp,media=video,encoding-name=H264,payload=98 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink | use software h264 decoder | 18.5%
2 | gst-launch-1.0 -v   rtspsrc location="rtsp://168.168.11.23/media.amp?streamprofile=Profile3" latency=0 !   rtph264depay ! h264parse ! nvh264dec !   cudaconvert ! cudadownload ! videoconvert ! autovideosink | use nvidia hardware decoder | 6.3%
