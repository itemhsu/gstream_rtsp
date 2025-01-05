# gstream_rtsp
Test gstreamer pipeline for rtsp with Nvidia GPU

# pipelines
Attempt | Pipeline | function | CPU% , GPU%
--- | --- | --- | --- 
1 | ffplay rtsp://168.168.11.23/media.amp?streamprofile=Profile3 | ffplay as rtsp base line | 14.5%
2 |gst-launch-1.0 -v rtspsrc location="rtsp://168.168.11.23/media.amp?streamprofile=Profile3" name=src protocols=tcp latency=0 src. ! application/x-rtp,media=video,encoding-name=H264,payload=98 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink | use software h264 decoder | 18.5%
3 | gst-launch-1.0 -v   rtspsrc location="rtsp://168.168.11.23/media.amp?streamprofile=Profile3" latency=0 !   rtph264depay ! h264parse ! nvh264dec !   cudaconvert ! cudadownload ! videoconvert ! autovideosink | use nvidia hardware decoder | 6.3%
4 | gst-launch-1.0   rtspsrc location=rtsp://168.168.11.23/media.amp?streamprofile=Profile3  protocols=tcp latency=0 name=src !   application/x-rtp,media=video,encoding-name=H264,payload=98 !    rtph264depay ! h264parse ! nvv4l2decoder !   nvvideoconvert ! queue !  mux.sink_0   src. ! application/x-rtp,media=audio ! queue !   fakesink   nvstreammux live-source=1 name=mux batch-size=1 width=640 height=360 !    nvinfer config-file-path=config_infer_primary.txt ! queue !  nvvideoconvert !     nvdsosd ! queue ! nveglglessink | use nvidia inference and live stream | CPU 14.3%, GPU 3%
5 | gst-launch-1.0     rtspsrc location="rtsp://168.168.11.23/media.amp?streamprofile=Profile3"             protocols=tcp latency=0 name=src          src. ! queue               ! application/x-rtp,media=video,encoding-name=H264,payload=98               ! rtph264depay               ! h264parse               ! nvv4l2decoder               ! nvvideoconvert               ! "video/x-raw(memory:NVMM), format=NV12"               ! mux.sink_0          src. ! queue               ! application/x-rtp,media=audio               ! fakesink     nvstreammux name=mux batch-size=1 width=640 height=360 live-source=1    ! nvinfer config-file-path=config_infer_primary.txt     ! nvvideoconvert     ! nvdsosd     ! nveglglessink | use nvidia inference and live stream | CPU 14.3%, GPU 3%
