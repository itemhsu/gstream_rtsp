 GST_DEBUG=3 gst-launch-1.0   rtspsrc location=rtsp://168.168.11.23/media.amp?streamprofile=Profile3 \
 protocols=tcp latency=200  drop-on-latency=true name=src ! queue ! \
 application/x-rtp,media=video,encoding-name=H264,payload=98 !   \
 rtph264depay ! h264parse ! video/x-h264,stream-format=byte-stream,alignment=au ! nvv4l2decoder  !   nvvideoconvert ! \
 mux.sink_0   src. ! application/x-rtp,media=audio ! queue !  \
 fakesink   nvstreammux name=mux batch-size=1 width=640 height=360  live-source=1 !   \
 nvinfer config-file-path=/home/itemhsu/deepstream/rtsp/config_infer_primary_DFINE.txt !   nvvideoconvert !    \
 nvdsosd ! nveglglessink

