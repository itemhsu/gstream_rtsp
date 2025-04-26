#!/bin/bash

gst-launch-1.0 \
rtspsrc location=rtsp://168.168.11.33/media.amp?streamprofile=Profile1 ! \
rtph264depay ! h264parse ! nvv4l2decoder ! mux.sink_0 \
nvstreammux name=mux batch-size=1 width=1920 height=1080 ! \
nvinfer config-file-path=/home/itemhsu/deepstream/gstream_rtsp/config_infer_primary_yoloLite.txt name=lpd_infer ! \
nvtracker ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so ll-config-file=/home/itemhsu/deepstream/gstream_rtsp/tracker_config.txt ! \
nvdspreprocess config-file=/home/itemhsu/deepstream/gstream_rtsp/preprocess_config.txt ! \
nvinfer config-file-path=/home/itemhsu/deepstream/gstream_rtsp/config_infer_primary_yoloOCR.txt name=lpr_infer ! \
tee name=t ! queue ! nvvideoconvert ! nvdsosd ! nvvideoconvert ! video/x-raw,format=RGB ! jpegenc ! multifilesink location=/home/itemhsu/deepstream/snapshot_%04d.jpg \
t. ! queue ! fakesink name=probe_sink
