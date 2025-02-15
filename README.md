# gstream_rtsp
Test gstreamer pipeline for rtsp with Nvidia GPU

# Summary
Item | Note | Yolov10 w/ 8 RTSP | D-FINE w/ 8 RTSP
--- | --- | --- | ---
CPU | Threads: 12, Model: Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz | 26% | 41%
GPU | Model: NVIDIA GeForce RTX 3060, 12G | 67%, 96W/170W, 2739MB/12288MB | 56%, 81W/170W, 2739MiB/12288MiB
RTSP | 8ch, 640x360, 30FPS | -- | --
Object detection | Yolov10 | D-FINE
OS | Ubuntu 22.04.5 LTS | -- | --
CUDA | Cuda compilation tools, release 12.6, V12.6.77 | --  | --
TensorRT | 10.6.0 | -- | --
PyTorch | 2.0.1 | -- | --
GStreamer | 1.20.3 | -- | --
DeepStream | 7.1 | -- | --

# Files
Name | Purpose
--- | ---
run-yolov10.sh | run gst-launch pipeline
export_yoloV10.py | "python export_yoloV10.py -w yolov10n.pt" to generate onnx from yolo pytorch model
config_infer_primary_yolov10.txt | config the deepstream 
MakeFile | patch for DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
upsample_layer.cpp | patch for DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/layers/upsample_layer.cpp
yolov10n.pt |  yolo pytorch model
-- onnx-file | define model onnx file
-- model-engine-file | define TensorRT model file (will auto generate from onnx if file doesn't exist)
-- custom-lib-path | define post processor lib
-- parse-bbox-func-name | post processing function in custom-lib
-- width | rtsp width
-- height | resp height
-- batch-size | 1 for no batch
-- live-source | 1 for true

# Dependent
URL | Purpose
--- | ---
https://github.com/shashikant-ghangare/DeepStream-Yolo/tree/add-yolov10 | git clone -b add-yolov10 https://github.com/shashikant-ghangare/DeepStream-Yolo.git to get the yolov10 support for deepstream (config and custom lib source)

# Steps
1. Install Ubuntu, GPU driver, CUDA, TensorRT, PyTorch, GStreamer, DeepStream SDK
1. Install RTSP Camera
1. build custom lib with https://github.com/shashikant-ghangare/DeepStream-Yolo/tree/add-yolov10
   * patch DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/Makefile
   * patch DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/layers/upsample_layer.cpp
   * make
1. get onnx model with export_yoloV10.py
1. edit config_infer_primary_yolov10 to fit your enviroment
1. run run-yolov10.sh

# Testing pipelines
Attempt | Pipeline | function | CPU% , GPU%
--- | --- | --- | --- 
1 | ffplay rtsp://168.168.11.23/media.amp?streamprofile=Profile3 | ffplay as rtsp base line | 14.5%
2 |gst-launch-1.0 -v rtspsrc location="rtsp://168.168.11.23/media.amp?streamprofile=Profile3" name=src protocols=tcp latency=0 src. ! application/x-rtp,media=video,encoding-name=H264,payload=98 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink | use software h264 decoder | 18.5%
3 | gst-launch-1.0 -v   rtspsrc location="rtsp://168.168.11.23/media.amp?streamprofile=Profile3" latency=0 !   rtph264depay ! h264parse ! nvh264dec !   cudaconvert ! cudadownload ! videoconvert ! autovideosink | use nvidia hardware decoder | 6.3%
4 | gst-launch-1.0   rtspsrc location=rtsp://168.168.11.23/media.amp?streamprofile=Profile3  protocols=tcp latency=0 name=src !   application/x-rtp,media=video,encoding-name=H264,payload=98 !    rtph264depay ! h264parse ! nvv4l2decoder !   nvvideoconvert ! queue !  mux.sink_0   src. ! application/x-rtp,media=audio ! queue !   fakesink   nvstreammux live-source=1 name=mux batch-size=1 width=640 height=360 !    nvinfer config-file-path=config_infer_primary.txt ! queue !  nvvideoconvert !     nvdsosd ! queue ! nveglglessink | use nvidia inference and live stream | CPU 14.3%, GPU 3%
5 | gst-launch-1.0     rtspsrc location="rtsp://168.168.11.23/media.amp?streamprofile=Profile3"             protocols=tcp latency=0 name=src          src. ! queue               ! application/x-rtp,media=video,encoding-name=H264,payload=98               ! rtph264depay               ! h264parse               ! nvv4l2decoder               ! nvvideoconvert               ! "video/x-raw(memory:NVMM), format=NV12"               ! mux.sink_0          src. ! queue               ! application/x-rtp,media=audio               ! fakesink     nvstreammux name=mux batch-size=1 width=640 height=360 live-source=1    ! nvinfer config-file-path=config_infer_primary.txt     ! nvvideoconvert     ! nvdsosd     ! nveglglessink | use nvv4l2decoder (hw 264 decoder) nvidia inference and live stream | CPU 14.3%, GPU 3%
6 | gst-launch-1.0   rtspsrc location="rtsp://168.168.11.23/media.amp?streamprofile=Profile1" protocols=tcp latency=0 name=src     src. ! queue          ! application/x-rtp,media=video,encoding-name=H264,payload=98          ! rtph264depay          ! h264parse          ! nvv4l2decoder          ! nvvideoconvert          ! "video/x-raw(memory:NVMM), format=NV12, width=640, height=360"          ! mux.sink_0     src. ! queue          ! application/x-rtp,media=audio          ! fakesink   nvstreammux name=mux batch-size=1 width=640 height=360 live-source=1   ! nvinfer config-file-path=config_infer_primary.txt   ! nvvideoconvert   ! nvdsosd   ! nveglglessink | input 1920x640 , gpu infer | CPU 20%, GPU 3% |
7 | gst-launch-1.0 rtspsrc location="rtsp://168.168.11.23/media.amp?streamprofile=Profile1" protocols=tcp latency=0 name=src src. ! queue ! application/x-rtp,media=video,encoding-name=H264,payload=98 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvideoconvert ! "video/x-raw(memory:NVMM), format=NV12" ! mux.sink_0 src. ! queue ! application/x-rtp,media=audio ! fakesink nvstreammux name=mux batch-size=1 width=640 height=360 live-source=1 ! nvinfer config-file-path=config_infer_primary.txt ! nvvideoconvert ! nvdsosd ! nveglglessink | remove scaled down before nvinfer | CPU 20%, GPU3% |
