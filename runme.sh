#ffmpeg -i  ~/Downloads/2165-155327596_tiny.mp4 -c:v libx264  -movflags +faststart 2165-155327596_tiny.h264

python sub_deepstream_test_2.py 2165-155327596_tiny.h264

# remove sgie2
# python noSgie2_sub_deepstream_test_2.py 2165-155327596_tiny.h264

# remove sgie1
# python 1_noSgie2_sub_deepstream_test_2.py 2165-155327596_tiny.h264

# remove tracker
# python trackNo_1_noSgie2_sub_deepstream_test_2.py 2165-155327596_tiny.h264

#play with recorded h264 from amtk rtsp
#python trackNo_1_noSgie2_sub_deepstream_test_2.py amtkRtsp.h264

#run c++
#link lib :  /opt/nvidia/deepstream/deepstream-7.1/lib/libnvdsgst_meta.so
source ./.bashrc
./rgb_crop_parse_clean_trackNo_1_noSgie2_sub_deepstream_test_2 2165-155327596_tiny.h264

#C++ crop rtsp
./yolov10_rtsp_crop "rtsp://168.168.11.23/media.amp?streamprofile=Profile1"

#C++ cmd crop rtsp
./cmd_rtsp_crop "rtsp://168.168.11.23/media.amp?streamprofile=Profile1" "config_infer_primary_yoloV10.txt"



# yolo rtsp 
python3 yolo_deepstream_test_3.py -i "rtsp://168.168.11.23/media.amp?streamprofile=Profile3"

#IMPORTANT disable proxy, it will cause rtsp error when running with python
export NO_PROXY=*
