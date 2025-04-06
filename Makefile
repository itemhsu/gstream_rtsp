DEEPSTREAM_INC=/opt/nvidia/deepstream/deepstream-7.1/sources/includes
CXX=g++
CXXFLAGS=-g -Wall -Wno-unused-function $(shell pkg-config --cflags gstreamer-1.0 glib-2.0) -I$(DEEPSTREAM_INC)
LDFLAGS=$(shell pkg-config --libs gstreamer-1.0 glib-2.0) -L/opt/nvidia/deepstream/deepstream-7.1/lib/ -lnvds_meta -lnvdsgst_meta -lopencv_core -lopencv_imgcodecs

NVDS_CUSTOM_DIR=nvdsinfer_custom_impl_Yolo

TARGETS=yolov10_crop yolov10_rtsp_crop cmd_rtsp_crop

.PHONY: all clean subdir

all: subdir $(TARGETS)

subdir:
	$(MAKE) -C $(NVDS_CUSTOM_DIR)

%: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	$(MAKE) -C $(NVDS_CUSTOM_DIR) clean
	rm -f $(TARGETS)

