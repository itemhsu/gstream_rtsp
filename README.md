# DeepStream 車牌辨識應用程式說明

## 概述

本應用程式基於 NVIDIA DeepStream SDK，實現即時車牌檢測（License Plate Detection, LPD）與車牌辨識（License Plate Recognition, LPR）。應用程式從 RTSP 視訊流中檢測車牌，並辨識車牌文字（如 `BCN2330`）。支援兩種輸出模式：
- **簡潔模式**：僅輸出車牌文字（如 `LPR Text: BCN2330`）。
- **除錯模式**：輸出詳細資訊，包括幀編號、物件 ID、類別 ID、邊界框等。

應用程式使用兩個 `nvinfer` 元件：
1. **LPD**：檢測車牌位置，生成邊界框。
2. **LPR**：辨識車牌中的文字，生成字符序列。

輸出可儲存為 JPEG 圖像或顯示於螢幕，且支援按 `left` 座標排序字符以確保正確的車牌文字順序。

## 功能

- **車牌檢測**：
  - 使用 YOLO 模型（`LPD_yoloV10.onnx`）檢測車牌。
  - 輸出邊界框（例如 `left: 475.6, width: 146.5, height: 63.6`）。

- **車牌辨識**：
  - 使用 YOLO 模型（`epoch290.onnx`）辨識車牌文字。
  - 按 `left` 座標排序字符，映射至 `lpr_labels.txt`，生成單行文字（如 `LPR Text: BCN2330`）。
  - 過濾大邊界框（`width > 50` 或 `height > 50`）以排除車牌物件。

- **輸出控制**：
  - **簡潔模式**（預設）：僅輸出 `LPR Text: BCN2330`。
  - **除錯模式**（啟用 `--debug`）：輸出幀資訊、物件詳細資訊及分類元數據狀態。

- **輸出選項**：
  - **儲存**：將結果儲存為 JPEG 圖像（`/home/itemhsu/deepstream/snapshot_%04d.jpg`）。
  - **顯示**：於螢幕顯示即時視訊（使用 `nveglglessink`）。

- **配置靈活性**：
  - 支援自訂 `lpr_labels.txt` 映射字符（如 `0, 1, 2, 3, ..., A, B, C, ..., N`）。
  - 支援自訂模型和解析器（`nvinfer_custom_yolov10_parser.so`）。

## 系統要求

- **硬體**：
  - NVIDIA GPU（支援 TensorRT 和 DeepStream）。
  - 至少 8GB RAM。

- **作業系統**：
  - Ubuntu 22.04。

- **軟體依賴**：
  - NVIDIA DeepStream SDK 7.1。
  - GStreamer 1.0。
  - Python 3.8+（用於安裝腳本或測試）。
  - NVIDIA 驅動程式和 CUDA。

- **檔案需求**：
  - 模型檔案：
    - `/home/itemhsu/deepstream/gstream_rtsp/LPD_yoloV10.onnx`（LPD 模型）。
    - `/home/itemhsu/deepstream/gstream_rtsp/epoch290.onnx`（LPR 模型）。
  - 配置文件：
    - `/home/itemhsu/deepstream/gstream_rtsp/config_infer_primary_yoloLite.txt`。
    - `/home/itemhsu/deepstream/gstream_rtsp/config_infer_primary_yoloOCR.txt`。
    - `/home/itemhsu/deepstream/gstream_rtsp/preprocess_config.txt`。
    - `/home/itemhsu/deepstream/gstream_rtsp/tracker_config.txt`。
  - 標籤檔案：
    - `/home/itemhsu/deepstream/gstream_rtsp/lpr_labels.txt`（例如 `0, 1, 2, 3, ..., A, B, C, ..., N`）。
  - 自訂解析器：
    - `/home/itemhsu/deepstream/gstream_rtsp/nvinfer_custom_yolov10_parser.so`。

## 安裝指南

1. **安裝 DeepStream SDK**：
   ```bash
   sudo apt-get install nvidia-deepstream-7.1
   ```

2. **安裝 GStreamer**：
   ```bash
   sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav
   ```

3. **安裝 Python 依賴（可選，用於測試）**：
   ```bash
   sudo apt-get install python3-gi python3-pip
   pip3 install pyds
   ```
   確保 `pyds` 位於 `/opt/nvidia/deepstream/deepstream/lib`。

4. **準備模型和配置文件**：
   - 放置於 `/home/itemhsu/deepstream/gstream_rtsp/`。
   - 準備 `lpr_labels.txt`。
   - 配置 `config_infer_primary_yoloLite.txt` 和 `config_infer_primary_yoloOCR.txt`（見下方範例）。

5. **編譯自訂解析器**：
   - 確保 `nvinfer_custom_yolov10_parser.so` 已編譯並位於 `/home/itemhsu/deepstream/gstream_rtsp/`。
   - 示例解析器程式碼：
     ```c
     NvDsInferStatus parse_yolo_ocr_output(NvDsInferLayerInfo *layers, /* 其他參數 */) {
         char lpr_text[] = "BCN2330";
         NvDsClassifierMeta *classifier_meta = nvds_acquire_classifier_meta();
         classifier_meta->unique_component_id = 2;
         NvDsLabelInfo *label_info = nvds_acquire_label_info();
         strncpy(label_info->result_label, lpr_text, MAX_LABEL_SIZE);
         nvds_add_label_info_to_classifier_meta(classifier_meta, label_info);
         nvds_add_classifier_meta_to_object(obj_meta, classifier_meta);
         return NVDSINFER_SUCCESS;
     }
     ```

## 配置範例

### `config_infer_primary_yoloLite.txt`
### `config_infer_primary_yoloOCR.txt`
### `preprocess_config.txt`
### `tracker_config.txt`
### `lpd_labels.txt`
```
license_plate
```

### `lpr_labels.txt`
```
0
1
2
3
...
A
B
C
...
N
...
```

## 使用方法

1. **準備環境**：
   - 確保 RTSP 流可存取：
     ```bash
     ffplay rtsp://168.168.11.33/media.amp?streamprofile=Profile1
     ```

2. **執行應用程式**：
   - **簡潔模式**（僅輸出 `LPR Text`）：
     ```bash
     deepstream-app -c /path/to/config.txt --output save
     ```
     輸出：
     ```
     LPR Text: BCN2330
     ```
   - **除錯模式**：
     ```bash
     deepstream-app -c /path/to/config.txt --output save --debug
     ```
     輸出：
     ```
     Loaded 36 labels from /home/itemhsu/deepstream/gstream_rtsp/lpr_labels.txt: ['0', '1', '2', '3', ..., 'A', 'B', 'C', ..., 'N', ...]
     Frame 246, Source ID: 0
       Object ID: 0, Class ID: 0, Confidence: 0.790
       Bounding Box: (left: 475.6, top: 1001.8, width: 146.5, height: 63.6)
         Skipping license plate object
       Object ID: 18446744073709551615, Class ID: 11, Confidence: 0.814
       Bounding Box: (left: 486.4, top: 1025.2, width: 18.5, height: 28.5)
         No classifier metadata (LPR results) available
       ...
       LPR Text: BCN2330
     ```

3. **輸出選項**：
   - **儲存 JPEG**：圖像儲存於 `/home/itemhsu/deepstream/snapshot_%04d.jpg`。
   - **螢幕顯示**：
     ```bash
     deepstream-app -c /path/to/config.txt --output display
     ```

4. **啟用詳細日誌**：
   ```bash
   GST_DEBUG=3,nvinfer:6,nvdspreprocess:6 deepstream-app -c /path/to/config.txt
   ```

## 除錯與常見問題

### 問題 1：無 `LPR Text` 輸出
- **原因**：第二 `nvinfer` 未生成 `NvDsClassifierMeta`。
- **解決方法**：
  - 檢查 `config_infer_primary_yoloOCR.txt`：
    - 確保 `process-mode=2`。
    - 確認 `operate-on-gie-id=1 operate-on-class-ids=0`。
  - 驗證 `yoloOCR.trt` 模型輸出字符序列。
  - 在解析器中添加日誌：
    ```c
    printf("LPR Parsed Text: %s\n", lpr_text);
    ```

### 問題 2：第一 `nvinfer` 檢測字符而非車牌
- **原因**：`config_infer_primary_yoloLite.txt` 中 `num-detected-classes` 不正確。
- **解決方法**：
  - 將 `num-detected-classes` 設為 `1`。
  - 確保 `lpd_labels.txt` 僅包含 `license_plate`。

### 問題 3：ROI 裁剪失敗
- **原因**：`preprocess_config.txt` 配置錯誤。
- **解決方法**：
  - 嘗試：
    ```ini
    processing-width=640
    processing-height=640
    ```
  - 保存 ROI 圖像：
    ```c
    save_image_to_file(roi_data, "/home/itemhsu/deepstream/roi_images/roi_%d.jpg", frame_id);
    ```

### 問題 4：物件 ID 異常（`18446744073709551615`）
- **原因**：`nvtracker` 配置錯誤。
- **解決方法**：
  - 檢查 `tracker_config.txt` 和 `nvtracker_config.yml`。
  - 臨時禁用 `nvtracker`：
    ```bash
    ... ! nvinfer name=lpd_infer ! nvdspreprocess ! ...
    ```

## 未來改進

- **增強 LPR 模型**：使用更精確的字符辨識模型。
- **多語言支援**：支援不同地區的車牌格式。
- **即時性能優化**：減少延遲，提高 FPS。
- **GUI 介面**：開發圖形介面顯示車牌和文字。

## 參考資料

- [NVIDIA DeepStream SDK 文件](https://docs.nvidia.com/metropolis/deepstream/dev-guide/)
- [GStreamer 文件](https://gstreamer.freedesktop.org/documentation/)
- [DeepStream Python 綁定](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)

----
  

# gstream_rtsp
Test gstreamer pipeline for rtsp with Nvidia GPU, and yolov10 d-fine

# Summary
Item | Note | Yolov10 w/ 8 RTSP | D-FINE w/ 8 RTSP
--- | --- | --- | ---
CPU | Threads: 12, Model: Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz | 26% | 21%
GPU | Model: NVIDIA GeForce RTX 3060, 12G | 67%, 96W/170W, 2739MB/12288MB | 56%, 81W/170W, 2739MiB/12288MiB
RTSP | 8ch, 640x360, 30FPS | -- | --
Object detection | -- | Yolov10 | D-FINE
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
yolov10_crop.cpp | use yolov10 and cropping the detected object with c++
Makefile.crop | make file for yolov10_crop.cpp
runme.sh | example commands for crop and rtsp

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

# Cropping pipelines
Node | Purpose
--- | ---
nvstreammux name=mux batch-size=1 width=1920 height=1080 batched-push-timeout=%d ! | mux 子branch
nvinfer config-file-path=dstest2_pgie_config.txt ! | inference 點，吃config_infer_primary_yoloV10 參數
queue ! | 重要，不然會出錯
nvvideoconvert name=rgb_conv ! video/x-raw,format=BGR ! | 先把YUV轉BGR 才能存jpg檔
nvvideoconvert name=rgb_conv2 ! | 轉回YUV，不用指定，自動辨認
nvdsosd name=osd ! | 虛擬顯示畫面
nveglglessink ; | 顯示硬體
filesrc location=%s ! | 主要的branch
h264parse ! | 264 通訊翻譯
nvv4l2decoder ! | 264 解壓縮 
queue ! mux.sink_0 | 連接到子branch



