#include <gst/gst.h>
#include <glib.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "nvdsmeta.h"
#include "gstnvdsmeta.h"

#define MUXER_BATCH_TIMEOUT_USEC 33000

// Bus 回调：用于输出管道消息（例如 EOS 和错误信息）
static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
    GMainLoop *loop = static_cast<GMainLoop *>(data);
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            std::cout << "[INFO] EOS" << std::endl;
            g_main_loop_quit(loop);
            break;
        case GST_MESSAGE_ERROR: {
            GError *err = nullptr;
            gchar *debug = nullptr;
            gst_message_parse_error(msg, &err, &debug);
            std::cerr << "[ERROR] " << (err ? err->message : "Unknown")
                      << ": " << (debug ? debug : "") << std::endl;
            if (err) g_error_free(err);
            if (debug) g_free(debug);
            g_main_loop_quit(loop);
            break;
        }
        default:
            break;
    }
    return TRUE;
}

std::string runLicensePlateRecognition(const cv::Mat &crop) {
    std::cout << "[INFO] runLicensePlateRecognition - start" << std::endl;

    // 模型僅加載一次
    static cv::dnn::Net lpNet = cv::dnn::readNetFromONNX("epoch290.onnx");
    if (lpNet.empty()) {
        std::cerr << "[ERROR] 無法加載 ONNX 模型 epoch290.onnx" << std::endl;
        return "";
    }

    // 確保圖片為 3 通道
    cv::Mat crop_rgb;
    if (crop.channels() == 1) {
        cv::cvtColor(crop, crop_rgb, cv::COLOR_GRAY2BGR);
    } else {
        crop_rgb = crop.clone();
    }

    // 保持比例 resize 並用黑邊填滿到 288x288
    int target_size = 288;
    int w = crop_rgb.cols;
    int h = crop_rgb.rows;
    float scale = static_cast<float>(target_size) / std::max(w, h);
    int new_w = static_cast<int>(w * scale);
    int new_h = static_cast<int>(h * scale);

    cv::Mat resized;
    cv::resize(crop_rgb, resized, cv::Size(new_w, new_h));

    cv::Mat padded = cv::Mat::zeros(target_size, target_size, CV_8UC3);
    int dx = (target_size - new_w) / 2;
    int dy = (target_size - new_h) / 2;
    resized.copyTo(padded(cv::Rect(dx, dy, new_w, new_h)));

    // 產生 blob，注意 size 必須與模型輸入一致
    cv::Mat blob = cv::dnn::blobFromImage(padded, 1.0 / 255.0, cv::Size(target_size, target_size));
    if (blob.empty()) {
        std::cerr << "[ERROR] 生成 blob 失敗" << std::endl;
        return "";
    }

    // 模型推論
    lpNet.setInput(blob);
    cv::Mat result = lpNet.forward();

    // 輸出 shape 檢查
    std::cout << "[DEBUG] result shape: " << result.size << std::endl;

    // 假設這是一個分類模型：取最大值的索引作為輸出類別
    cv::Point classIdPoint;
    double confidence;
    cv::minMaxLoc(result, nullptr, &confidence, nullptr, &classIdPoint);
    int label = classIdPoint.x;

    std::cout << "[INFO] 推論結果：label = " << label << ", confidence = " << confidence << std::endl;
    return std::to_string(label);
}

// GstPad Probe 回调函数：处理每个 buffer 中的帧。
// 对检测到的对象（车牌区域）进行裁剪，并调用第二模型识别车牌。
static GstPadProbeReturn rgb_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    if (!(GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER))
        return GST_PAD_PROBE_OK;

    GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf)
        return GST_PAD_PROBE_OK;

    GstMapInfo map_info;
    if (!gst_buffer_map(buf, &map_info, GST_MAP_READ))
        return GST_PAD_PROBE_OK;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) {
        gst_buffer_unmap(buf, &map_info);
        return GST_PAD_PROBE_OK;
    }

    // 遍历每一帧
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = reinterpret_cast<NvDsFrameMeta *>(l_frame->data);
        // 遍历当前帧中所有检测到的对象
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = reinterpret_cast<NvDsObjectMeta *>(l_obj->data);
            int inf_left   = std::max<int>(0, obj_meta->rect_params.left);
            int inf_top    = std::max<int>(0, obj_meta->rect_params.top);
            int inf_width  = obj_meta->rect_params.width;
            int inf_height = obj_meta->rect_params.height;

            // 调整区域确保不超过帧的边界（假定帧大小为 1920x1080）
            if (inf_left + inf_width > 1920)
                inf_width = 1920 - inf_left;
            if (inf_top + inf_height > 1080)
                inf_height = 1080 - inf_top;
            if (inf_width <= 0 || inf_height <= 0)
                continue;

            // 从整帧数据构造 OpenCV Mat (假定分辨率为1920×1080)
            cv::Mat frame(1080, 1920, CV_8UC3, map_info.data, 1920 * 3);
            // 裁剪检测出的车牌区域
            cv::Rect roi(inf_left, inf_top, inf_width, inf_height);
            cv::Mat objectCrop = frame(roi);

            static int obj_count = 0;
            std::string filename = "object_" + std::to_string(obj_count++) + ".jpg";
            // 可选：保存裁剪的图像
            // cv::imwrite(filename, objectCrop);
            std::cout << "[INFO] 裁剪的车牌区域保存为: " << filename << std::endl;

            // 调用车牌识别函数，对裁剪区域进行第二次模型推断
            std::string lp_result = runLicensePlateRecognition(objectCrop);
            std::cout << "[INFO] 车牌识别结果: " << lp_result << std::endl;
        }
    }

    gst_buffer_unmap(buf, &map_info);
    return GST_PAD_PROBE_OK;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <rtsp_source> <nvinfer_config_file>" << std::endl;
        return -1;
    }

    gst_init(&argc, &argv);

    // 构造 DeepStream 管道字符串，包含 nvstreammux、nvinfer、nvvideoconvert 等元素
    gchar *pipeline_str = g_strdup_printf(
        "nvstreammux name=mux batch-size=1 width=1920 height=1080 live-source=1 batched-push-timeout=%d ! "
        "nvinfer config-file-path=%s ! queue ! "
        "nvvideoconvert name=rgb_conv ! video/x-raw,format=BGR ! "
        "nvvideoconvert name=rgb_conv2 !"
        "nvdsosd name=osd ! nveglglessink ; "
        "rtspsrc location=%s protocols=tcp latency=200 drop-on-latency=true name=src ! queue ! "
        "application/x-rtp,media=video,encoding-name=H264,payload=98 ! rtph264depay ! h264parse !"
        "nvv4l2decoder ! queue ! mux.sink_0",
        MUXER_BATCH_TIMEOUT_USEC, argv[2], argv[1]);

    GstElement *pipeline = gst_parse_launch(pipeline_str, nullptr);
    g_free(pipeline_str);
    if (!pipeline) {
        std::cerr << "[ERROR] 无法创建管道" << std::endl;
        return -1;
    }

    GstBus *bus = gst_element_get_bus(pipeline);
    GMainLoop *loop = g_main_loop_new(nullptr, FALSE);
    gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    // 获取 rgb_conv2 元素并在其 sink pad 上添加 probe
    GstElement *rgb_conv2 = gst_bin_get_by_name(GST_BIN(pipeline), "rgb_conv2");
    if (rgb_conv2) {
        GstPad *pad = gst_element_get_static_pad(rgb_conv2, "sink");
        if (pad) {
            gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, rgb_probe, nullptr, nullptr);
            gst_object_unref(pad);
        } else {
            std::cerr << "[ERROR] 无法获取 rgb_conv2 的 sink pad" << std::endl;
        }
        gst_object_unref(rgb_conv2);
    } else {
        std::cerr << "[ERROR] 无法获取元素 rgb_conv2" << std::endl;
    }

    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_main_loop_run(loop);

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    g_main_loop_unref(loop);
    return 0;
}
