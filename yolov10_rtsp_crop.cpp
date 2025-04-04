#include <gst/gst.h>
#include <glib.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "nvdsmeta.h"
#include "gstnvdsmeta.h"

#define MUXER_BATCH_TIMEOUT_USEC 33000

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
    GMainLoop *loop = static_cast<GMainLoop *>(data);
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            std::cout << "EOS" << std::endl;
            g_main_loop_quit(loop);
            break;
        case GST_MESSAGE_ERROR: {
            GError *err = nullptr;
            gchar *debug = nullptr;
            gst_message_parse_error(msg, &err, &debug);
            std::cerr << "Error: " << (err ? err->message : "Unknown")
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

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = reinterpret_cast<NvDsFrameMeta *>(l_frame->data);
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = reinterpret_cast<NvDsObjectMeta *>(l_obj->data);
            int inf_left   = std::max<int>(0, obj_meta->rect_params.left);
            int inf_top    = std::max<int>(0, obj_meta->rect_params.top);
            int inf_width  = obj_meta->rect_params.width;
            int inf_height = obj_meta->rect_params.height;

            std::cout << "inf_left: " << inf_left 
                      << ", inf_top: " << inf_top
                      << ", inf_width: " << inf_width
                      << ", inf_height: " << inf_height << std::endl;

            if (inf_left + inf_width > 1920)
                inf_width = 1920 - inf_left;
            if (inf_top + inf_height > 1080)
                inf_height = 1080 - inf_top;
            if (inf_width <= 0 || inf_height <= 0)
                continue;

            cv::Mat frame(1080, 1920, CV_8UC3, map_info.data, 1920 * 3);
            cv::Rect roi(inf_left, inf_top, inf_width, inf_height);
            cv::Mat objectCrop = frame(roi);

            static int obj_count = 0;
            std::string filename = "object_" + std::to_string(obj_count++) + ".jpg";
            cv::imwrite(filename, objectCrop);
            std::cout << "Saved: " << filename << std::endl;
        }
    }

    gst_buffer_unmap(buf, &map_info);
    return GST_PAD_PROBE_OK;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <h264_file>" << std::endl;
        return -1;
    }

    gst_init(&argc, &argv);

    gchar *pipeline_str = g_strdup_printf(
        "nvstreammux name=mux batch-size=1 width=1920 height=1080  live-source=1  batched-push-timeout=%d ! "
        "nvinfer config-file-path=config_infer_primary_yoloV10.txt ! queue ! "
        "nvvideoconvert name=rgb_conv ! video/x-raw,format=BGR ! "
        "nvvideoconvert name=rgb_conv2 !"
        "nvdsosd name=osd ! nveglglessink ; "
        "rtspsrc location=%s "
        "protocols=tcp latency=200 drop-on-latency=true name=src ! queue ! "
        "application/x-rtp,media=video,encoding-name=H264,payload=98 ! "
        "rtph264depay ! h264parse !"
        "nvv4l2decoder ! queue ! mux.sink_0",
        MUXER_BATCH_TIMEOUT_USEC, argv[1]);

    GstElement *pipeline = gst_parse_launch(pipeline_str, nullptr);
    g_free(pipeline_str);
    if (!pipeline) {
        std::cerr << "Failed to create pipeline" << std::endl;
        return -1;
    }

    GstBus *bus = gst_element_get_bus(pipeline);
    GMainLoop *loop = g_main_loop_new(nullptr, FALSE);
    gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    // Attach probe to the sink pad of "rgb_conv2"
    GstElement *rgb_conv2 = gst_bin_get_by_name(GST_BIN(pipeline), "rgb_conv2");
    if (rgb_conv2) {
        GstPad *pad = gst_element_get_static_pad(rgb_conv2, "sink");
        if (pad) {
            gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, rgb_probe, nullptr, nullptr);
            gst_object_unref(pad);
        } else {
            std::cerr << "Failed to get sink pad from rgb_conv2" << std::endl;
        }
        gst_object_unref(rgb_conv2);
    } else {
        std::cerr << "Failed to get element rgb_conv2" << std::endl;
    }

    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_main_loop_run(loop);

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    g_main_loop_unref(loop);
    return 0;
}
