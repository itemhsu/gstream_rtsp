#!/usr/bin/env python3

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds
import argparse

def load_label_map(file_path, debug=False):
    """从 lpr_labels.txt 加载标签映射"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f if line.strip()]
        if debug:
            print(f"Loaded {len(labels)} labels from {file_path}: {labels}")
        return labels
    except Exception as e:
        print(f"Error loading lpr_labels.txt: {e}")
        return []

def pad_probe_callback(pad, info, user_data):
    label_map, debug = user_data
    buffer = info.get_buffer()
    if not buffer:
        if debug:
            print("No buffer available")
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
    if not batch_meta:
        if debug:
            print("No batch metadata available")
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        frame_number = frame_meta.frame_num
        source_id = frame_meta.source_id
        if debug:
            print(f"\nFrame {frame_number}, Source ID: {source_id}")

        # 收集所有对象
        objects = []
        l_obj = frame_meta.obj_meta_list
        while l_obj:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            objects.append(obj_meta)
            l_obj = l_obj.next

        # 按 left 坐标排序
        objects.sort(key=lambda obj: obj.rect_params.left)

        # 处理排序后的对象
        lpr_chars = []
        for obj_meta in objects:
            class_id = obj_meta.class_id
            confidence = obj_meta.confidence
            object_id = obj_meta.object_id
            rect_params = obj_meta.rect_params
            bbox_left = rect_params.left
            bbox_top = rect_params.top
            bbox_width = rect_params.width
            bbox_height = rect_params.height

            if debug:
                print(f"  Object ID: {object_id}, Class ID: {class_id}, Confidence: {confidence:.3f}")
                print(f"  Bounding Box: (left: {bbox_left:.1f}, top: {bbox_top:.1f}, width: {bbox_width:.1f}, height: {bbox_height:.1f})")

            # 跳过车牌对象（width > 50 或 height > 50）
            if bbox_width > 50 or bbox_height > 50:
                if debug:
                    print("    Skipping license plate object")
                continue

            # 映射 Class ID 到字符
            if 0 <= class_id < len(label_map):
                char = label_map[class_id]
                lpr_chars.append((bbox_left, char))
            elif debug:
                print(f"    Invalid Class ID: {class_id}, label_map size: {len(label_map)}")

            # 检查 LPR 分类元数据
            l_classifier = obj_meta.classifier_meta_list
            if not l_classifier:
                if debug:
                    print("    No classifier metadata (LPR results) available")
            else:
                while l_classifier:
                    classifier_meta = pyds.NvDsClassifierMeta.cast(l_classifier.data)
                    classifier_id = classifier_meta.unique_component_id
                    if debug:
                        print(f"    Classifier ID: {classifier_id}")

                    l_label = classifier_meta.label_info_list
                    if not l_label and debug:
                        print("      No label info available")
                    while l_label:
                        label_info = pyds.NvDsLabelInfo.cast(l_label.data)
                        if label_info.result_label and len(label_info.result_label) > 1:
                            if debug:
                                print(f"      LPR Label (full string): {label_info.result_label}")
                        else:
                            class_id = label_info.result_class_id
                            if 0 <= class_id < len(label_map):
                                if debug:
                                    print(f"      LPR Label: {label_map[class_id]}, Class ID: {class_id}")
                            elif debug:
                                print(f"      Invalid LPR Class ID: {class_id}")
                        l_label = l_label.next
                    l_classifier = l_classifier.next

        # 打印按 left 排序的 LPR 文本
        if lpr_chars:
            lpr_chars.sort(key=lambda x: x[0])  # 确保按 left 排序
            lpr_text = ''.join(char for _, char in lpr_chars)
            print(f"LPR Text: {lpr_text}")

        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK

def main():
    parser = argparse.ArgumentParser(description="DeepStream pipeline with output mode selection")
    parser.add_argument('--output', choices=['save', 'display'], default='display',
                        help="Output mode: 'save' to save JPEG images, 'display' to show on screen")
    parser.add_argument('--debug', action='store_true',
                        help="Enable debug output (frame and object details)")
    args = parser.parse_args()

    label_map = load_label_map('/home/itemhsu/deepstream/gstream_rtsp/lpr_labels.txt', args.debug)

    Gst.init(None)

    pipeline_base = (
        "rtspsrc location=rtsp://168.168.11.33/media.amp?streamprofile=Profile1 ! "
        "rtph264depay ! h264parse ! nvv4l2decoder ! mux.sink_0 "
        "nvstreammux name=mux batch-size=1 width=1920 height=1080 ! "
        "nvinfer config-file-path=/home/itemhsu/deepstream/gstream_rtsp/config_infer_primary_yoloLite.txt name=lpd_infer ! "
        "nvtracker ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so "
        "ll-config-file=/home/itemhsu/deepstream/gstream_rtsp/tracker_config.txt ! "
        "nvdspreprocess config-file=/home/itemhsu/deepstream/gstream_rtsp/preprocess_config.txt ! "
        "nvinfer config-file-path=/home/itemhsu/deepstream/gstream_rtsp/config_infer_primary_yoloOCR.txt name=lpr_infer ! "
        "tee name=t ! queue ! nvvideoconvert ! nvdsosd ! "
    )

    if args.output == 'save':
        output_branch = (
            "nvvideoconvert ! video/x-raw,format=RGB ! jpegenc ! "
            "multifilesink location=/home/itemhsu/deepstream/snapshot_%04d.jpg"
        )
    else:
        output_branch = "nveglglessink"

    pipeline_str = (
        f"{pipeline_base} {output_branch} "
        "t. ! queue ! fakesink name=probe_sink"
    )

    try:
        pipeline = Gst.parse_launch(pipeline_str)
    except GLib.Error as e:
        print(f"Failed to parse pipeline: {e}")
        return

    fakesink = pipeline.get_by_name("probe_sink")
    if not fakesink:
        print("Error: Could not find probe_sink")
        return
    sink_pad = fakesink.get_static_pad("sink")
    sink_pad.add_probe(Gst.PadProbeType.BUFFER, pad_probe_callback, (label_map, args.debug))

    pipeline.set_state(Gst.State.PLAYING)
    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("Stopping pipeline...")
    finally:
        pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()