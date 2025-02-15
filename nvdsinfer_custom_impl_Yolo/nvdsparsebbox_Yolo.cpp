/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */
#include <iostream>
#include <iomanip>
#include "nvdsinfer_custom_impl.h"

#include "utils.h"

extern "C" bool
NvDsInferParseYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool
NvDsInferParseYoloE(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool
NvDsInferParseDFINE(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList);


static NvDsInferParseObjectInfo
convertBBox(const float& bx1, const float& by1, const float& bx2, const float& by2, const uint& netW, const uint& netH)
{
  NvDsInferParseObjectInfo b;

  float x1 = bx1;
  float y1 = by1;
  float x2 = bx2;
  float y2 = by2;

  x1 = clamp(x1, 0, netW);
  y1 = clamp(y1, 0, netH);
  x2 = clamp(x2, 0, netW);
  y2 = clamp(y2, 0, netH);

  b.left = x1;
  b.width = clamp(x2 - x1, 0, netW);
  b.top = y1;
  b.height = clamp(y2 - y1, 0, netH);

  return b;
}

static void
addBBoxProposal(const float bx1, const float by1, const float bx2, const float by2, const uint& netW, const uint& netH,
    const int maxIndex, const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
  NvDsInferParseObjectInfo bbi = convertBBox(bx1, by1, bx2, by2, netW, netH);
  //std::cout << std::fixed << std::setprecision(2) << "[bx1, by1, bx2, by2; netW, netH; maxIndex, maxProb]=[ " << bx1 << ", " << by1 << ", " << bx2 << ", " << by2 << "; " << netW << ", " << netH << "; " << maxIndex << ", " << maxProb << "]" <<  std::endl;

  if (bbi.width < 1 || bbi.height < 1) {
      return;
  }

  bbi.detectionConfidence = maxProb;
  bbi.classId = maxIndex;
  binfo.push_back(bbi);
}

static std::vector<NvDsInferParseObjectInfo>
decodeTensorYolo(const float* boxes, const float* scores, const float* classes, const uint& outputSize, const uint& netW,
    const uint& netH, const std::vector<float>& preclusterThreshold)
{
  std::vector<NvDsInferParseObjectInfo> binfo;

  for (uint b = 0; b < outputSize; ++b) {
    float maxProb = scores[b];
    int maxIndex = (int) classes[b];

    if (maxProb < preclusterThreshold[maxIndex]) {
      continue;
    }

    float bxc = boxes[b * 4 + 0];
    float byc = boxes[b * 4 + 1];
    float bw = boxes[b * 4 + 2];
    float bh = boxes[b * 4 + 3];

    float bx1 = bxc - bw / 2;
    float by1 = byc - bh / 2;
    float bx2 = bx1 + bw;
    float by2 = by1 + bh;

    addBBoxProposal(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, binfo);
  }

  return binfo;
}

static std::vector<NvDsInferParseObjectInfo>
decodeTensorYoloE(const float* boxes, const float* scores, const float* classes, const uint& outputSize, const uint& netW,
    const uint& netH, const std::vector<float>& preclusterThreshold)
{
  std::vector<NvDsInferParseObjectInfo> binfo;

  for (uint b = 0; b < outputSize; ++b) {
    float maxProb = scores[b];
    int maxIndex = (int) classes[b];

    if (maxProb < preclusterThreshold[maxIndex]) {
      continue;
    }

    float bx1 = boxes[b * 4 + 0];
    float by1 = boxes[b * 4 + 1];
    float bx2 = boxes[b * 4 + 2];
    float by2 = boxes[b * 4 + 3];

    addBBoxProposal(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, binfo);
  }

  return binfo;
}

void printVector(const std::vector<float>& vec) {
    std::cerr << "[NvDsInferParseCustomDFINE] detectionParams.perClassPreclusterThreshold = [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cerr << vec[i];
        if (i < vec.size() - 1) {
            std::cerr << ", ";
        }
    }
    std::cerr << "]" << std::endl;
}

void printArray(const float* arr, size_t size) {
    std::cerr << "[NvDsInferParseCustomDFINE] array = [";
    for (size_t i = 0; i < size; ++i) {
        std::cerr << arr[i];
        if (i < size - 1) {
            std::cerr << ", ";
        }
    }
    std::cerr << "]" << std::endl;
}

void printInt64Array(const int64_t* arr, size_t size) {
    std::cerr << "[NvDsInferParseCustomDFINE] array = [";
    for (size_t i = 0; i < size; ++i) {
        std::cerr << arr[i];
        if (i < size - 1) {
            std::cerr << ", ";
        }
    }
    std::cerr << "]" << std::endl;
}

// 解析 D-FINE Tensor 輸出
static std::vector<NvDsInferParseObjectInfo>
decodeTensorDFINE(const float* boxes, const float* scores, const int64_t* classes, 
                  const uint& outputSize, const uint& netW, const uint& netH, 
                  const std::vector<float>& preclusterThreshold) 
{
    std::vector<NvDsInferParseObjectInfo> binfo; // 存放有效物件資訊
    //Matthew: 
    //printArray(scores, outputSize);
    //printInt64Array(classes, outputSize);
    //printArray(boxes, outputSize);

    for (uint b = 0; b < outputSize; ++b) {
        float confidence = scores[b]; // 取得置信度
        int classId = static_cast<int>(classes[b]); // 取得類別索引

        // 過濾低置信度物件
        if (classId < 0 || confidence < preclusterThreshold[classId]) {
            continue;
        }

        // 取得邊界框座標
        float x1 = boxes[b * 4 + 0]*netW; // 左上角 X
        float y1 = boxes[b * 4 + 1]*netH; // 左上角 Y
        float x2 = boxes[b * 4 + 2]*netW; // 右下角 X
        float y2 = boxes[b * 4 + 3]*netH; // 右下角 Y

        // 確保邊界框座標合理
        if (x1 >= x2 || y1 >= y2 || x1 < 0 || y1 < 0 || x2 > netW || y2 > netH) {
            continue;
        }

        // 呼叫 addBBoxProposal，將物件加入 binfo
        addBBoxProposal(x1, y1, x2, y2, netW, netH, classId, confidence, binfo);
    }

    return binfo;
}

static bool NvDsInferParseCustomDFINE(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo, 
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, 
    std::vector<NvDsInferParseObjectInfo>& objectList) 
{
    //std::cout << "[NvDsInferParseCustomDFINE] Start parsing D-FINE detections\n";
    // 檢查是否有足夠的輸出層
    if (outputLayersInfo.size() < 3) {
        std::cerr << "ERROR: Missing output layers in D-FINE bbox parsing" << std::endl;
        return false;
    }
    //std::cerr << "[NvDsInferParseCustomDFINE] outputLayersInfo.size()= 3," << outputLayersInfo.size() << std::endl; 

    // 取得輸出層資訊, Matthew : the sequency is important
    const NvDsInferLayerInfo& boxesLayer = outputLayersInfo[2];   // 邊界框座標
    const NvDsInferLayerInfo& scoresLayer = outputLayersInfo[1];  // 信心分數
    const NvDsInferLayerInfo& classesLayer = outputLayersInfo[0]; // 類別索引

    // 確保每個輸出層的 buffer 不是空的
    if (!boxesLayer.buffer || !scoresLayer.buffer || !classesLayer.buffer) {
        std::cerr << "ERROR: One or more output buffers are null" << std::endl;
        return false;
    }

    // 獲取輸出大小 (物件數量)
    const uint outputSize = boxesLayer.inferDims.d[0];
    //std::cerr << "[NvDsInferParseCustomDFINE] outputSize = 300," << outputSize << std::endl;
    //printVector(detectionParams.perClassPreclusterThreshold);
    
    // 解析 tensor，轉換為 NvDsInferParseObjectInfo 格式
    std::vector<NvDsInferParseObjectInfo> detectedObjects = decodeTensorDFINE(
        (const float*) boxesLayer.buffer, 
        (const float*) scoresLayer.buffer, 
        (const int64_t*) classesLayer.buffer, 
        outputSize, 
        networkInfo.width, 
        networkInfo.height, 
        detectionParams.perClassPreclusterThreshold);

    // 將解析出的物件存入 objectList
    objectList.insert(objectList.end(), detectedObjects.begin(), detectedObjects.end());

    return true;
}

static bool
NvDsInferParseCustomYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  std::vector<NvDsInferParseObjectInfo> objects;

  const NvDsInferLayerInfo& boxes = outputLayersInfo[0];
  const NvDsInferLayerInfo& scores = outputLayersInfo[1];
  const NvDsInferLayerInfo& classes = outputLayersInfo[2];

  const uint outputSize = boxes.inferDims.d[0];

  std::vector<NvDsInferParseObjectInfo> outObjs = decodeTensorYolo((const float*) (boxes.buffer),
      (const float*) (scores.buffer), (const float*) (classes.buffer), outputSize, networkInfo.width, networkInfo.height,
      detectionParams.perClassPreclusterThreshold);

  objects.insert(objects.end(), outObjs.begin(), outObjs.end());

  objectList = objects;

  return true;
}

static bool
NvDsInferParseCustomYoloE(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  std::vector<NvDsInferParseObjectInfo> objects;

  const NvDsInferLayerInfo& boxes = outputLayersInfo[0];
  const NvDsInferLayerInfo& scores = outputLayersInfo[1];
  const NvDsInferLayerInfo& classes = outputLayersInfo[2];

  const uint outputSize = boxes.inferDims.d[0];

  std::vector<NvDsInferParseObjectInfo> outObjs = decodeTensorYoloE((const float*) (boxes.buffer),
      (const float*) (scores.buffer), (const float*) (classes.buffer), outputSize, networkInfo.width, networkInfo.height,
      detectionParams.perClassPreclusterThreshold);

  objects.insert(objects.end(), outObjs.begin(), outObjs.end());

  objectList = objects;

  return true;
}

extern "C" bool
NvDsInferParseYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  return NvDsInferParseCustomYolo(outputLayersInfo, networkInfo, detectionParams, objectList);
}

extern "C" bool
NvDsInferParseYoloE(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  return NvDsInferParseCustomYoloE(outputLayersInfo, networkInfo, detectionParams, objectList);
}

extern "C" bool
NvDsInferParseDFINE(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  return NvDsInferParseCustomDFINE(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloE);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseDFINE);
