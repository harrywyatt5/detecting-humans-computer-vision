#include "MaskDecoderSessionFactory.h"

#include "MaskDecoderSession.h"
#include "CudaTensor.h"
#include "Sam3Context.h"
#include "TextEncoderSession.h"
#include "VisionEncoderSession.h"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>
#include <cstdint>

std::unique_ptr<MaskDecoderSession> MaskDecoderSessionFactory::createSession(const Sam3Context& samContext, TextEncoderSession& textEncoder, VisionEncoderSession& visionEncoder) const {
    // Inputs
    auto fpnFeat0 = visionEncoder.getFpnFeat0Tensor();
    auto fpnFeat1 = visionEncoder.getFpnFeat1Tensor();
    auto fpnFeat2 = visionEncoder.getFpnFeat2Tensor();
    auto fpnPos2 = visionEncoder.getFpnPos2Tensor();

    auto textFeatures = textEncoder.getTextFeaturesTensor();
    auto textMasks = textEncoder.getTextMaskTensor();

    // We don't want to pass bounding boxes to track, so we block out these values
    auto inputBoxes = CPUTensor<float>::createCPUTensor({1, 1, 4}, samContext);
    inputBoxes->copyToBuffer(std::vector<float>(4, 0.0f));
    auto inputBoxLabels = CPUTensor<int64_t>::createCPUTensor({1, 1}, samContext);
    inputBoxLabels->copyToBuffer(std::vector<int64_t>(1, 0));

    // We don't preallocate outputs unfortunately as we can have a varied number of outputs
    auto session = std::make_unique<Ort::Session>(samContext.getEnvironment(), samContext.getDecoderPath().c_str(), samContext.getSessionOptions());
    Ort::IoBinding bindings{*session};

    bindings.BindInput("fpn_feat_0", fpnFeat0->getTensor());
    bindings.BindInput("fpn_feat_1", fpnFeat1->getTensor());
    bindings.BindInput("fpn_feat_2", fpnFeat2->getTensor());
    bindings.BindInput("fpn_pos_2", fpnPos2->getTensor());
    bindings.BindInput("text_features", textFeatures->getTensor());
    bindings.BindInput("text_mask", textMasks->getTensor());
    bindings.BindInput("input_boxes", inputBoxes->getTensor());
    bindings.BindInput("input_boxes_labels", inputBoxLabels->getTensor());
    // TODO: change to GPU for image processing bit
    bindings.BindOutput("pred_masks", samContext.getCpuMemoryInfo());
    bindings.BindOutput("pred_boxes", samContext.getCpuMemoryInfo());
    bindings.BindOutput("pred_logits", samContext.getCpuMemoryInfo());
    bindings.BindOutput("presence_logits", samContext.getCpuMemoryInfo());

    return std::make_unique<MaskDecoderSession>(
        std::move(session),
        std::move(bindings),
        std::move(fpnFeat0),
        std::move(fpnFeat1),
        std::move(fpnFeat2),
        std::move(fpnPos2),
        std::move(textFeatures),
        std::move(textMasks),
        std::move(inputBoxes),
        std::move(inputBoxLabels)
    );
}
