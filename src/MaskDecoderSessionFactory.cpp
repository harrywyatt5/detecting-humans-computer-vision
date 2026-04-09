#include "MaskDecoderSessionFactory.h"

#include "MaskDecoderSession.h"
#include "BindingBlueprint.h"
#include "Binding.h"
#include "CudaTensor.h"
#include "Sam3Context.h"
#include "TextEncoderSession.h"
#include "VisionEncoderSession.h"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>
#include <cstdint>
#include <algorithm>

std::unique_ptr<MaskDecoderSession> MaskDecoderSessionFactory::createSession(const Sam3Context& samContext, TextEncoderSession& textEncoder, VisionEncoderSession& visionEncoder) const {
    // Inputs
    auto fpnFeat0 = visionEncoder.getFpnFeat0Tensor();
    auto fpnFeat1 = visionEncoder.getFpnFeat1Tensor();
    auto fpnFeat2 = visionEncoder.getFpnFeat2Tensor();
    auto fpnPos2 = visionEncoder.getFpnPos2Tensor();

    auto textFeatures = textEncoder.getTextFeaturesTensor();

    // Create text masks, which is just the same as our attention mask. But we have to do this at runtime
    std::shared_ptr<CudaTensor<uint8_t>> textMasks = CudaTensor<uint8_t>::createCudaTensorWithTypeOverride({1, 32}, samContext, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);

    // We don't want to pass bounding boxes to track, so we block out these values
    std::shared_ptr<CudaTensor<float>> inputBoxes = CudaTensor<float>::createCudaTensor({1, 1, 4}, samContext);
    inputBoxes->copyToBuffer(std::vector<float>(4, 0.0f));
    std::shared_ptr<CudaTensor<int64_t>> inputBoxLabels = CudaTensor<int64_t>::createCudaTensor({1, 1}, samContext);
    inputBoxLabels->copyToBuffer(std::vector<int64_t>(1, -10));

    // Outputs
    // the fpn_feat_0 tensor tell us how big our predMask has to be
    int predMaskHeight = fpnFeat0->getTensorShape()[2];
    int predMaskWidth = fpnFeat0->getTensorShape()[3];
    std::shared_ptr<CudaTensor<float>> predMasks = CudaTensor<float>::createCudaTensor({1, 200, predMaskHeight, predMaskWidth}, samContext);
    std::shared_ptr<CPUTensor<float>> predBoxes = CPUTensor<float>::createCPUTensor({1, 200, 4}, samContext);
    std::shared_ptr<CPUTensor<float>> predLogits = CPUTensor<float>::createCPUTensor({1, 200}, samContext);
    std::shared_ptr<CPUTensor<float>> predLogic = CPUTensor<float>::createCPUTensor({1, 1}, samContext);

    auto session = std::make_unique<Ort::Session>(samContext.getEnvironment(), samContext.getDecoderPath().c_str(), samContext.getSessionOptions());
    BindingBlueprint bindings;

    // Bind all inputs
    bindings.addBinding(Binding("fpn_feat_0", fpnFeat0, Binding::BindingType::INPUT));
    bindings.addBinding(Binding("fpn_feat_1", fpnFeat1, Binding::BindingType::INPUT));
    bindings.addBinding(Binding("fpn_feat_2", fpnFeat2, Binding::BindingType::INPUT));
    bindings.addBinding(Binding("fpn_pos_2", fpnPos2, Binding::BindingType::INPUT));
    bindings.addBinding(Binding("text_features", textFeatures, Binding::BindingType::INPUT));
    bindings.addBinding(Binding("text_mask", textMasks, Binding::BindingType::INPUT));
    bindings.addBinding(Binding("input_boxes", inputBoxes, Binding::BindingType::INPUT));
    bindings.addBinding(Binding("input_boxes_labels", inputBoxLabels, Binding::BindingType::INPUT));
    // Bind all outputs
    bindings.addBinding(Binding("pred_masks", predMasks, Binding::BindingType::OUTPUT));
    bindings.addBinding(Binding("pred_boxes", predBoxes, Binding::BindingType::OUTPUT));
    bindings.addBinding(Binding("pred_logits", predLogits, Binding::BindingType::OUTPUT));
    bindings.addBinding(Binding("presence_logits", predLogic, Binding::BindingType::OUTPUT));

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
        std::move(inputBoxLabels),
        std::move(predMasks),
        std::move(predBoxes),
        std::move(predLogits),
        std::move(predLogic)
    );
}
