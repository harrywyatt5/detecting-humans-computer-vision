#include "TextEncoderSessionFactory.h"

#include "Binding.h"
#include "BindingBlueprint.h"
#include "TextEncoderSession.h"
#include "Sam3Context.h"
#include "CPUTensor.h"
#include "CudaTensor.h"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <cstdint>
#include <vector>

std::unique_ptr<TextEncoderSession> TextEncoderSessionFactory::createSession(const Sam3Context& context) const {
    // Inputs
    std::shared_ptr<CPUTensor<int64_t>> inputIds = CPUTensor<int64_t>::createCPUTensorWithTypeOverride({1, 32}, context, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    std::shared_ptr<CPUTensor<int64_t>> attentionMasks = CPUTensor<int64_t>::createCPUTensorWithTypeOverride({1, 32}, context, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);

    // Outputs
    // Our inputs are only ever one because we only ever input one text prompt at a time
    std::shared_ptr<CudaTensor<float>> textFeatures = CudaTensor<float>::createCudaTensor({1, 32, 256}, context);
    std::shared_ptr<CudaTensor<uint8_t>> textMask = CudaTensor<uint8_t>::createCudaTensorWithTypeOverride({1, 32}, context, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);

    auto session = std::make_unique<Ort::Session>(context.getEnvironment(), context.getTextEncoderPath().c_str(), context.getSessionOptions());
    BindingBlueprint blueprint;
    blueprint.addBinding(Binding("input_ids", inputIds, Binding::BindingType::INPUT));
    blueprint.addBinding(Binding("attention_mask", attentionMasks, Binding::BindingType::INPUT));
    blueprint.addBinding(Binding("text_features", textFeatures, Binding::BindingType::OUTPUT));
    // This output is unused, but we continue to bind
    blueprint.addBinding(Binding("text_mask", textMask, Binding::BindingType::OUTPUT));

    return std::make_unique<TextEncoderSession>(
        std::move(session),
        std::move(blueprint),
        std::move(inputIds),
        std::move(attentionMasks),
        std::move(textFeatures),
        std::move(textMask)
    );
}
