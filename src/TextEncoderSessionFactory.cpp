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

    auto session = std::make_unique<Ort::Session>(context.getEnvironment(), context.getTextEncoderPath().c_str(), context.getSessionOptions());
    BindingBlueprint blueprint;
    blueprint.addBinding(Binding("input_ids", inputIds, Binding::BindingType::INPUT));
    blueprint.addBinding(Binding("attention_mask", attentionMasks, Binding::BindingType::INPUT));
    blueprint.addBinding(Binding("text_features", textFeatures, Binding::BindingType::OUTPUT));

    // textEncoderBindings.BindInput("input_ids", inputIds->getTensor());
    // textEncoderBindings.BindInput("attention_mask", attentionMasks->getTensor());
    // textEncoderBindings.BindOutput("text_features", textFeatures->getTensor());
    // OnnxRuntime HATES the fact that this tensor is boolean, and won't write to it. Luckily, this output
    // is the same as the attention_mask input, so we just use that instead...
    //textEncoderBindings.BindOutput("text_mask", context.getCpuMemoryInfo());

    return std::make_unique<TextEncoderSession>(
        std::move(session),
        std::move(blueprint),
        std::move(inputIds),
        std::move(attentionMasks),
        std::move(textFeatures)
    );
}
