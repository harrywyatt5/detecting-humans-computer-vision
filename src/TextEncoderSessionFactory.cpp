#include "TextEncoderSessionFactory.h"

#include "AbstractSession.h"
#include "TextEncoderSession.h"
#include "Sam3Context.h"
#include "CPUTensor.h"
#include "CudaTensor.h"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <cstdint>

std::unique_ptr<AbstractSession> TextEncoderSessionFactory::createSession(const Sam3Context& context) const {
    // Inputs
    auto inputIds = CPUTensor<int64_t>::createCPUTensor({1, 32}, context);
    auto attentionMasks = CPUTensor<int64_t>::createCPUTensor({1, 32}, context);

    // Outputs
    // Our inputs are only ever one because we only ever input one text prompt at a time
    auto textFeatures = CudaTensor<float>::createCudaTensor({1, 32, 256}, context);
    auto textMask = CudaTensor<bool>::createCudaTensor({1, 32}, context);

    auto session = std::make_unique<Ort::Session>(context.getEnvironment(), context.getTextEncoderPath().c_str(), context.getSessionOptions());
    // As the session is allocated on the heap, it should be no problem that we are moving these pieces of data
    Ort::IoBinding textEncoderBindings{*session};

    textEncoderBindings.BindInput("input_ids", inputIds->getTensor());
    textEncoderBindings.BindInput("attention_mask", attentionMasks->getTensor());
    textEncoderBindings.BindOutput("text_features", textFeatures->getTensor());
    textEncoderBindings.BindOutput("text_mask", textMask->getTensor());

    return std::make_unique<TextEncoderSession>(
        std::move(session),
        std::move(textEncoderBindings),
        std::move(inputIds),
        std::move(attentionMasks),
        std::move(textFeatures),
        std::move(textMask)
    );
}
