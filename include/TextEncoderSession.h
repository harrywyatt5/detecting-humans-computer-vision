#pragma once

#include "CPUTensor.h"
#include "CudaTensor.h"
#include "LanguageToken.h"
#include "UninitialisedSession.h"
#include "BindingBlueprint.h"
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cstdint>
#include <memory>

class TextEncoderSession : public UninitialisedSession {
private:
    // Actual ORT details
    std::unique_ptr<Ort::Session> session;
    BindingBlueprint bindingBlueprint;

    std::shared_ptr<CPUTensor<int64_t>> inputIdsTensor;
    std::shared_ptr<CPUTensor<int64_t>> attentionMaskTensor;
    std::shared_ptr<CudaTensor<float>> textFeaturesTensor;
    std::shared_ptr<CudaTensor<uint8_t>> textMaskTensor;
public:
    // You should generally use TextEncoderSessionFactory rather than calling this method yourself...
    TextEncoderSession(
        std::unique_ptr<Ort::Session> session,
        BindingBlueprint blueprint,
        std::shared_ptr<CPUTensor<int64_t>> inputIds,
        std::shared_ptr<CPUTensor<int64_t>> attentionMask,
        std::shared_ptr<CudaTensor<float>> textFeatures,
        std::shared_ptr<CudaTensor<uint8_t>> textMask
    ) : session(std::move(session)),
        bindingBlueprint(std::move(blueprint)),
        inputIdsTensor(std::move(inputIds)),
        attentionMaskTensor(std::move(attentionMask)),
        textFeaturesTensor(std::move(textFeatures)),
        textMaskTensor(std::move(textMask)),
        UninitialisedSession() {}

    void run() override;

    // Getters
    std::shared_ptr<CPUTensor<int64_t>> getInputIdsTensor();
    std::shared_ptr<CPUTensor<int64_t>> getAttentionMaskTensor();
    std::shared_ptr<CudaTensor<float>> getTextFeaturesTensor();
    std::shared_ptr<CudaTensor<uint8_t>> getTextMaskTensor();
};
