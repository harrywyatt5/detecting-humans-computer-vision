#pragma once

#include "CPUTensor.h"
#include "CudaTensor.h"
#include "LanguageToken.h"
#include "UninitialisedSession.h"

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cstdint>
#include <memory>

class TextEncoderSession : public UninitialisedSession {
private:
    // Actual ORT details
    std::unique_ptr<Ort::Session> session;
    Ort::IoBinding bindings{nullptr};

    std::shared_ptr<CPUTensor<int64_t>> inputIdsTensor;
    std::shared_ptr<CPUTensor<int64_t>> attentionMaskTensor;
    std::shared_ptr<CudaTensor<float>> textFeaturesTensor;
public:
    // You should generally use TextEncoderSessionFactory rather than calling this method yourself...
    TextEncoderSession(
        std::unique_ptr<Ort::Session> session,
        Ort::IoBinding bindings,
        std::shared_ptr<CPUTensor<int64_t>> inputIdsTensor,
        std::shared_ptr<CPUTensor<int64_t>> attentionMaskTensor,
        std::shared_ptr<CudaTensor<float>> textFeaturesTensor
    ) : session(std::move(session)),
        bindings(std::move(bindings)),
        inputIdsTensor(std::move(inputIdsTensor)),
        attentionMaskTensor(std::move(attentionMaskTensor)),
        textFeaturesTensor(std::move(textFeaturesTensor)),
        UninitialisedSession() {}

    void run() override;
    std::vector<Ort::Value> runWithResult() override;

    // Getters
    std::shared_ptr<CPUTensor<int64_t>> getInputIdsTensor();
    std::shared_ptr<CPUTensor<int64_t>> getAttentionMaskTensor();
    std::shared_ptr<CudaTensor<float>> getTextFeaturesTensor();
};
