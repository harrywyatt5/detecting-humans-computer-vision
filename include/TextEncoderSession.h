#pragma once

#include "CPUTensor.h"
#include "CudaTensor.h"
#include "LanguageToken.h"
#include "StartFlowSession.h"

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cstdint>
#include <memory>

class TextEncoderSession : public StartFlowSession<LanguageToken> {
private:
    // Actual ORT details
    std::unique_ptr<Ort::Session> session;
    Ort::IoBinding bindings{nullptr};

    std::unique_ptr<CPUTensor<int64_t>> inputIdsTensor;
    // We use a shared pointer for textFeatures and attentionMasks because they are the input to the decoder...
    std::shared_ptr<CPUTensor<int64_t>> attentionMaskTensor;
    std::shared_ptr<CudaTensor<float>> textFeaturesTensor;
public:
    // You should generally use TextEncoderSessionFactory rather than calling this method yourself...
    TextEncoderSession(
        std::unique_ptr<Ort::Session> session,
        Ort::IoBinding bindings,
        std::unique_ptr<CPUTensor<int64_t>> inputIdsTensor,
        std::shared_ptr<CPUTensor<int64_t>> attentionMaskTensor,
        std::shared_ptr<CudaTensor<float>> textFeaturesTensor
    ) : session(std::move(session)),
        bindings(std::move(bindings)),
        inputIdsTensor(std::move(inputIdsTensor)),
        attentionMaskTensor(std::move(attentionMaskTensor)),
        textFeaturesTensor(std::move(textFeaturesTensor)),
        StartFlowSession<LanguageToken>() {}

    void initialiseSession(LanguageToken& token) override;
    void run() override;
    std::vector<Ort::Value> runWithResult() override;

    // Getters
    std::shared_ptr<CPUTensor<int64_t>> getAttentionMaskTensor();
    std::shared_ptr<CudaTensor<float>> getTextFeaturesTensor();
};
