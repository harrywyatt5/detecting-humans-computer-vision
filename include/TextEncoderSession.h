#pragma once

#include "AbstractSession.h"
#include "CPUTensor.h"
#include "CudaTensor.h"
#include "LanguageToken.h"

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cstdint>
#include <memory>

class TextEncoderSession : public AbstractSession {
private:
    // Actual ORT details
    std::unique_ptr<Ort::Session> session;
    Ort::IoBinding bindings{nullptr};

    std::unique_ptr<CPUTensor<int64_t>> inputIdsTensor;
    std::unique_ptr<CPUTensor<int64_t>> attentionMaskTensor;
    // We use a shared pointer for the textFeatures and textMasks because they are the inputs to the decoder...
    std::shared_ptr<CudaTensor<bool>> textFeaturesTensor;
    std::shared_ptr<CudaTensor<float>> textMaskTensor;

    bool isInitialised;
    void throwOnUninitialised() const;
public:
    // You should generally use TextEncoderSessionFactory rather than calling this method yourself...
    TextEncoderSession(
        std::unique_ptr<Ort::Session> session,
        Ort::IoBinding bindings,
        std::unique_ptr<CPUTensor<int64_t>> inputIdsTensor,
        std::unique_ptr<CPUTensor<int64_t>> attentionMaskTensor,
        std::shared_ptr<CudaTensor<bool>> textFeaturesTensor,
        std::shared_ptr<CudaTensor<float>> textMaskTensor
    ) : session(std::move(session)),
        bindings(std::move(bindings)),
        inputIdsTensor(std::move(inputIdsTensor)),
        attentionMaskTensor(std::move(attentionMaskTensor)),
        textFeaturesTensor(std::move(textFeaturesTensor)),
        textMaskTensor(std::move(textMaskTensor)),
        isInitialised(false) {}

    void initialiseSession(const LanguageToken& token);
    void run() override;
    std::vector<Ort::Value> runWithResult() override;
};
