#include "TextEncoderSession.h"

#include "LanguageToken.h"
#include "CudaTensor.h"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <cstdint>
#include <vector>
#include <stdexcept>

void TextEncoderSession::initialiseSession(LanguageToken& token) {
    // This fills in the int64_t to find our target
    token.populateTensorsWithToken(*inputIdsTensor, *attentionMaskTensor);
    this->isInitialised = true;
}

void TextEncoderSession::run() {
    thowIfNotInitialised();
    this->session->Run(Ort::RunOptions{nullptr}, bindings);
}

std::vector<Ort::Value> TextEncoderSession::runWithResult() {
    run();
    return bindings.GetOutputValues();
}

std::shared_ptr<CudaTensor<float>> TextEncoderSession::getTextFeaturesTensor() {
    return textFeaturesTensor;
}

std::shared_ptr<CudaTensor<uint8_t>> TextEncoderSession::getTextMaskTensor() {
    return textMaskTensor;
}
