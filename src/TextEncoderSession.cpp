#include "TextEncoderSession.h"

#include "LanguageToken.h"
#include "CudaTensor.h"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <iostream>

void TextEncoderSession::run() {
    throwIfNotInitialised();
    this->session->Run(Ort::RunOptions{nullptr}, bindings);
}

std::vector<Ort::Value> TextEncoderSession::runWithResult() {
    run();
    return bindings.GetOutputValues();
}

std::shared_ptr<CudaTensor<float>> TextEncoderSession::getTextFeaturesTensor() {
    return textFeaturesTensor;
}

std::shared_ptr<CPUTensor<int64_t>> TextEncoderSession::getAttentionMaskTensor() {
    return attentionMaskTensor;
}

std::shared_ptr<CPUTensor<int64_t>> TextEncoderSession::getInputIdsTensor() {
    return inputIdsTensor;
}
