#include "MaskDecoderSession.h"

#include "CudaTensor.h"
#include "UninitialisedSession.h"
#include <vector>
#include <memory>
#include <cstdint>
#include <onnxruntime_cxx_api.h>

void MaskDecoderSession::run() {
    throwIfNotInitialised();
    session->Run(Ort::RunOptions{nullptr}, bindings);
}

std::vector<Ort::Value> MaskDecoderSession::runWithResult() {
    run();
    return bindings.GetOutputValues();
}

std::shared_ptr<CudaTensor<uint8_t>> MaskDecoderSession::getTextMasksTensor() {
    return textMasks;
}
