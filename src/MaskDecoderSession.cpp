#include "MaskDecoderSession.h"

#include "CudaTensor.h"
#include "UninitialisedSession.h"
#include <vector>
#include <memory>
#include <cstdint>
#include <onnxruntime_cxx_api.h>

void MaskDecoderSession::run() {
    throwIfNotInitialised();
    auto sessionBindings = bindings.createIoBindingObject(*session);
    session->Run(Ort::RunOptions{nullptr}, sessionBindings);
}

std::shared_ptr<CudaTensor<uint8_t>> MaskDecoderSession::getTextMasksTensor() {
    return textMasks;
}

std::shared_ptr<CudaTensor<float>> MaskDecoderSession::getPredicateMasks() {
    return predicateMasks;
}

std::shared_ptr<CPUTensor<float>> MaskDecoderSession::getPredicateBoxes() {
    return predicateBoxes;
}

std::shared_ptr<CPUTensor<float>> MaskDecoderSession::getPredicateLogits() {
    return predicateLogits;
}

std::shared_ptr<CPUTensor<float>> MaskDecoderSession::getPredicateLogic() {
    return predicateLogic;
}
