#include "VisionEncoderSession.h"

#include "PersistentImageInput.h"
#include "CudaTensor.h"
#include <memory>
#include <vector>
#include <onnxruntime_cxx_api.h>

void VisionEncoderSession::run() {
    throwIfNotInitialised();
    isInitialised = false;
    auto sessionBindings = bindings.createIoBindingObject(*session);
    session->Run(Ort::RunOptions{nullptr}, sessionBindings);
}

std::shared_ptr<CudaTensor<float>> VisionEncoderSession::getImageTensor() {
    return image;
}

std::shared_ptr<CudaTensor<float>> VisionEncoderSession::getFpnFeat0Tensor() {
    return fpnFeat0;
}

std::shared_ptr<CudaTensor<float>> VisionEncoderSession::getFpnFeat1Tensor() {
    return fpnFeat1;
}

std::shared_ptr<CudaTensor<float>> VisionEncoderSession::getFpnFeat2Tensor() {
    return fpnFeat2;
}

std::shared_ptr<CudaTensor<float>> VisionEncoderSession::getFpnPos2Tensor() {
    return fpnPos2;
}
