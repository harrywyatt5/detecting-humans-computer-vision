#include "VisionEncoderSession.h"

#include "PersistentImageInput.h"
#include "CudaTensor.h"
#include <memory>
#include <vector>
#include <onnxruntime_cxx_api.h>

void VisionEncoderSession::initialiseSession(PersistentImageInput& input) {
    input.writeImageToTensor(*image);
    isInitialised = true;
}

void VisionEncoderSession::run() {
    thowIfNotInitialised();
    isInitialised = false;

    session->Run(Ort::RunOptions{nullptr}, bindings);
}

std::vector<Ort::Value> VisionEncoderSession::runWithResult() {
    run();
    return bindings.GetOutputValues();
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
