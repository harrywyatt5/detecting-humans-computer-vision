#include "VisionEncoderSession.h"

#include "PersistentImageInput.h"
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
