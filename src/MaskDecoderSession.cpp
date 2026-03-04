#include "MaskDecoderSession.h"

#include <vector>
#include <onnxruntime_cxx_api.h>

void MaskDecoderSession::run() {
    session->Run(Ort::RunOptions{nullptr}, bindings);
}

std::vector<Ort::Value> MaskDecoderSession::runWithResult() {
    run();
    return bindings.GetOutputValues();
}
