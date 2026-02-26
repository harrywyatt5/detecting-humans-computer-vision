#pragma once

#include "AbstractSession.h"
#include "CPUTensor.h"
#include "CudaTensor.h"

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cstdint>
#include <memory>

class TextEncoderSession : public AbstractSession {
private:
    // Actual ORT details
    std::unique_ptr<Ort::Session> session;
    Ort::IoBinding bindings{nullptr};

    std::vector<CPUTensor<int64_t>> inputTensors;
    CudaTensor<bool> textFeaturesTensor;
    CudaTensor<float> textMaskTensor;
};
