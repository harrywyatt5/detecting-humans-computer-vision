#pragma once

#include "AbstractSession.h"
#include "CudaTensor.h"
#include <memory>
#include <cstdint>
#include <onnxruntime_cxx_api.h>

class MaskDecoderSession : public AbstractSession {
private:
    std::unique_ptr<Ort::Session> session;
    Ort::IoBinding bindings;

    // Inputs
    std::shared_ptr<CudaTensor<float>> fpnFeat0;
    std::shared_ptr<CudaTensor<float>> fpnFeat1;
    std::shared_ptr<CudaTensor<float>> fpnFeat2;
    std::shared_ptr<CudaTensor<float>> fpnPos2;
    std::shared_ptr<CudaTensor<float>> textFeatures;
    std::shared_ptr<CudaTensor<uint8_t>> textMasks;
    std::shared_ptr<CudaTensor<float>> inputBoxes;
    std::shared_ptr<CudaTensor<int64_t>> inputBoxesLabels;

    // Outputs
    // TODO: do we actually want these to be on the gpu? 
    std::unique_ptr<CudaTensor<float>> predicateMasks;
    std::unique_ptr<CudaTensor<float>> predicateBoxes;
    std::unique_ptr<CudaTensor<float>> predicateLogits;
    std::unique_ptr<CudaTensor<float>> predicateLogic;
};
