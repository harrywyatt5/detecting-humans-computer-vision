#pragma once

#include "CudaTensor.h"
#include "Image.h"
#include "StartFlowSession.h"
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <vector>

class VisionEncoderSession : public StartFlowSession<Image> {
private:
    std::unique_ptr<Ort::Session> session;
    Ort::IoBinding bindings{nullptr};

    // Input
    std::unique_ptr<CudaTensor<float>> image;
    // Outputs
    std::shared_ptr<CudaTensor<float>> fpnFeat0;
    std::shared_ptr<CudaTensor<float>> fpnFeat1;
    std::shared_ptr<CudaTensor<float>> fpnFeat2;
    std::shared_ptr<CudaTensor<float>> fpnPos2;

public:
    void initialiseSession(const Image& image) override;
    void run() override;
    std::vector<Ort::Value> runWithResult() override;
};
