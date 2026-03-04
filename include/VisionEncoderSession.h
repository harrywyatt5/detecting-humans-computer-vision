#pragma once

#include "CudaTensor.h"
#include "PersistentImageInput.h"
#include "StartFlowSession.h"
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <vector>

class VisionEncoderSession : public StartFlowSession<PersistentImageInput> {
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
    VisionEncoderSession(
        std::unique_ptr<Ort::Session> session,
        Ort::IoBinding bindings,
        std::unique_ptr<CudaTensor<float>> image,
        std::shared_ptr<CudaTensor<float>> fpnFeat0,
        std::shared_ptr<CudaTensor<float>> fpnFeat1,
        std::shared_ptr<CudaTensor<float>> fpnFeat2,
        std::shared_ptr<CudaTensor<float>> fpnPos2
    ) : session(std::move(session)),
        bindings(std::move(bindings)),
        image(std::move(image)),
        fpnFeat0(std::move(fpnFeat0)),
        fpnFeat1(std::move(fpnFeat1)),
        fpnFeat2(std::move(fpnFeat2)),
        fpnPos2(std::move(fpnPos2)) {}
    void initialiseSession(PersistentImageInput& image) override;
    void run() override;
    std::vector<Ort::Value> runWithResult() override;

    // Getters
    std::shared_ptr<CudaTensor<float>> getFpnFeat0Tensor();
    std::shared_ptr<CudaTensor<float>> getFpnFeat1Tensor();
    std::shared_ptr<CudaTensor<float>> getFpnFeat2Tensor();
    std::shared_ptr<CudaTensor<float>> getFpnPos2Tensor();
};
