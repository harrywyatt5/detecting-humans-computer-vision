#pragma once

#include "UninitialisedSession.h"
#include "CudaTensor.h"
#include "CPUTensor.h"
#include <memory>
#include <vector>
#include <cstdint>
#include <onnxruntime_cxx_api.h>

class MaskDecoderSession : public UninitialisedSession {
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
    std::unique_ptr<CudaTensor<float>> inputBoxes;
    // int64_t have to be on the CPU unfortunately...
    std::unique_ptr<CPUTensor<int64_t>> inputBoxesLabels;

    // Outputs
    std::shared_ptr<CudaTensor<float>> predicateMasks;
    std::shared_ptr<CPUTensor<float>> predicateBoxes;
    std::shared_ptr<CPUTensor<float>> predicateLogits;
    std::shared_ptr<CPUTensor<float>> predicateLogic;
public:
    MaskDecoderSession(
        std::unique_ptr<Ort::Session> session,
        Ort::IoBinding bindings,
        std::shared_ptr<CudaTensor<float>> fpnFeat0,
        std::shared_ptr<CudaTensor<float>> fpnFeat1,
        std::shared_ptr<CudaTensor<float>> fpnFeat2,
        std::shared_ptr<CudaTensor<float>> fpnPos2,
        std::shared_ptr<CudaTensor<float>> textFeatures,
        std::shared_ptr<CudaTensor<uint8_t>> textMasks,
        std::unique_ptr<CudaTensor<float>> inputBoxes,
        std::unique_ptr<CPUTensor<int64_t>> inputBoxesLabels,
        std::shared_ptr<CudaTensor<float>> pMasks,
        std::shared_ptr<CPUTensor<float>> pBoxes,
        std::shared_ptr<CPUTensor<float>> pLogits,
        std::shared_ptr<CPUTensor<float>> pLogic
    ) : session(std::move(session)),
        bindings(std::move(bindings)),
        fpnFeat0(std::move(fpnFeat0)),
        fpnFeat1(std::move(fpnFeat1)),
        fpnFeat2(std::move(fpnFeat2)),
        fpnPos2(std::move(fpnPos2)),
        textFeatures(std::move(textFeatures)),
        textMasks(std::move(textMasks)),
        inputBoxes(std::move(inputBoxes)),
        inputBoxesLabels(std::move(inputBoxesLabels)),
        predicateMasks(std::move(pMasks)),
        predicateBoxes(std::move(pBoxes)),
        predicateLogits(std::move(pLogits)),
        predicateLogic(std::move(pLogic)),
        UninitialisedSession() {}

    void run() override;
    std::vector<Ort::Value> runWithResult() override;

    // Getters
    std::shared_ptr<CudaTensor<uint8_t>> getTextMasksTensor();
    std::shared_ptr<CudaTensor<float>> getPredicateMasks();
    std::shared_ptr<CPUTensor<float>> getPredicateBoxes();
    std::shared_ptr<CPUTensor<float>> getPredicateLogits();
    std::shared_ptr<CPUTensor<float>> getPredicateLogic();
};
