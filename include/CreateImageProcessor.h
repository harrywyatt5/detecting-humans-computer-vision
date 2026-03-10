#pragma once

#include "OutputProcessor.h"
#include "CPUTensor.h"
#include "Sam3Context.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include <cstdint>
#include <string>

class CreateImageProcessor : public OutputProcessor {
private:
    int deviceId;
    cv::cuda::GpuMat intermediateMask;
    cv::cuda::GpuMat outputMask;
    std::vector<uint8_t> masksInclusionCpu;
    uint8_t* maskInclusionPtr;
    int masksCount;
    int finalX;
    int finalY;
    float threshold;
    cv::cuda::Stream stream;

    void allocateMemory();
    void syncAndCheckCuda();
public:
    CreateImageProcessor(
        int finalX,
        int finalY,
        int intermediateX,
        int intermediateY,
        int masks,
        float threshold,
        int deviceId
    );
    void processOutput(
        const CudaTensor<float>& outputMasksTensor,
        const CPUTensor<float>& outputBoxesTensor,
        const CPUTensor<float>& outputLogitsTensor,
        const CPUTensor<float>& outputLogicTensor
    ) override;
    void outputMaskedImage(cv::cuda::GpuMat& baseImage, const float mixPercentage);

    ~CreateImageProcessor() override;

    static std::unique_ptr<CreateImageProcessor> createCreateImageProcessor(
        int x,
        int y,
        int intermediateX,
        int intermediateY,
        int masks,
        float threshold,
        const Sam3Context& context
    );
};
