#pragma once

#include "OutputProcessor.h"
#include "CPUTensor.h"
#include "GpuImage.h"
#include "Sam3Context.h"
#include <opencv2/opencv.hpp>
#include <cstdint>
#include <string>

class CreateImageProcessor : public OutputProcessor {
private:
    int finalX;
    int finalY;
    int masksCount;
    float threshold;
    int deviceId;
    cv::cuda::GpuMat intermediateMask;
    cv::cuda::GpuMat outputMask;
    std::vector<uint8_t> masksInclusionCpu;
    uint8_t* maskInclusionPtr;
    cv::cuda::Stream stream;

    void allocateMemory();
    void syncAndCheckCuda();
public:
    CreateImageProcessor(
        int x,
        int y,
        int intermediateX,
        int intermediateY,
        int masks,
        float thres,
        int devId
    );
    void processOutput(
        const CudaTensor<float>& outputMasksTensor,
        const CPUTensor<float>& outputBoxesTensor,
        const CPUTensor<float>& outputLogitsTensor,
        const CPUTensor<float>& outputLogicTensor
    ) override;
    void outputMaskedImage(GpuImage& baseImage, const float mixPercentage);

    ~CreateImageProcessor() override;

    static CreateImageProcessor createCreateImageProcessor(
        int x,
        int y,
        int intermediateX,
        int intermediateY,
        int masks,
        float threshold,
        const Sam3Context& context
    );
};
