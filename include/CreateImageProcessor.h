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
    cv::cuda::GpuMat redMask;
    cv::cuda::GpuMat intermediateMask;
    cv::cuda::GpuMat outputImage;
    cv::cuda::GpuMat outputMask;
    std::vector<uint8_t> masksInclusionCpu;
    uint8_t* maskInclusionPtr;
    int masksCount;
    int finalX;
    int finalY;
    float threshold;
    cv::cuda::Stream stream;

    void allocateMemory();
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
    const cv::cuda::GpuMat& outputMaskedImage(const cv::cuda::GpuMat& base);

    ~CreateImageProcessor() override;

    static std::unique_ptr<CreateImageProcessor> createCreateImageProcessor(
        int x,
        int y,
        int intermediateX,
        int intermediateY,
        int masks,
        float threshold,
        const std::string& savePath,
        const Sam3Context& context
    );
};
