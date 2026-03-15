#pragma once

#include "CudaDevicesSingleton.h"
#include "CudaDevice.h"
#include "OutputProcessor.h"
#include "CPUTensor.h"
#include "GpuImage.h"
#include "Sam3Context.h"
#include <opencv2/opencv.hpp>
#include <cstdint>
#include <memory>
#include <string>

class CreateImageProcessor : public OutputProcessor {
private:
    int finalX;
    int finalY;
    int masksCount;
    float threshold;
    cv::cuda::GpuMat intermediateMask;
    cv::cuda::GpuMat outputMask;
    uint8_t* maskInclusionCpuPtr;
    uint8_t* maskInclusionPtr;
    std::shared_ptr<CudaDevice> cudaDevice;

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
    CreateImageProcessor(CreateImageProcessor&& other) noexcept;
    CreateImageProcessor(const CreateImageProcessor&) = delete;
    CreateImageProcessor& operator=(const CreateImageProcessor&) = delete;

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
