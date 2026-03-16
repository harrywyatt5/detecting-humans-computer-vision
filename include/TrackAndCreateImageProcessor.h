#pragma once

#include "CudaDevicesSingleton.h"
#include "CudaDevice.h"
#include "OutputProcessor.h"
#include "CPUTensor.h"
#include "GpuImage.h"
#include "Sam3Context.h"
#include <BYTETracker.h>
#include <Object.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdint>
#include <memory>
#include <string>

class TrackAndCreateImageProcessor : public OutputProcessor {
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
    std::unique_ptr<BYTETracker> tracker;
    std::vector<Object> trackedObjects;
    int sampledFrameCounter;

    void allocateMemory();
    void syncAndCheckCuda();
public:
    TrackAndCreateImageProcessor(
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

    ~TrackAndCreateImageProcessor() override;
    TrackAndCreateImageProcessor(TrackAndCreateImageProcessor&& other) noexcept;
    TrackAndCreateImageProcessor(const TrackAndCreateImageProcessor&) = delete;
    TrackAndCreateImageProcessor& operator=(const TrackAndCreateImageProcessor&) = delete;

    static TrackAndCreateImageProcessor createTrackAndCreateImageProcessor(
        int x,
        int y,
        int intermediateX,
        int intermediateY,
        int masks,
        float threshold,
        const Sam3Context& context
    );
};
