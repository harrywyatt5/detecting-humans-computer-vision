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
    // TODO: change this into a uint8[] buffer that we can just pass to ROS2
    std::shared_ptr<cv::Mat> cpuImage;
    cv::cuda::GpuMat intermediateImage;
    cv::cuda::GpuMat outputImage;
    std::vector<uint8_t> masksInclusionCpu;
    uint8_t* maskInclusionPtr;
    int masksCount;
    int finalX;
    int finalY;
    float threshold;
    std::string outputPath;
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
        const std::string& outPath,
        int deviceId
    );
    void processOutput(
        const CudaTensor<float>& outputMasksTensor,
        const CPUTensor<float>& outputBoxesTensor,
        const CPUTensor<float>& outputLogitsTensor,
        const CPUTensor<float>& outputLogicTensor
    ) override;
    std::shared_ptr<cv::Mat> getOutput() const;
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
