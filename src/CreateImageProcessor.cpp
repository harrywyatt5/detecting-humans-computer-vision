#include "CreateImageProcessor.h"

#include "CreateImageKernel.h"
#include "Sam3Context.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <cuda_runtime.h>
#include <stdexcept>
#include <memory>
#include <cstdint>
#include <iostream>
#include <string>

CreateImageProcessor::CreateImageProcessor(
    int x,
    int y,
    int iX,
    int iY,
    int masks,
    float thres,
    const std::string& outPath,
    int devId
) : finalX(x), finalY(y), masksCount(masks), outputPath(outPath), deviceId(devId), threshold(thres) {
    cpuImage = std::make_shared<cv::Mat>(x, y, CV_8UC3);
    cv::cuda::setDevice(devId);
    outputImage = cv::cuda::GpuMat(cv::Size(x, y), CV_8UC3);
    intermediateImage = cv::cuda::GpuMat(cv::Size(iX, iY), CV_8UC3);
    masksInclusionCpu.resize(masksCount);

    allocateMemory();
}

void CreateImageProcessor::allocateMemory() {
    auto switchError = cudaSetDevice(deviceId);

    if (switchError != cudaSuccess) {
        throw std::runtime_error(
            std::string("Failed to switch CUDA device. Reason: ") 
            + cudaGetErrorString(switchError)
        );
    }

    auto allocateError = cudaMalloc((void**)&maskInclusionPtr, masksCount * sizeof(uint8_t));

    if (allocateError != cudaSuccess) {
        throw std::runtime_error(
            std::string("Failed to allocate bytes on the CUDA device. Reason: ")
            + cudaGetErrorString(allocateError)
        );
    }
}

void CreateImageProcessor::processOutput(
    const CudaTensor<float>& outputMasksTensor,
    const CPUTensor<float>& outputBoxesTensor,
    const CPUTensor<float>& outputLogitsTensor,
    const CPUTensor<float>& outputLogicTensor
) {
    // Sanity check
    if (outputLogitsTensor.getSize() != masksCount) {
        throw std::runtime_error("Mismatch between CreateImageProcessor config and numbers of logits in tensor");
    }

    const float* logitsPtr = outputLogitsTensor.getConstStartPtr();

    float presenceScore = 1.0f / (1.0f + std::exp(-outputLogicTensor.getConstStartPtr()[0]));

    for (auto i = 0; i < masksCount; ++i) {
        float score = (1.0f / (1.0f + std::exp(-logitsPtr[i]))) * presenceScore;
        std::cout << "Mask score: " << score << "\n";
        masksInclusionCpu[i] = score >= threshold ? 1 : 0;
    }

    // Copy our inclusion array onto the gpu
    auto error = cudaMemcpy(
        (void*)maskInclusionPtr,
        (void*)masksInclusionCpu.data(),
        masksCount * sizeof(uint8_t),
        cudaMemcpyHostToDevice
    );
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("Failed to copy logits to CUDA. Reason: ")
            + cudaGetErrorString(error)
        );
    }

    launchCreateImage(intermediateImage, stream, outputMasksTensor.getConstStartPtr(), maskInclusionPtr, masksCount);
    cv::cuda::resize(intermediateImage, outputImage, cv::Size(finalX, finalY), 0, 0, cv::INTER_LINEAR, stream);
    
    // Wait until all GPU actions have finished before downloading
    stream.waitForCompletion();
    auto lastError = cudaGetLastError();
    if (lastError != cudaSuccess) {
        throw std::runtime_error(std::string("Creating image on CUDA device failed. Reason: ") + cudaGetErrorString(lastError));
    }

    outputImage.download(*cpuImage);
    cv::imwrite(outputPath, *cpuImage);
}

std::shared_ptr<cv::Mat> CreateImageProcessor::getOutput() const {
    return cpuImage;
}

CreateImageProcessor::~CreateImageProcessor() {
    if (maskInclusionPtr == nullptr) {
        return;
    }

    auto error = cudaFree((void*)maskInclusionPtr);
    if (error != cudaSuccess) {
        // We don't throw here because we could be unwinding anyway...
        std::cerr 
            << "Could not free CUDA memory. This application may be leaking memory. Reason: " 
            << cudaGetErrorString(error)
            << std::endl;
    }
}

std::unique_ptr<CreateImageProcessor> CreateImageProcessor::createCreateImageProcessor(
    int x,
    int y,
    int intermediateX,
    int intermediateY,
    int masks,
    float thres,
    const std::string& savePath,
    const Sam3Context& context
) {
    return std::make_unique<CreateImageProcessor>(
        x,
        y,
        intermediateX,
        intermediateY,
        masks,
        thres,
        savePath,
        context.getDeviceId()
    );
}
