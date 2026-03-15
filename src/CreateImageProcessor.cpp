#include "CreateImageProcessor.h"

#include "CreateImageKernel.h"
#include "GpuImage.h"
#include "Sam3Context.h"
#include "CudaDevicesSingleton.h"
#include "CudaDevice.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
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
    int devId
) : finalX(x), finalY(y), masksCount(masks), threshold(thres) {
    cudaDevice = CudaDevicesSingleton::getInstance()->getForId(devId);
    cudaDevice->switchCudaDevice();
    outputMask = cv::cuda::GpuMat(cv::Size(x, y), CV_8UC1);
    intermediateMask = cv::cuda::GpuMat(cv::Size(iX, iY), CV_8UC1);

    allocateMemory();
}

void CreateImageProcessor::allocateMemory() {
    cudaDevice->switchCudaDevice();
    auto allocateError = cudaMalloc((void**)&maskInclusionPtr, masksCount * sizeof(uint8_t));

    if (allocateError != cudaSuccess) {
        throw std::runtime_error(
            std::string("Failed to allocate bytes on the CUDA device. Reason: ")
            + cudaGetErrorString(allocateError)
        );
    }

    auto hostAllocateError = cudaHostAlloc((void**)&maskInclusionCpuPtr, masksCount * sizeof(uint8_t), cudaHostAllocDefault);
    if (hostAllocateError != cudaSuccess) {
        throw std::runtime_error(
            std::string("Failed to allocate pinned memory. Reason: ")
            + cudaGetErrorString(hostAllocateError)
        );
    }
}

void CreateImageProcessor::processOutput(
    const CudaTensor<float>& outputMasksTensor,
    const CPUTensor<float>& outputBoxesTensor,
    const CPUTensor<float>& outputLogitsTensor,
    const CPUTensor<float>& outputLogicTensor
) {
    // Sync the stream here so our CPUTensors have all our values
    cudaDevice->waitForCompletion();

    // Sanity check
    if (outputLogitsTensor.getSize() != (size_t)masksCount) {
        throw std::runtime_error("Mismatch between CreateImageProcessor config and numbers of logits in tensor");
    }

    const float* logitsPtr = outputLogitsTensor.getConstStartPtr();

    float presenceScore = 1.0f / (1.0f + std::exp(-outputLogicTensor.getConstStartPtr()[0]));
    int count = 0;

    for (auto i = 0; i < masksCount; ++i) {
        float score = (1.0f / (1.0f + std::exp(-logitsPtr[i]))) * presenceScore;
        if (score >= threshold) {
            maskInclusionCpuPtr[i] = 1;
            ++count;
        } else {
            maskInclusionCpuPtr[i] = 0;
        }
    }

    std::cout << "Number of masks detected: " << count << "\n";

    // Copy our inclusion array onto the gpu
    auto error = cudaMemcpyAsync(
        (void*)maskInclusionPtr,
        (void*)maskInclusionCpuPtr,
        masksCount * sizeof(uint8_t),
        cudaMemcpyHostToDevice,
        cudaDevice->getCudaStream()
    );
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("Failed to copy logits to CUDA. Reason: ")
            + cudaGetErrorString(error)
        );
    }

    launchCreateMask(intermediateMask, cudaDevice->getOpenCVCudaStream(), outputMasksTensor.getConstStartPtr(), maskInclusionPtr, masksCount);
    cv::cuda::resize(intermediateMask, outputMask, cv::Size(finalX, finalY), 0, 0, cv::INTER_NEAREST, cudaDevice->getOpenCVCudaStream());
    
    // Wait until all GPU actions have finished
    syncAndCheckCuda();
}

void CreateImageProcessor::outputMaskedImage(GpuImage& base, const float mixPercentage) {
    auto baseImage = base.getMutableGpuMat();
    if (mixPercentage < 0.0f || mixPercentage > 1.0f) {
        throw std::runtime_error("mixPercentage must be between 0.0f and 1.0f (inclusive)");
    }

    if (baseImage.cols != finalX || baseImage.rows != finalY) {
        throw std::runtime_error("Provided image must be the same dimensions as mask");
    }

    launchCreateImage(outputMask, baseImage, cudaDevice->getOpenCVCudaStream(), mixPercentage);
}

CreateImageProcessor::~CreateImageProcessor() {
    if (maskInclusionPtr != nullptr) {
        auto error = cudaFree((void*)maskInclusionPtr);

        if (error != cudaSuccess) {
            // We don't throw here because we could be unwinding anyway...
            std::cerr 
                << "Could not free CUDA memory. This application may be leaking memory. Reason: " 
                << cudaGetErrorString(error)
                << std::endl;
        }
        maskInclusionPtr = nullptr;
    }

    if (maskInclusionCpuPtr != nullptr) {
        auto error = cudaFreeHost((void*)maskInclusionCpuPtr);

        if (error != cudaSuccess) {
            std::cerr 
                << "Could not free pinned host memory. This application may be leaking memory. Reason: " 
                << cudaGetErrorString(error)
                << std::endl;
        }
        maskInclusionCpuPtr = nullptr;
    }
}

void CreateImageProcessor::syncAndCheckCuda() {
    cudaDevice->waitForCompletion();

    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("A CUDA error occurred when trying to process CreateImageProcessor. Reason: ") + cudaGetErrorString(error));
    }
}

CreateImageProcessor CreateImageProcessor::createCreateImageProcessor(
    int x,
    int y,
    int intermediateX,
    int intermediateY,
    int masks,
    float thres,
    const Sam3Context& context
) {
    return CreateImageProcessor(
        x,
        y,
        intermediateX,
        intermediateY,
        masks,
        thres,
        context.getDeviceId()
    );
}

CreateImageProcessor::CreateImageProcessor(CreateImageProcessor&& other) noexcept 
    : finalX(other.finalX), finalY(other.finalY), masksCount(other.masksCount),
        threshold(other.threshold), intermediateMask(std::move(other.intermediateMask)), outputMask(std::move(other.outputMask)),
        maskInclusionCpuPtr(other.maskInclusionCpuPtr), maskInclusionPtr(other.maskInclusionPtr), cudaDevice(other.cudaDevice) {
            other.maskInclusionCpuPtr = nullptr;
            other.maskInclusionPtr = nullptr;
}
