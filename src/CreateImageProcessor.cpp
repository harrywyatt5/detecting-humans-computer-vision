#include "CreateImageProcessor.h"

#include "CreateImageKernel.h"
#include "Sam3Context.h"
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
) : finalX(x), finalY(y), masksCount(masks), threshold(thres), deviceId(devId) {
    cv::cuda::setDevice(devId);
    outputMask = cv::cuda::GpuMat(cv::Size(x, y), CV_8UC1);
    intermediateMask = cv::cuda::GpuMat(cv::Size(iX, iY), CV_8UC1);
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
    if (outputLogitsTensor.getSize() != (size_t)masksCount) {
        throw std::runtime_error("Mismatch between CreateImageProcessor config and numbers of logits in tensor");
    }

    const float* logitsPtr = outputLogitsTensor.getConstStartPtr();

    float presenceScore = 1.0f / (1.0f + std::exp(-outputLogicTensor.getConstStartPtr()[0]));
    int count = 0;

    for (auto i = 0; i < masksCount; ++i) {
        float score = (1.0f / (1.0f + std::exp(-logitsPtr[i]))) * presenceScore;
        if (score >= threshold) {
            masksInclusionCpu[i] = 1;
            ++count;
        } else {
            masksInclusionCpu[i] = 0;
        }
    }

    std::cout << "Number of masks detected: " << count << "\n";

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

    launchCreateMask(intermediateMask, stream, outputMasksTensor.getConstStartPtr(), maskInclusionPtr, masksCount);
    cv::cuda::resize(intermediateMask, outputMask, cv::Size(finalX, finalY), 0, 0, cv::INTER_NEAREST, stream);
    
    // Wait until all GPU actions have finished
    syncAndCheckCuda();
}

void CreateImageProcessor::outputMaskedImage(cv::cuda::GpuMat& base, const float mixPercentage) {
    if (mixPercentage < 0.0f || mixPercentage > 1.0f) {
        throw std::runtime_error("mixPercentage must be between 0.0f and 1.0f (inclusive)");
    }

    if (base.cols != finalX || base.rows != finalY) {
        throw std::runtime_error("Provided image must be the same dimensions as mask");
    }

    launchCreateImage(outputMask, base, stream, mixPercentage);
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

void CreateImageProcessor::syncAndCheckCuda() {
    stream.waitForCompletion();

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
