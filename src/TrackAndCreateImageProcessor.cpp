#include "TrackAndCreateImageProcessor.h"

#include "CreateImageKernel.h"
#include "GpuImage.h"
#include "Sam3Context.h"
#include "CudaDevicesSingleton.h"
#include "CudaDevice.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <ByteTrack/BYTETracker.h>
#include <ByteTrack/Object.h>
#include <ByteTrack/STrack.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <memory>
#include <cstdint>
#include <iostream>
#include <string>

TrackAndCreateImageProcessor::TrackAndCreateImageProcessor(
    int x,
    int y,
    int intermediateX,
    int intermediateY,
    int masks,
    float thres,
    int minimumFrames,
    std::shared_ptr<FrameSampler> sampler,
    int devId
) : finalX(x), finalY(y), masksCount(masks), threshold(thres),
    tracker(nullptr), minimumFrameThreshold(minimumFrames), frameSampler(sampler) {
    cudaDevice = CudaDevicesSingleton::getInstance()->getForId(devId);
    cudaDevice->switchCudaDevice();
    outputMask = cv::cuda::GpuMat(cv::Size(x, y), CV_8UC1);
    intermediateMask = cv::cuda::GpuMat(cv::Size(intermediateX, intermediateY), CV_8UC1);
    trackedObjects.reserve(200);

    allocateMemory();
}

void TrackAndCreateImageProcessor::allocateMemory() {
    cudaDevice->switchCudaDevice();
    auto allocateError = cudaMalloc((void**)&maskMappingsGpuPtr, masksCount * sizeof(MappedMask));

    if (allocateError != cudaSuccess) {
        throw std::runtime_error(
            std::string("Failed to allocate bytes on the CUDA device. Reason: ")
            + cudaGetErrorString(allocateError)
        );
    }

    auto hostAllocateError = cudaHostAlloc((void**)&maskMappingsCpuPtr, masksCount * sizeof(MappedMask), cudaHostAllocDefault);
    if (hostAllocateError != cudaSuccess) {
        throw std::runtime_error(
            std::string("Failed to allocate pinned memory. Reason: ")
            + cudaGetErrorString(hostAllocateError)
        );
    }
}

float TrackAndCreateImageProcessor::calculateScore(const CPUTensor<float>& logitsTensor, const CPUTensor<float>& logicTensor, int index) const {
    const float* logicPtr = logicTensor.getConstStartPtr();
    const float* logitsPtr = logitsTensor.getConstStartPtr();

    return (1.0f / (1.0f + std::exp(-logitsPtr[index]))) * (1.0f / (1.0f + std::exp(logicPtr[0])));
}

std::vector<std::shared_ptr<byte_track::STrack>> TrackAndCreateImageProcessor::generateTrackedTracks(
    const CPUTensor<float>& boxesTensor,
    const CPUTensor<float>& logitsTensor,
    const CPUTensor<float>& logicTensor
) {
    trackedObjects.clear();

    // TODO: Also write in the coordinates to write text? I.e. DrawableTextPlan
    for (int i = 0; i < masksCount; ++i) {
        
    }
}

void TrackAndCreateImageProcessor::populateMappingArray(
    const CPUTensor<float>& logitsTensor,
    const CPUTensor<float>& logicTensor,
    const std::vector<std::shared_ptr<byte_track::STrack>>& tracks
) {
    for (int i = 0; i < masksCount; ++i) {
        if (calculateScore(logitsTensor, logicTensor, i) >= threshold) {
            int target = -1;
            if (tracks.empty()) {
                target = i;
            } else {
                for (int j = 0; j < tracks.size(); ++j) {
                    if (tracks[j]->getOriginalIndex() == i) {
                        target = tracks[j]->getTrackId();
                        break;
                    }
                }
            }

            maskMappingsCpuPtr[i] = MappedMask(target != -1, target);
        } else {
            maskMappingsCpuPtr[i] = MappedMask(false, i);
        }
    }
}

void TrackAndCreateImageProcessor::processOutput(
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

    // If we've already observed some frames, then we can start to track targets
    if (frameSampler->getFrameCount() >= minimumFrameThreshold) {

    }

    populateMappingArray(outputBoxesTensor, outputLogitsTensor, outputLogicTensor);

    // Copy our mappings array onto the gpu
    auto error = cudaMemcpyAsync(
        (void*)maskMappingsGpuPtr,
        (void*)maskMappingsCpuPtr,
        masksCount * sizeof(MappedMask),
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

void TrackAndCreateImageProcessor::outputMaskedImage(GpuImage& base, const float mixPercentage) {
    auto baseImage = base.getMutableGpuMat();
    if (mixPercentage < 0.0f || mixPercentage > 1.0f) {
        throw std::runtime_error("mixPercentage must be between 0.0f and 1.0f (inclusive)");
    }

    if (baseImage.cols != finalX || baseImage.rows != finalY) {
        throw std::runtime_error("Provided image must be the same dimensions as mask");
    }

    launchCreateImage(outputMask, baseImage, cudaDevice->getOpenCVCudaStream(), mixPercentage);
}

TrackAndCreateImageProcessor::~TrackAndCreateImageProcessor() {
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

void TrackAndCreateImageProcessor::syncAndCheckCuda() {
    cudaDevice->waitForCompletion();

    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("A CUDA error occurred when trying to process TrackAndCreateImageProcessor. Reason: ") + cudaGetErrorString(error));
    }
}

TrackAndCreateImageProcessor::TrackAndCreateImageProcessor(TrackAndCreateImageProcessor&& other) noexcept 
    : finalX(other.finalX), finalY(other.finalY), masksCount(other.masksCount),
        threshold(other.threshold), intermediateMask(std::move(other.intermediateMask)), outputMask(std::move(other.outputMask)),
        maskInclusionCpuPtr(other.maskInclusionCpuPtr), maskInclusionPtr(other.maskInclusionPtr), cudaDevice(other.cudaDevice) {
            other.maskInclusionCpuPtr = nullptr;
            other.maskInclusionPtr = nullptr;
}
