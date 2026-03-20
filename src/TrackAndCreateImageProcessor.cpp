#include "TrackAndCreateImageProcessor.h"

#include "CreateImageKernel.h"
#include "TrackKernel.h"
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
#include <ByteTrack/Rect.h>
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

    return (1.0f / (1.0f + std::exp(-logitsPtr[index]))) * (1.0f / (1.0f + std::exp(-logicPtr[0])));
}

std::vector<std::shared_ptr<byte_track::STrack>> TrackAndCreateImageProcessor::generateTrackedTracks(
    const CPUTensor<float>& boxesTensor,
    const CPUTensor<float>& logitsTensor,
    const CPUTensor<float>& logicTensor
) {
    // If this is first time, we have to generate a BYTETrack instance
    if (tracker == nullptr) {
        auto framerate = frameSampler->getFrameRate();
        tracker = std::make_unique<byte_track::BYTETracker>(framerate, framerate, 0.5f, 0.6f, 0.8f);
    }
    const float* basePtr = boxesTensor.getConstStartPtr();
    trackedObjects.clear();

    // TODO: Also write in the coordinates to write text? I.e. DrawableTextPlan
    for (int i = 0; i < masksCount; ++i) {
        int baseIndex = i * 4;
        // Even though bytetrack does thresholding, it's cheaper to do it here as well
        float score = calculateScore(logitsTensor, logicTensor, i);

        if (score >= threshold) {
            byte_track::Tlbr<float> aabb;
            aabb << basePtr[baseIndex], basePtr[baseIndex + 1], basePtr[baseIndex + 2], basePtr[baseIndex + 3];

            byte_track::Rect<float> rect = byte_track::generate_rect_by_tlbr<float>(aabb);
            trackedObjects.push_back(byte_track::Object(rect, i, score, i));
        }
    }

    return tracker->update(trackedObjects);
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
                // Remap target, if we actually got tracks this frame (first couple we dont)
                // This is, by textbook, kinda computationally inefficient and yields a time
                // complexity of O(N * M). However, as we'll probably only have about 20 bounding
                // boxes on screen at a time, a linear match through both rather using something more
                // advanced, like a hash map, is more appropriate
                for (size_t j = 0; j < tracks.size(); ++j) {
                    if (tracks[j]->getOriginalIndex() == i) {
                        target = tracks[j]->getTrackId();
                        break;
                    }
                }
            }

            // There is a chance that we have tracks, but the byte track decided to discard it
            maskMappingsCpuPtr[i] = MappedMask(target != -1, target);
        } else {
            maskMappingsCpuPtr[i] = MappedMask(false, -1);
        }
    }
}

void TrackAndCreateImageProcessor::copyMappingArray() {
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
    if (outputLogitsTensor.getSize() != (size_t)masksCount || outputBoxesTensor.getSize() != (size_t)masksCount * 4) {
        throw std::runtime_error("Mismatch between CreateImageProcessor config and numbers of logits/boxes in tensor");
    }

    // If we've already observed some frames, then we can start to track targets
    std::vector<std::shared_ptr<byte_track::STrack>> tracks;
    if (frameSampler->getFrameCount() >= minimumFrameThreshold) {
        tracks = generateTrackedTracks(outputBoxesTensor, outputLogitsTensor, outputLogicTensor);
    }

    populateMappingArray(outputLogitsTensor, outputLogicTensor, tracks);

    // Copy our mappings array onto the gpu
    copyMappingArray();

    launchCreateTrackedMask(intermediateMask, cudaDevice->getOpenCVCudaStream(), outputMasksTensor.getConstStartPtr(), maskMappingsGpuPtr, masksCount);
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
    if (maskMappingsGpuPtr != nullptr) {
        auto error = cudaFree((void*)maskMappingsGpuPtr);

        if (error != cudaSuccess) {
            // We don't throw here because we could be unwinding anyway...
            std::cerr 
                << "Could not free CUDA memory. This application may be leaking memory. Reason: " 
                << cudaGetErrorString(error)
                << std::endl;
        }
        maskMappingsGpuPtr = nullptr;
    }

    if (maskMappingsCpuPtr != nullptr) {
        auto error = cudaFreeHost((void*)maskMappingsCpuPtr);

        if (error != cudaSuccess) {
            std::cerr 
                << "Could not free pinned host memory. This application may be leaking memory. Reason: " 
                << cudaGetErrorString(error)
                << std::endl;
        }
        maskMappingsCpuPtr = nullptr;
    }
}

void TrackAndCreateImageProcessor::syncAndCheckCuda() {
    cudaDevice->waitForCompletion();

    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("A CUDA error occurred when trying to process TrackAndCreateImageProcessor. Reason: ") + cudaGetErrorString(error));
    }
}

void TrackAndCreateImageProcessor::generateInsertableNumbers(int count) {
    // TODO: make these parameters? 
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    int thickness = 3;
    double baseSize = 2;

    for (int i = 0; i < count; ++i) {
        cv::Mat newNumber();

    }
}

TrackAndCreateImageProcessor::TrackAndCreateImageProcessor(TrackAndCreateImageProcessor&& other) noexcept 
    : finalX(other.finalX), finalY(other.finalY), masksCount(other.masksCount),
        threshold(other.threshold), intermediateMask(std::move(other.intermediateMask)), outputMask(std::move(other.outputMask)),
        maskMappingsCpuPtr(other.maskMappingsCpuPtr), maskMappingsGpuPtr(other.maskMappingsGpuPtr), cudaDevice(other.cudaDevice),
        tracker(std::move(other.tracker)), trackedObjects(std::move(other.trackedObjects)), minimumFrameThreshold(other.minimumFrameThreshold),
        frameSampler(other.frameSampler) {
            other.maskMappingsCpuPtr = nullptr;
            other.maskMappingsGpuPtr = nullptr;
}
