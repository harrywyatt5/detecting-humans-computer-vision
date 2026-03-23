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
    std::unique_ptr<TextProvider> provider,
    int devId
) : finalX(x), finalY(y), masksCount(masks), threshold(thres), tracker(nullptr), minimumFrameThreshold(minimumFrames),
    frameSampler(sampler), textProvider(std::move(provider)) {
    cudaDevice = CudaDevicesSingleton::getInstance()->getForId(devId);
    cudaDevice->switchCudaDevice();
    outputMask = cv::cuda::GpuMat(cv::Size(x, y), CV_16UC1);
    intermediateMask = cv::cuda::GpuMat(cv::Size(intermediateX, intermediateY), CV_16UC1);
    textTemplates.reserve(masksCount);
    trackedObjects.reserve(masksCount);

    allocateMemory();
}

void TrackAndCreateImageProcessor::allocateMemory() {
    cudaDevice->switchCudaDevice();
    // Allocate the space for the mask mappings
    auto maskGpuError = cudaMalloc((void**)&maskMappingsGpuPtr, masksCount * sizeof(MappedMask));
    if (maskGpuError != cudaSuccess) {
        throw std::runtime_error(
            std::string("Failed to allocate bytes on the CUDA device. Reason: ")
            + cudaGetErrorString(maskGpuError)
        );
    }

    auto hostMaskError = cudaHostAlloc((void**)&maskMappingsCpuPtr, masksCount * sizeof(MappedMask), cudaHostAllocDefault);
    if (hostMaskError != cudaSuccess) {
        throw std::runtime_error(
            std::string("Failed to allocate pinned memory. Reason: ")
            + cudaGetErrorString(hostMaskError)
        );
    }

    // Allocate the space for text templates
    auto hostTemplateError = cudaHostAlloc((void**)&textTemplateCpuPtr, masksCount * sizeof(TextTemplateBlueprint), cudaHostAllocDefault);
    if (hostTemplateError != cudaSuccess) {
        throw std::runtime_error(
            std::string("Failed to allocate pinned memory. Reason: ")
            + cudaGetErrorString(hostTemplateError)
        );
    }

    auto templateGpuError = cudaMalloc((void**)&textTemplateGpuPtr, masksCount * sizeof(GPUTextTemplateBlueprint));
    if (templateGpuError != cudaSuccess) {
        throw std::runtime_error(
            std::string("Failed to allocate bytes on the CUDA device. Reason: ")
            + cudaGetErrorString(templateGpuError)
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
    // It would be cheaper to populate the text blueprints here (one less loop through the array) 
    // but they could end up being removed by the bytetrack, so there is chance that it forms an incorrect result
    // DISCUSS
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
            // Having -1 in this value will cause the value to wrap to 65,535 but we don't
            // care as we will ignore it
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

void TrackAndCreateImageProcessor::copyTextTemplates() {
    // Check we have enough space
    if (textTemplates.size() > (unsigned int)masksCount) {
        throw std::runtime_error("Too many templates provided for template buffer");
    }

    // Copy them into our pinned memory, which is easier for the GPU to access
    for (unsigned int i = 0; i < textTemplates.size(); ++i) {
        textTemplateCpuPtr[i] = static_cast<GPUTextTemplateBlueprint>(textTemplates[i]);
    }

    auto error = cudaMemcpyAsync(
        (void*)textTemplateGpuPtr,
        (void*)textTemplateCpuPtr,
        // We only copy the count, even though we allocate masksCount instances
        // which likely leaves us with uninitialised memory. This is why it is so important
        // to also pass 'count' to our kernel
        textTemplates.size() * sizeof(GPUTextTemplateBlueprint),
        cudaMemcpyHostToDevice,
        cudaDevice->getCudaStream()
    );
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("Failed to copy text template data to CUDA device. Reason: ")
            + cudaGetErrorString(error)
        );
    }
}

void TrackAndCreateImageProcessor::populateTextTemplates(const std::vector<std::shared_ptr<byte_track::STrack>>& tracks) {
    for (unsigned int i = 0; i < tracks.size(); ++i) {
        int trackId = tracks[i]->getTrackId();

        if (textProvider->hasTextForNumber(trackId)) {
            std::cout << "Rect here " << std::to_string(tracks[i]->getRect().tl_x()) << " and " << std::to_string(tracks[i]->getRect().tl_y()) << "\n";
            textTemplates.push_back(TextTemplateBlueprint::createBlueprintFromRect(
                trackId,
                tracks[i]->getRect(),
                intermediateMask.cols,
                intermediateMask.rows,
                finalX,
                finalY,
                textProvider.get()
            ));
        } else {
            // TODO: expand how many tracks we are storing. Not implemented currently so just ignore and log
            std::cerr << "No text for the number " << std::to_string(trackId) << ". Will be ignored\n";
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
    
    textTemplates.clear();
    populateTextTemplates(tracks);
    // If it turns out we are going to draw text templates on the image
    if (textTemplates.size() > 0) {
        copyTextTemplates();
    }
}

void TrackAndCreateImageProcessor::outputMaskedImage(GpuImage& base, const float mixPercentage) {
    auto baseImage = base.getMutableGpuMat();
    if (mixPercentage < 0.0f || mixPercentage > 1.0f) {
        throw std::runtime_error("mixPercentage must be between 0.0f and 1.0f (inclusive)");
    }

    if (baseImage.cols != finalX || baseImage.rows != finalY) {
        throw std::runtime_error("Provided image must be the same dimensions as mask");
    }

    std::cout << "Will draw " << std::to_string(textTemplates.size()) << "\n";
    launchCreateImageWithText(
        outputMask,
        baseImage,
        textTemplateGpuPtr,
        textTemplates.size(),
        cudaDevice->getOpenCVCudaStream(),
        mixPercentage
    );

    // Wait for all gpu actions to finish
    syncAndCheckCuda();
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

    if (textTemplateCpuPtr != nullptr) {
        auto error = cudaFreeHost((void*)textTemplateCpuPtr);

        if (error != cudaSuccess) {
            std::cerr 
                << "Could not free pinned host memory. This application may be leaking memory. Reason: " 
                << cudaGetErrorString(error)
                << std::endl;
        }
        textTemplateCpuPtr = nullptr;
    }

    if (textTemplateGpuPtr != nullptr) {
        auto error = cudaFree((void*)textTemplateGpuPtr);

        if (error != cudaSuccess) {
            std::cerr 
                << "Could not free CUDA memory. This application may be leaking memory. Reason: " 
                << cudaGetErrorString(error)
                << std::endl;
        }
        textTemplateGpuPtr = nullptr;
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
        maskMappingsCpuPtr(other.maskMappingsCpuPtr), maskMappingsGpuPtr(other.maskMappingsGpuPtr), textTemplateCpuPtr(other.textTemplateCpuPtr),
        textTemplateGpuPtr(other.textTemplateGpuPtr), cudaDevice(other.cudaDevice), tracker(std::move(other.tracker)),
        trackedObjects(std::move(other.trackedObjects)), minimumFrameThreshold(other.minimumFrameThreshold), frameSampler(other.frameSampler),
        textProvider(std::move(other.textProvider)) {
            other.maskMappingsCpuPtr = nullptr;
            other.maskMappingsGpuPtr = nullptr;
            other.textTemplateCpuPtr = nullptr;
            other.textTemplateGpuPtr = nullptr;
}
