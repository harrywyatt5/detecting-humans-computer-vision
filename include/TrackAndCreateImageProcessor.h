#pragma once

#include "CudaDevicesSingleton.h"
#include "CudaDevice.h"
#include "FrameSampler.h"
#include "OutputProcessor.h"
#include "CPUTensor.h"
#include "GpuImage.h"
#include "Sam3Context.h"
#include "MappedMask.h"
#include <ByteTrack/BYTETracker.h>
#include <ByteTrack/Object.h>
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
    MappedMask* maskMappingsGpuPtr;
    MappedMask* maskMappingsCpuPtr;
    std::shared_ptr<CudaDevice> cudaDevice;
    std::unique_ptr<byte_track::BYTETracker> tracker;
    std::vector<byte_track::Object> trackedObjects;
    int minimumFrameThreshold;
    std::shared_ptr<FrameSampler> frameSampler;
    int insertableNumXSize;
    int insertableNumYSize;
    std::vector<cv::cuda::GpuMat> insertableNums;

    void allocateMemory();
    void syncAndCheckCuda();
    void copyMappingArray();
    std::vector<std::shared_ptr<byte_track::STrack>> generateTrackedTracks(
        const CPUTensor<float>& boxesTensor,
        const CPUTensor<float>& logitsTensor,
        const CPUTensor<float>& logicTensor
    );
    void populateMappingArray(
        const CPUTensor<float>& logitsTensor,
        const CPUTensor<float>& logicTensor,
        const std::vector<std::shared_ptr<byte_track::STrack>>& tracks
    );
    float calculateScore(const CPUTensor<float>& logitsTensor, const CPUTensor<float>& logicTensor, int index) const;
    void generateInsertableNumbers(int count);
public:
    TrackAndCreateImageProcessor(
        int x,
        int y,
        int intermediateX,
        int intermediateY,
        int masks,
        float thres,
        int minimumFrames,
        std::shared_ptr<FrameSampler> sampler,
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
};
