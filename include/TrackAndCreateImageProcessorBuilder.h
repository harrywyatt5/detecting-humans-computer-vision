#pragma once

#include "FrameSampler.h"
#include "Sam3Context.h"
#include "TrackAndCreateImageProcessor.h"
#include <memory>

class TrackAndCreateImageProcessorBuilder {
private:
    int finalX;
    int finalY;
    int intermediateX;
    int intermediateY;
    int masksCount;
    float threshold;
    int minimumFrames;
    std::shared_ptr<FrameSampler> sampler;
    int devId;
public:
    TrackAndCreateImageProcessorBuilder();

    TrackAndCreateImageProcessorBuilder& withImageWidth(const int x);
    TrackAndCreateImageProcessorBuilder& withImageHeight(const int y);
    TrackAndCreateImageProcessorBuilder& withIntermediateWidth(const int x);
    TrackAndCreateImageProcessorBuilder& withIntermediateHeight(const int y);
    TrackAndCreateImageProcessorBuilder& withMasksCount(const int count);
    TrackAndCreateImageProcessorBuilder& withThreshold(const float thres);
    TrackAndCreateImageProcessorBuilder& withMinimumFramesToSample(const int count);
    TrackAndCreateImageProcessorBuilder& withFrameSampler(std::shared_ptr<FrameSampler> frameSampler);
    TrackAndCreateImageProcessorBuilder& withDeviceId(const int id);
    TrackAndCreateImageProcessorBuilder& withDeviceIdFromContext(const Sam3Context& context);
    TrackAndCreateImageProcessor build() const;
};
