#include "TrackAndCreateImageProcessorBuilder.h"

#include "FrameSampler.h"
#include "Sam3Context.h"
#include "TrackAndCreateImageProcessor.h"
#include <stdexcept>
#include <memory>

TrackAndCreateImageProcessorBuilder::TrackAndCreateImageProcessorBuilder() 
    : finalX(1280), finalY(720), intermediateX(1008), intermediateY(1008),
        masksCount(200), threshold(0.85f), minimumFrames(5),
        sampler(nullptr), devId(0) {}

TrackAndCreateImageProcessorBuilder& TrackAndCreateImageProcessorBuilder::withImageWidth(const int x) {
    finalX = x;
    return *this;
}

TrackAndCreateImageProcessorBuilder& TrackAndCreateImageProcessorBuilder::withImageHeight(const int y) {
    finalY = y;
    return *this;
}

TrackAndCreateImageProcessorBuilder& TrackAndCreateImageProcessorBuilder::withIntermediateWidth(const int x) {
    intermediateX = x;
    return *this;
}

TrackAndCreateImageProcessorBuilder& TrackAndCreateImageProcessorBuilder::TrackAndCreateImageProcessorBuilder::withIntermediateHeight(const int y) {
    intermediateY = y;
    return *this;
}

TrackAndCreateImageProcessorBuilder& TrackAndCreateImageProcessorBuilder::withMasksCount(const int count) {
    masksCount = count;
    return *this;
}

TrackAndCreateImageProcessorBuilder& TrackAndCreateImageProcessorBuilder::withThreshold(const float thres) {
    threshold = thres;
    return *this;
}

TrackAndCreateImageProcessorBuilder& TrackAndCreateImageProcessorBuilder::withMinimumFramesToSample(const int count) {
    minimumFrames = count;
    return *this;
}

TrackAndCreateImageProcessorBuilder& TrackAndCreateImageProcessorBuilder::withFrameSampler(std::shared_ptr<FrameSampler> frameSampler) {
    sampler = frameSampler;
    return *this;
}

TrackAndCreateImageProcessorBuilder& TrackAndCreateImageProcessorBuilder::withDeviceId(const int id) {
    devId = id;
    return *this;
}


TrackAndCreateImageProcessorBuilder& TrackAndCreateImageProcessorBuilder::withDeviceIdFromContext(const Sam3Context& context) {
    devId = context.getDeviceId();
    return *this;
}

TrackAndCreateImageProcessor TrackAndCreateImageProcessorBuilder::build() const {
    // A sampler must be set
    if (sampler == nullptr) {
        throw std::runtime_error("No sampler value was set");
    }

    return TrackAndCreateImageProcessor(
        finalX,
        finalY,
        intermediateX,
        intermediateY,
        masksCount,
        threshold,
        minimumFrames,
        sampler,
        devId
    );
}
