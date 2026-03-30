#include "TrackAndCreateImageProcessorBuilder.h"

#include "FrameSampler.h"
#include "TextProvider.h"
#include "TextConfig.h"
#include "TextProviderImpl.h"
#include "Sam3Context.h"
#include "TrackAndCreateImageProcessor.h"
#include <stdexcept>
#include <memory>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>

TrackAndCreateImageProcessorBuilder::TrackAndCreateImageProcessorBuilder() 
    : finalX(1280), finalY(720), intermediateX(1008), intermediateY(1008),
        masksCount(200), threshold(0.85f), minimumFrames(5),
        sampler(nullptr), textProvider(nullptr), devId(0) {}

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

TrackAndCreateImageProcessorBuilder& TrackAndCreateImageProcessorBuilder::withTextProvider(std::unique_ptr<TextProvider> provider) {
    textProvider = std::move(provider);
    return *this;
}

TrackAndCreateImageProcessorBuilder& TrackAndCreateImageProcessorBuilder::withDeviceIdFromContext(const Sam3Context& context) {
    devId = context.getDeviceId();
    return *this;
}

TrackAndCreateImageProcessor TrackAndCreateImageProcessorBuilder::build() {
    // A sampler must be set
    if (sampler == nullptr) {
        throw std::runtime_error("No sampler value was set");
    }

    // If we have no TextProvider, we create a default one
    if (textProvider == nullptr) {
        textProvider = std::make_unique<TextProviderImpl>(
            256,
            devId,
            (int)((float)finalX * 0.03f),
            (int)((float)finalY * 0.03f),
            TextConfig(cv::FONT_HERSHEY_SIMPLEX, cv::Scalar(255, 0, 0, 255), 2, 1.0f)
        );
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
        std::move(textProvider),
        devId
    );
}
