#include "FrameSampler.h"

#include <chrono>
#include <stdexcept>
#include <cstdint>
#include <iostream>

int64_t FrameSampler::getFrameCount() const {
    return frameCount;
}

bool FrameSampler::isCurrentlyMidFrame() const {
    return startTime != -1;
}

void FrameSampler::toggleFrame(bool debugLog) {
    // Means we are registering a new frame's start time
    if (!isCurrentlyMidFrame()) {
        startTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    } else {
        auto currTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        totalTime += currTime - startTime;
        startTime = -1;
        ++frameCount;

        if (debugLog) {
            std::cerr << "Frame time: " << (currTime - startTime) << "ms\n";
        }
    }
}

int64_t FrameSampler::getFrameTime() const {
    if (frameCount == 0) {
        return 0;
    }

    return totalTime / frameCount;
}

float FrameSampler::getFrameRate() const {
    return (float)1000 / (float)getFrameTime();
}
