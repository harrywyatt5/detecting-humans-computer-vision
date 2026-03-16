#pragma once

#include <cstdint>

class FrameSampler {
private:
    int64_t totalTime;
    int64_t startTime;
    int64_t frameCount;
public:
    FrameSampler() : totalTime(0), startTime(-1), frameCount(0) {}

    int64_t getFrameCount() const;
    bool isCurrentlyMidFrame() const;
    void toggleFrame(bool debugLog = false);
    float getFrameRate() const;
    int64_t getFrameTime() const; 
};
