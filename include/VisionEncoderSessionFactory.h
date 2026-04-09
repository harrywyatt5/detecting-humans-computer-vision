#pragma once

#include "VisionEncoderSession.h"
#include <vector>
#include <memory>

class VisionEncoderSessionFactory {
private:
    std::vector<float> createPositionVector(int height, int width, int channels) const;
public:
    VisionEncoderSessionFactory() {}
    std::unique_ptr<VisionEncoderSession> createSession(int intermediateHeight, int intermediateWidth, const Sam3Context& samContext) const;
};
