#pragma once

#include "VisionEncoderSession.h"
#include <memory>

class VisionEncoderSessionFactory {
public:
    VisionEncoderSessionFactory() {}
    std::unique_ptr<VisionEncoderSession> createSession(const Sam3Context& samContext) const;
};
