#pragma once

#include "MaskDecoderSession.h"
#include "Sam3Context.h"
#include "TextEncoderSession.h"
#include "VisionEncoderSession.h"
#include <memory>

class MaskDecoderSessionFactory {
public:
    MaskDecoderSessionFactory() {}
    std::unique_ptr<MaskDecoderSession> createSession(const Sam3Context& samContext, TextEncoderSession& textEncoder, VisionEncoderSession& visionEncoder) const;
};
