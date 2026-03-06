#pragma once

#include "SessionInitialiser.h"
#include "VisionEncoderSession.h"
#include "ImageProvider.h"
#include "VisionEncoderInitialiser.h"
#include <memory>

class VisionEncoderInitialiser : public SessionInitialiser<VisionEncoderSession> {
private:
    std::shared_ptr<ImageProvider> provider;
public:
    VisionEncoderInitialiser(std::shared_ptr<ImageProvider> p) : provider(std::move(p)) {};
    void initialiseSession(VisionEncoderSession& session) override;
};
