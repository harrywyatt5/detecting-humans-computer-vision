#pragma once

#include "VisionEncoderSession.h"
#include "AbstractStartSessionFactory.h"
#include <memory>

class VisionEncoderSessionFactory : public AbstractStartSessionFactory {
public:
    VisionEncoderSessionFactory() {}
    std::unique_ptr<AbstractSession> createSession(const Sam3Context& samContext) const override;
};
