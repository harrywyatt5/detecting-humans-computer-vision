#pragma once

#include "AbstractSessionFactory.h"
#include "AbstractSession.h"
#include "Sam3Context.h"

#include <memory>

class TextEncoderSessionFactory : public AbstractSessionFactory {
public:
    TextEncoderSessionFactory() {};
    std::unique_ptr<AbstractSession> createSession(const Sam3Context& context) const override;
};
