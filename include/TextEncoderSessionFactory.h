#pragma once

#include "AbstractSession.h"
#include "AbstractStartSessionFactory.h"
#include "TextEncoderSession.h"
#include "Sam3Context.h"
#include <memory>

class TextEncoderSessionFactory : public AbstractStartSessionFactory {
public:
    TextEncoderSessionFactory() {};
    std::unique_ptr<AbstractSession> createSession(const Sam3Context& context) const override;
};
