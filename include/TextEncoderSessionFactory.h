#pragma once

#include "TextEncoderSession.h"
#include "Sam3Context.h"
#include <memory>

class TextEncoderSessionFactory {
public:
    TextEncoderSessionFactory() {};
    std::unique_ptr<TextEncoderSession> createSession(const Sam3Context& context) const;
};
