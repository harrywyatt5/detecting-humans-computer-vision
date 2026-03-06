#pragma once

#include "SessionInitialiser.h"
#include "TextEncoderSession.h"
#include <memory>

class TextEncoderInitialiser : public SessionInitialiser<TextEncoderSession> {
private:
    std::shared_ptr<LanguageToken> langToken;
public:
    TextEncoderInitialiser(std::shared_ptr<LanguageToken> token) : langToken(std::move(token)) {};
    void initialiseSession(TextEncoderSession& session) override;
};
