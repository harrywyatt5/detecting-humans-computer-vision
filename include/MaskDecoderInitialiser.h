#pragma once

#include "SessionInitialiser.h"
#include "MaskDecoderSession.h"
#include "LanguageToken.h"
#include <memory>

class MaskDecoderInitialiser : public SessionInitialiser<MaskDecoderSession> {
private:
    std::shared_ptr<LanguageToken> langToken;
public:
    MaskDecoderInitialiser(std::shared_ptr<LanguageToken> token) : langToken(std::move(token)) {}
    void initialiseSession(MaskDecoderSession& session) override; 
};
