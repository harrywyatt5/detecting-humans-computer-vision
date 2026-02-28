#pragma once

#include "AbstractSession.h"
#include "TextEncoderSession.h"
#include <memory>

class PersistentSam3Model {
private:
    std::unique_ptr<TextEncoderSession> textEncoderSession;
    std::unique_ptr<AbstractSession> visionEncoderSession;
    std::unique_ptr<AbstractSession> decoder;
};
