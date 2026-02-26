#include "TextEncoderSession.h"

#include "LanguageToken.h"
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <stdexcept>

void TextEncoderSession::initialiseSession(const LanguageToken& token) {

}

void TextEncoderSession::run() {

}

std::vector<Ort::Value> TextEncoderSession::runWithResult() {

}

void TextEncoderSession::throwOnUninitialised() const {
    if (!this->isInitialised) {
        throw std::runtime_error("TextEncoderSession cannot be run before it has been initialised. Use TextEncoderSession::initialiseSession first!");
    }
}
