#include "TextEncoderSession.h"

#include "LanguageToken.h"
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <stdexcept>

void TextEncoderSession::initialiseSession(LanguageToken& token) {
    // This fills in the int64_t to find our target
    token.populateTensorWithToken(*inputIdsTensor);

    // We don't use the attention mask bit (for tracking boxes), so just fill it in with zeros
    attentionMaskTensor->copyToBuffer(std::vector<int64_t>(attentionMaskTensor->getSize(), 0));

    this->isInitialised = true;
}

void TextEncoderSession::run() {
    thowIfNotInitialised();
    this->session->Run(Ort::RunOptions{nullptr}, bindings);
}

std::vector<Ort::Value> TextEncoderSession::runWithResult() {
    run();
    return bindings.GetOutputValues();
}
