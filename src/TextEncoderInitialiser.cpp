#include "TextEncoderInitialiser.h"

#include "TextEncoderSession.h"

void TextEncoderInitialiser::initialiseSession(TextEncoderSession& session) {
    auto inputIdsTensor = session.getInputIdsTensor();
    auto attentionMaskTensor = session.getAttentionMaskTensor();

    langToken->populateTextIdsTensor(*inputIdsTensor);
    langToken->populateAttentionMaskTensor(*attentionMaskTensor);
    session.setSessionAsInitialised();
}
