#include "MaskDecoderInitialiser.h"

#include "MaskDecoderSession.h"

void MaskDecoderInitialiser::initialiseSession(MaskDecoderSession& session) {
    auto attentionMask = session.getTextMasksTensor();
    langToken->populateAttentionMaskTensor(*attentionMask);
    session.setSessionAsInitialised();
}
