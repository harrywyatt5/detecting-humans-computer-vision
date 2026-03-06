#include "VisionEncoderInitialiser.h"

#include "VisionEncoderSession.h"

void VisionEncoderInitialiser::initialiseSession(VisionEncoderSession& session) {
    auto imageTensor = session.getImageTensor();
    provider->writeImageToCudaTensor(*imageTensor);
    session.setSessionAsInitialised();
}
