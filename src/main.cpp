#include "LanguageToken.h"
#include "PersistentSam3Model.h"
#include "Sam3ContextBuilder.h"
#include "TextEncoderSessionFactory.h"
#include "PersistentImageInputFactory.h"
#include "VisionEncoderSessionFactory.h"
#include "MaskDecoderSessionFactory.h"
#include <onnxruntime_c_api.h>

int main() {
    // Required objects
    auto languageToken = LanguageToken::createFromFile("./language.token");
    auto sam3ModelContext = Sam3ContextBuilder()
                            .withApplicationName("RealTimeHumans")
                            .withCPUThreadMax(1)
                            .withTextEncoderPath("./sam3-onnx/text-encoder-fp16.onnx")
                            .withVisionEncoderPath("./sam3-onnx/vision-encoder-fp16.onnx")
                            .withDecoderPath("./sam3-onnx/geo-encoder-mask-decoder-fp16.onnx")
                            .withFP16Enabled(true)
                            .withDeviceId(0)
                            .withEngineCacheDir("./.engine-cache")
                            .withGraphOptimistionLevel(GraphOptimizationLevel::ORT_ENABLE_ALL)
                            .withLoggingLevel(ORT_LOGGING_LEVEL_ERROR)
                            .withBatchLimit(1)
                            .withNumBoxesLimit(1)
                            .build();
    auto imageInput = PersistentImageInputFactory().createPersistentImageInput(1920, 1080, 1008, 1008, sam3ModelContext);
    auto textEncoderSession = TextEncoderSessionFactory().createSession(sam3ModelContext);
    auto visionEncoderSession = VisionEncoderSessionFactory().createSession(sam3ModelContext);
    auto decoderSession = MaskDecoderSessionFactory().createSession(sam3ModelContext, *textEncoderSession, *visionEncoderSession);
    auto persistentModel = PersistentSam3Model(
        std::move(textEncoderSession),
        std::move(visionEncoderSession),
        std::move(decoderSession)
    );
    persistentModel.mountAndCalculatePrompt(languageToken);

    // Mount image
    imageInput.uploadImageFromDisk("img.jpg");
    persistentModel.detect(imageInput);

    return 0;
}
