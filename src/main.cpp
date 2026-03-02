#include "LanguageToken.h"
#include "PersistentSam3Model.h"
#include "Sam3ContextBuilder.h"
#include "TextEncoderSessionFactory.h"
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
                            .withLoggingLevel(ORT_LOGGING_LEVEL_WARNING)
                            .build();

    auto persistentModel = PersistentSam3Model(
        TextEncoderSessionFactory().createSession(sam3ModelContext),
        nullptr,
        nullptr
    );
    persistentModel.mountAndCalculatePrompt(languageToken);

    return 0;
}
