#include "LanguageToken.h"
#include "PersistentSam3Model.h"
#include "Sam3ContextBuilder.h"
#include "TextEncoderSessionFactory.h"
#include "PersistentImageInputFactory.h"
#include "VisionEncoderSessionFactory.h"
#include "MaskDecoderSessionFactory.h"
#include "CreateImageProcessor.h"
#include <onnxruntime_c_api.h>
#include <chrono>
#include <iostream>

int main() {
    // Required objects
    std::shared_ptr<LanguageToken> languageToken = LanguageToken::createFromFile("./language.token");
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
                            .withBatchLimit(1)
                            .withNumBoxesLimit(1)
                            .build();
    std::shared_ptr<PersistentImageInput> imageInput = PersistentImageInputFactory().createPersistentImageInput(3000, 2001, 1008, 1008, sam3ModelContext);
    std::shared_ptr<CreateImageProcessor> createImageProcessor = CreateImageProcessor::createCreateImageProcessor(3000, 2001, 288, 288, 200, 0.8f, sam3ModelContext);

    auto persistentModel = PersistentSam3Model::createSam3Model(sam3ModelContext);
    persistentModel.registerOutputProcessor(createImageProcessor);
    persistentModel.mountAndCalculatePrompt(languageToken);
    // imageInput->uploadImageFromDisk("img.jpg");
    // persistentModel.detect(imageInput);


    // Mount image
    imageInput->uploadImageFromDisk("img3.jpg");
    auto startTime = std::chrono::high_resolution_clock::now();
    persistentModel.detect(imageInput);
    persistentModel.processOutput();
    auto mutableInputImage = imageInput->getMutableGpuImage();
    createImageProcessor->outputMaskedImage(*mutableInputImage, 0.3f);
    auto endTime = std::chrono::high_resolution_clock::now();
    cv::Mat downloadedImage;
    cv::Mat image2;
    mutableInputImage->download(downloadedImage);
    cv::cvtColor(downloadedImage, image2, cv::COLOR_RGB2BGR);
    cv::imwrite("masks.jpg", image2);
    std::cout << "Taken " << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() << std::endl;

    return 0;
}
