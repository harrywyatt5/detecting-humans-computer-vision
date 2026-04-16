#include "HumanDetectionNode.h"

#include "LanguageToken.h"
#include "Sam3ContextBuilder.h"
#include "CudaDevicesSingleton.h"
#include "LoggingLevel.h"
#include "FrameSampler.h"
#include "PersistentSam3Model.h"
#include "PersistentImageInputFactory.h"
#include "TrackAndCreateImageProcessor.h"
#include "TrackAndCreateImageProcessorBuilder.h"
#include "TextProvider.h"
#include "TextProviderImpl.h"
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <isaac_ros_managed_nitros/managed_nitros_publisher.hpp>
#include <isaac_ros_managed_nitros/managed_nitros_subscriber.hpp>
#include <isaac_ros_nitros_image_type/nitros_image_view.hpp>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <chrono>
#include <stdexcept>
#include <string>
#include <filesystem>

HumanDetectionNode::HumanDetectionNode() 
    : samModel(nullptr),
        imageInput(nullptr),
        samContext(nullptr),
        promptToken(nullptr),
        trackCreateProcessor(nullptr),
        frameSampler(std::make_shared<FrameSampler>()),
        threshold(0.0f),
        overlayPercentage(0.3f),
        isFullyConfigured(false),
        Node("human_detection_node")
{
    auto shareLocation = ament_index_cpp::get_package_share_directory("real_time_humans");

    this->declare_parameter<std::string>("sam3_engine_cache_dir", std::getenv("HOME") + std::string("/.cache/detect_humans"));
    this->declare_parameter<int>("max_cpu_threads", 1);
    this->declare_parameter<bool>("use_fp16", true);
    this->declare_parameter<std::string>("int8_calibration_table", "");
    this->declare_parameter<std::string>("log_level", "error");
    this->declare_parameter<std::string>("sam3_text_encoder_path", shareLocation + "/sam3-onnx/text-encoder-fp16.onnx");
    this->declare_parameter<std::string>("sam3_vision_encoder_path", shareLocation + "/sam3-onnx/vision-encoder-fp16.onnx");
    this->declare_parameter<std::string>("sam3_decoder_path", shareLocation + "/sam3-onnx/geo-encoder-mask-decoder-fp16.onnx");
    this->declare_parameter<std::string>("encoded_prompt_path", shareLocation + "/language.token");
    this->declare_parameter<int64_t>("maximum_vram", 6442450944LL); // TODO: allow input that is more readable?
    this->declare_parameter<int>("sam3_image_input_size", 504);
    this->declare_parameter<int>("cuda_device_id", 0);
    this->declare_parameter<float>("threshold", 0.85f);
    this->declare_parameter<float>("overlay_percentage", 0.3);
    this->declare_parameter<float>("text_box_size", 0.03);
    this->declare_parameter<std::string>("camera_topic", "/left_eye_cam");
    this->declare_parameter<std::string>("masked_image_topic", "masked_image");
    this->declare_parameter<std::string>("masked_image_frame_id", "image_frame");

    // Prepare nitros subscriber and publisher
    rclcpp::QoS subQosProfile(1);
    subQosProfile.keep_last(1);
    subQosProfile.best_effort();
    subQosProfile.durability_volatile();
    cameraSub = std::make_shared<nitros::ManagedNitrosSubscriber<nitros::NitrosImageView>>(
        this,
        this->get_parameter("camera_topic").as_string(),
        nitros::nitros_image_rgb8_t::supported_type_name,
        std::bind(&HumanDetectionNode::cameraImageCallback, this, std::placeholders::_1),
        nitros::NitrosDiagnosticsConfig{},
        subQosProfile
    );

    rclcpp::QoS pubQosProfile(5);
    pubQosProfile.keep_last(5);
    pubQosProfile.best_effort();
    pubQosProfile.durability_volatile();
    maskedImagePub = std::make_shared<nitros::ManagedNitrosPublisher<nitros::NitrosImage>>(
        this,
        this->get_parameter("masked_image_topic").as_string(),
        nitros::nitros_image_rgb8_t::supported_type_name, 
        nitros::NitrosDiagnosticsConfig{},
        pubQosProfile
    );

    // Configurables
    threshold = this->get_parameter("threshold").as_double();
    overlayPercentage = this->get_parameter("overlay_percentage").as_double();
    imageFrameId = this->get_parameter("masked_image_frame_id").as_string();
    auto loggingLevel = LoggingLevel::fromString(this->get_parameter("log_level").as_string(), true);
    promptToken = std::make_shared<LanguageToken>(LanguageToken::createFromFile(this->get_parameter("encoded_prompt_path").as_string()));
    intermediateImageSize = this->get_parameter("sam3_image_input_size").as_int();
    RCLCPP_INFO(this->get_logger(), "About to load Sam3Model...");
    configureSam3Model(loggingLevel);
    RCLCPP_INFO(this->get_logger(), "Sam3 model was loaded. Compiling engine for language prompt and then will be ready!");
    mountPrompt();
    RCLCPP_INFO(this->get_logger(), "Done! Ready to receive events");
}

void HumanDetectionNode::configureSam3Model(const LoggingLevel& loggingLevel) {
    auto cudaDeviceId = this->get_parameter("cuda_device_id").as_int();
    auto calibrationTablePath = this->get_parameter("int8_calibration_table").as_string();
    auto builder = Sam3ContextBuilder()
                    .withApplicationName("real_time_humans")
                    .withCPUThreadMax(1)
                    .withTextEncoderPath(this->get_parameter("sam3_text_encoder_path").as_string())
                    .withVisionEncoderPath(this->get_parameter("sam3_vision_encoder_path").as_string())
                    .withDecoderPath(this->get_parameter("sam3_decoder_path").as_string())
                    .withFP16Enabled(true)
                    .withDeviceId(cudaDeviceId)
                    .withEngineCacheDir(this->get_parameter("sam3_engine_cache_dir").as_string())
                    .withGraphOptimistionLevel(GraphOptimizationLevel::ORT_ENABLE_ALL)
                    .withLoggingLevel(loggingLevel.toOrtLoggingLevel())
                    .withMaxGPUMemory(this->get_parameter("maximum_vram").as_int())
                    .withComputeStreamEnabled(true)
                    .withComputeStream(CudaDevicesSingleton::getInstance()->getForId(cudaDeviceId)->getCudaStream())
                    .withCudaGraphsEnabled(false);
    
    if (calibrationTablePath != "") {
        RCLCPP_INFO(this->get_logger(), "Int8 will be enabled");
        builder.withUseInt8ForEncoder(true)
            .withInt8NativeCalibrationTable(calibrationTablePath)
            // ENABLE_ALL might insert new nodes we didn't expect when calibrating, so
            // we downgrade the optimisation to basic when a table is used
            .withGraphOptimistionLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    }

    samContext = std::make_unique<Sam3Context>(builder.build());
    samModel = std::make_unique<PersistentSam3Model>(PersistentSam3Model::createSam3Model(intermediateImageSize, *samContext));
}

void HumanDetectionNode::mountPrompt() {
    samModel->mountAndCalculatePrompt(promptToken);
}

void HumanDetectionNode::configureCameraImageConversion(const nitros::NitrosImageView& img) {
    auto encoding = img.GetEncoding();
    if (encoding == sensor_msgs::image_encodings::BGR8) {
        inputConversion = cv::COLOR_BGR2RGB;
    } else if (encoding == sensor_msgs::image_encodings::RGBA8) {
        inputConversion = cv::COLOR_RGBA2RGB;
    } else if (encoding == sensor_msgs::image_encodings::RGB8) {
        inputConversion = std::nullopt;
    } else {
        throw std::runtime_error("Unknown image camera configuration " + encoding);
    }
}

void HumanDetectionNode::cameraImageCallback(const nitros::NitrosImageView& msg) {
    if (!isFullyConfigured) {
        configureNodeFromInitialImage(msg);
        RCLCPP_INFO(this->get_logger(), "Configured environment using initial frame correctly");
        // We don't process the current frame, as we're probably far behind due to having to configure the space
        return;
    }

    frameSampler->toggleFrame(true);

    // Mount the image.
    imageInput->copyImageFromNitros(msg, inputConversion);
    samModel->detect(imageInput);
    samModel->processOutput();

    trackCreateProcessor->outputMaskedImage(imageInput->getGpuImage(), overlayPercentage);

    auto finalMsg = imageInput->getGpuImage().createNitrosImageMessage(imageFrameId, this->get_clock()->now());
    frameSampler->toggleFrame(true);
    maskedImagePub->publish(std::move(finalMsg));
}

void HumanDetectionNode::configureNodeFromInitialImage(const nitros::NitrosImageView& image) {
    // Both these values are actually unsigned so have a larger range than int - 
    // sure it won't be a real problem!
    int imageHeight = (int)image.GetHeight();
    int imageWidth = (int)image.GetWidth();

    imageInput = std::make_shared<PersistentImageInput>(
        PersistentImageInputFactory().createPersistentImageInput(
            imageWidth,
            imageHeight,
            intermediateImageSize,
            intermediateImageSize,
            *samContext
        )
    );
    float textBoxScale = this->get_parameter("text_box_size").as_double();
    int textBoxSize = imageWidth > imageHeight ? imageWidth * textBoxScale : imageHeight * textBoxScale;
    std::unique_ptr<TextProvider> textProvider = std::make_unique<TextProviderImpl>(
        500, 
        this->get_parameter("cuda_device_id").as_int(),
        imageWidth * textBoxScale,
        imageHeight * textBoxScale,
        TextConfig(cv::FONT_HERSHEY_SIMPLEX, cv::Scalar(255, 0, 0, 255), 4, 1.0f)
    );
    auto builder = TrackAndCreateImageProcessorBuilder();
    builder
        .withDeviceIdFromContext(*samContext)
        .withFrameSampler(frameSampler)
        .withImageHeight(imageHeight)
        .withImageWidth(imageWidth)
        .withIntermediateHeight((intermediateImageSize * 2) / 7)
        .withIntermediateWidth((intermediateImageSize * 2) / 7)
        .withMasksCount(200)
        .withMinimumFramesToSample(10)
        .withTextProvider(std::move(textProvider))
        .withThreshold(threshold);
    trackCreateProcessor = std::make_unique<TrackAndCreateImageProcessor>(builder.build());

    samModel->registerOutputProcessor(trackCreateProcessor);
    configureCameraImageConversion(image);

    // SAM3 is now ready to start accepting frames
    isFullyConfigured = true;
}
