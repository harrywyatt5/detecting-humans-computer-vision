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
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
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
        isFullyConfigured(false),
        Node("human_detection_node")
{
    auto shareLocation = ament_index_cpp::get_package_share_directory("real_time_humans");

    this->declare_parameter<std::string>("sam3_engine_cache_dir", std::getenv("HOME") + std::string("/.cache/detect_humans"));
    this->declare_parameter<int>("max_cpu_threads", 1);
    this->declare_parameter<bool>("use_fp16", true);
    this->declare_parameter<int>("cuda_device", 0);
    this->declare_parameter<std::string>("log_level", "error");
    this->declare_parameter<std::string>("sam3_text_encoder_path", shareLocation + "/sam3-onnx/text-encoder-fp16.onnx");
    this->declare_parameter<std::string>("sam3_vision_encoder_path", shareLocation + "/sam3-onnx/vision-encoder-fp16.onnx");
    this->declare_parameter<std::string>("sam3_decoder_path", shareLocation + "/sam3-onnx/geo-encoder-mask-decoder-fp16.onnx");
    this->declare_parameter<std::string>("encoded_prompt_path", shareLocation + "/language.token");
    this->declare_parameter<int64_t>("maximum_vram", 6442450944LL); // TODO: allow input that is more readable?
    this->declare_parameter<int>("cuda_device_id", 0);
    this->declare_parameter<float>("threshold", 0.85f);
    this->declare_parameter<std::string>("camera_left_topic", "/left_eye_cam");
    this->declare_parameter<std::string>("camera_right_topic", "/right_eye_cam");
    this->declare_parameter<std::string>("masked_image_topic", "masked_image");
    this->declare_parameter<std::string>("masked_image_frame_id", "image_frame");

    // Prepare subscribers
    rclcpp::QoS qosProfile(1);
    qosProfile.keep_last(1);
    qosProfile.best_effort();
    leftCameraSub = this->create_subscription<sensor_msgs::msg::Image>(
        this->get_parameter("camera_left_topic").as_string(),
        qosProfile,
        std::bind(&HumanDetectionNode::leftImageCallback, this, std::placeholders::_1)
    );
    maskedImagePub = this->create_publisher<sensor_msgs::msg::Image>(
        this->get_parameter("masked_image_topic").as_string(),
        10
    );

    // Configurables
    threshold = this->get_parameter("threshold").as_double();
    imageFrameId = this->get_parameter("masked_image_frame_id").as_string();
    auto loggingLevel = LoggingLevel::fromString(this->get_parameter("log_level").as_string(), true);
    promptToken = std::make_shared<LanguageToken>(LanguageToken::createFromFile(this->get_parameter("encoded_prompt_path").as_string()));
    RCLCPP_INFO(this->get_logger(), "About to load Sam3Model...");
    configureSam3Model(loggingLevel);
    RCLCPP_INFO(this->get_logger(), "Sam3 model was loaded. Compiling engine for language prompt and then will be ready!");
    mountPrompt();
    RCLCPP_INFO(this->get_logger(), "Done! Ready to receive events");
}

void HumanDetectionNode::configureSam3Model(const LoggingLevel& loggingLevel) {
    auto cudaDeviceId = this->get_parameter("cuda_device_id").as_int();
    auto builder = Sam3ContextBuilder()
                    .withApplicationName("real_time_humans")
                    .withBatchLimit(1)
                    .withNumBoxesLimit(1)
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
    samContext = std::make_unique<Sam3Context>(builder.build());
    samModel = std::make_unique<PersistentSam3Model>(PersistentSam3Model::createSam3Model(*samContext));
}

void HumanDetectionNode::mountPrompt() {
    samModel->mountAndCalculatePrompt(promptToken);
}

void HumanDetectionNode::configureCameraImageConversion(const sensor_msgs::msg::Image& img) {
    auto encoding = img.encoding;
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

void HumanDetectionNode::leftImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
    if (!isFullyConfigured) {
        configureNodeFromInitialImage(*msg);
        RCLCPP_INFO(this->get_logger(), "Configured environment using initial frame correctly");
        // We don't process the current frame, as we're probably far behind due to having to configure the space
        return;
    }

    frameSampler->toggleFrame(true);

    // Mount the image. This is zero copy on the CPU (although has to be uploaded to GPU and resized)
    imageInput->uploadImageFromSensorMsg(*msg, inputConversion);
    samModel->detect(imageInput);
    samModel->processOutput();

    trackCreateProcessor->outputMaskedImage(*imageInput->getMutableGpuImage(), threshold);

    auto finalMsg = imageInput->getConstGpuImage()->createRos2ImageMessage(imageFrameId, this->get_clock()->now());
    frameSampler->toggleFrame(true);
    maskedImagePub->publish(std::move(finalMsg));
}

void HumanDetectionNode::configureNodeFromInitialImage(const sensor_msgs::msg::Image& image) {
    int imageHeight = (int)image.height;
    int imageWidth = (int)image.width;

    imageInput = std::make_shared<PersistentImageInput>(PersistentImageInputFactory().createPersistentImageInput(imageWidth, imageHeight, 1008, 1008, *samContext));
    auto builder = TrackAndCreateImageProcessorBuilder()
                    .withDeviceIdFromContext(*samContext)
                    .withFrameSampler(frameSampler)
                    .withImageHeight(imageHeight)
                    .withImageWidth(imageWidth)
                    .withIntermediateHeight(288)
                    .withIntermediateWidth(288)
                    .withMasksCount(200)
                    .withMinimumFramesToSample(10)
                    .withThreshold(threshold);
    trackCreateProcessor = std::make_unique<TrackAndCreateImageProcessor>(builder.build());

    samModel->registerOutputProcessor(trackCreateProcessor);
    configureCameraImageConversion(image);

    // SAM3 is now ready to start accepting frames
    isFullyConfigured = true;
}
