#include "HumanDetectionNode.h"

#include "LanguageToken.h"
#include "Sam3ContextBuilder.h"
#include "LoggingLevel.h"
#include "PersistentSam3Model.h"
#include "PersistentImageInputFactory.h"
#include "CreateImageProcessor.h"
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <onnxruntime_cxx_api.h>
#include <sensor_msgs/msg/image.hpp>
#include <cstdlib>
#include <string>
#include <filesystem>

HumanDetectionNode::HumanDetectionNode() 
    : samModel(nullptr),
        imageInput(nullptr),
        samContext(nullptr),
        promptToken(nullptr),
        createImageProcessor(nullptr),
        isFullyConfigured(false),
        Node("human_detection_node")
{
    auto shareLocation = ament_index_cpp::get_package_share_directory("real_time_humans");

    this->declare_parameter<std::string>("sam3_engine_cache_dir", std::getenv("HOME") + "/.cache/detect_humans");
    this->declare_parameter<int>("max_cpu_threads", 1);
    this->declare_parameter<bool>("use_fp16", true);
    this->declare_parameter<int>("cuda_device", 0);
    this->declare_parameter<std::string>("log_level", "info");
    this->declare_parameter<std::string>("sam3_text_encoder_path", shareLocation + "/sam3-onnx/text-encoder-fp16.onnx");
    this->declare_parameter<std::string>("sam3_vision_encoder_path", shareLocation + "/sam3-onnx/vision-encoder-fp16.onnx");
    this->declare_parameter<std::string>("sam3_decoder_path", shareLocation + "/sam3-onnx/geo-encoder-mask-decoder-fp16.onnx");
    this->declare_parameter<std::string>("encoded_prompt_path", shareLocation + "language.token");
    this->declare_parameter<long long>("maximum_vram", 6442450944LL); // TODO: allow input that is more readable?
    this->declare_parameter<int>("cuda_device_id", 0);
    this->declare_parameter<float>("threshold", 0.85f);
    this->declare_parameter<std::string>("camera_left_topic", "");
    this->declare_parameter<std::string>("camera_right_topic", "");

    // Prepare subscribers
    rclcpp::QoS qosProfile(1);
    qosProfile.keep_last(1);
    qosProfile.best_effort();
    leftCameraSub = this->create_subscription<sensor_msgs::msg::Image>(
        this->get_parameter<std::string>("camera_left_topic"),
        qosProfile,
        std::bind(&HumanDetectionNode::leftImageCallback, std::placeholders::_1)
    );

    // Configurables
    auto loggingLevel = LoggingLevel::fromString(this->get_parameter<std::string>("log_level"), true);
    promptToken = LanguageToken::createFromFile(this->get_parameter<std::string>("encoded_prompt_path"));
    configureSam3Model(loggingLevel);
    mountPrompt();
}

void HumanDetectionNode::configureSam3Model(const LoggingLevel& loggingLevel) {
    context = Sam3ContextBuilder()
                .withApplicationName("real_time_humans")
                .withBatchLimit(1)
                .withNumBoxesLimit(1)
                .withCPUThreadMax(1)
                .withTextEncoderPath(this->get_parameter<std::string>("sam3_text_encoder_path"))
                .withVisionEncoderPath(this->get_parameter<std::string>("sam3_vision_encoder_path"))
                .withDecoderPath(this->get_parameter<std::string>("sam3_decoder_path"))
                .withFP16Enabled(true)
                .withDeviceId(this->get_parameter<int>("cuda_device_id"))
                .withEngineCacheDir(this->get_parameter<std::string>("sam3_engine_cache_dir"))
                .withGraphOptimistionLevel(GraphOptimizationLevel::ORT_ENABLE_ALL)
                .withLoggingLevel(loggingLevel.toOrtLoggingLevel())
                .withMaxGPUMemory(this->get_parameter<long long>("maximum_vram"))
                .withCudaGraphsEnabled(false)
                .build();
    samModel = PersistentSam3Model::createSam3Model(*context);
}

void HumanDetectionNode::mountPrompt() {
    samModel->mountAndCalculatePrompt(promptToken);
}

void HumanDetectionNode::leftImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
    if (!isFullyConfigured) {
        configureNodeFromInitialImage(*msg);
        // We don't process the current frame, as we're probably far behind due to having to configure the space
        return;
    }

    
    cv::COLOR_BGR2GRAY
}

void HumanDetectionNode::configureNodeFromInitialImage(const sensor_msgs::msg::Image& image) {
    int imageHeight = (int)image.height;
    int imageWidth = (int)image.width;

    imageInput = PersistentImageInputFactory().createPersistentImageInput(imageWidth, imageHeight, 1008, 1008, *samContext);
    createImageProcessor = CreateImageProcessor::createCreateImageProcessor(
        imageWidth,
        imageHeight,
        1008,
        1008,
        200, 
        this->get_parameter<float>("threshold")
    );

    samModel->registerOutputProcessor(createImageProcessor);

    // SAM3 is now ready to start accepting frames
    isFullyConfigured = true;
}
