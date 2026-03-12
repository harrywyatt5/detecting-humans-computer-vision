#pragma once

#include "PersistentSam3Model.h"
#include "CreateImageProcessor.h"
#include "Sam3Context.h"
#include "LoggingLevel.h"
#include "LanguageToken.h"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>
#include <memory>
#include <optional>

class HumanDetectionNode : public rclcpp::Node {
private:
    std::unique_ptr<PersistentSam3Model> samModel;
    std::unique_ptr<Sam3Context> samContext;
    std::shared_ptr<PersistentImageInput> imageInput;
    std::shared_ptr<CreateImageProcessor> createImageProcessor;
    std::shared_ptr<LanguageToken> promptToken;
    std::optional<cv::ColorConversionCodes> inputConversion;
    // Any parameters which are needed in the main runtime loop, we cache here
    std::string imageFrameId; 
    float threshold;
    bool isFullyConfigured;

    // ROS2 data
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr leftCameraSub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr maskedImagePub;

    // Callbacks
    void leftImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);

    // Helpers
    void configureSam3Model(const LoggingLevel& loggingLevel);
    void mountPrompt();
    void configureCameraImageConversion(const sensor_msgs::msg::Image& image);
    void configureNodeFromInitialImage(const sensor_msgs::msg::Image& image);
public:
    HumanDetectionNode();
};
