#pragma once

#include "PersistentSam3Model.h"
#include "CreateImageProcessor.h"
#include "Sam3Context.h"
#include "LoggingLevel.h"
#include "LanguageToken.h"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <memory>

class HumanDetectionNode : public rclcpp::Node {
private:
    std::unique_ptr<PersistentSam3Model> samModel;
    std::unique_ptr<PersistentImageInput> imageInput;
    std::unique_ptr<Sam3Context> samContext;
    std::shared_ptr<CreateImageProcessor> createImageProcessor;
    std::shared_ptr<LanguageToken> promptToken;
    bool isFullyConfigured;

    // ROS2 data
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr leftCameraSub;

    // Callbacks
    void leftImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);

    // Helpers
    void configureSam3Model(const LoggingLevel& loggingLevel);
    void mountPrompt();
    void configureNodeFromInitialImage(const sensor_msgs::msg::Image& image);
public:
    HumanDetectionNode();
};
