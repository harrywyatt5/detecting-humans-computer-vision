#pragma once

#include "PersistentSam3Model.h"
#include "TrackAndCreateImageProcessor.h"
#include "Sam3Context.h"
#include "LoggingLevel.h"
#include "LanguageToken.h"
#include "FrameSampler.h"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>
#include <isaac_ros_managed_nitros/managed_nitros_publisher.hpp>
#include <isaac_ros_managed_nitros/managed_nitros_subscriber.hpp>
#include <isaac_ros_nitros_image_type/nitros_image_view.hpp>
#include <memory>
#include <optional>

namespace nitros = nvidia::isaac_ros::nitros;

class HumanDetectionNode : public rclcpp::Node {
private:
    std::unique_ptr<PersistentSam3Model> samModel;
    std::unique_ptr<Sam3Context> samContext;
    std::shared_ptr<PersistentImageInput> imageInput;
    std::shared_ptr<TrackAndCreateImageProcessor> trackCreateProcessor;
    std::shared_ptr<LanguageToken> promptToken;
    std::shared_ptr<FrameSampler> frameSampler;
    std::optional<cv::ColorConversionCodes> inputConversion;
    // Any parameters which are needed in the main runtime loop, we cache here
    std::string imageFrameId;
    float threshold;
    bool isFullyConfigured;
    int intermediateImageSize;

    // ROS2 data
    std::shared_ptr<nitros::ManagedNitrosSubscriber<nitros::NitrosImageView>> cameraSub;
    std::shared_ptr<nitros::ManagedNitrosPublisher<nitros::NitrosImage>> maskedImagePub;

    // Callbacks
    void cameraImageCallback(const nitros::NitrosImageView& msg);

    // Helpers
    void configureSam3Model(const LoggingLevel& loggingLevel);
    void mountPrompt();
    void configureCameraImageConversion(const nitros::NitrosImageView& image);
    void configureNodeFromInitialImage(const nitros::NitrosImageView& image);
public:
    HumanDetectionNode();
};
