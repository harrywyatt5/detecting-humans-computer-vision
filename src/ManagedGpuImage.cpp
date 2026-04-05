#include "ManagedGpuImage.h"

#include "GpuImage.h"
#include <std_msgs/msg/header.hpp>
#include <isaac_ros_nitros_image_type/nitros_image.hpp>
#include <isaac_ros_nitros_image_type/nitros_image_builder.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <stdexcept>
#include <iostream>

void ManagedGpuImage::releaseClaim() {
    if (!inUse) {
        std::cerr << "ManagedGpuImage is not in use. Ignoring request to unclaim it...\n";
    }

    inUse = false;
}

void ManagedGpuImage::claim() {
    if (inUse) {
        throw std::runtime_error("Cannot claim ManagedGpuImage because it has already been claimed");
    }

    inUse = true;
}

bool ManagedGpuImage::isBeingUsed() const {
    return inUse;
}

bool ManagedGpuImage::isBeingUsedByNitros() const {
    return inUseByNitros;
}

std::unique_ptr<sensor_msgs::msg::Image> ManagedGpuImage::createRos2ImageMessage(const std::string& frameName, rclcpp::Time broadcastTime) {
    // When we are defaling with a ros2 message (cpu) it's fairly easy - just unclaim the message and return the result of the base class
    auto result = GpuImage::createRos2ImageMessage(frameName, broadcastTime);
    releaseClaim();
    return result;
}

nitros::NitrosImage ManagedGpuImage::createNitrosImageMessage(const std::string& frameName, rclcpp::Time broadcastTime) {
    // This function is basically copied from GpuImage, apart from the fact that it properly initialises the NitrosImage
    // to unmark the GpuImage (or destroy it if it's been orthoned) after use
    cudaDevice->switchCudaDevice();

    if (isBeingUsed()) {
        throw std::runtime_error("Invalid state. This ManagedGpuImage should already be marked as 'in use' before creating a nitros message");
    }

    std_msgs::msg::Header header;
    header.stamp = broadcastTime;
    header.frame_id = frameName;

    auto nitrosImage = nitros::NitrosImageBuilder()
                        .WithDimensions(internalData.rows, internalData.cols)
                        .WithEncoding(sensor_msgs::image_encodings::RGB8)
                        .WithGpuData()
                        .
}
