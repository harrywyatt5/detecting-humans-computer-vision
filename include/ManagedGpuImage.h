#pragma once

#include "GpuImage.h"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <isaac_ros_nitros_image_type/nitros_image.hpp>
#include <string>

namespace nitros = nvidia::isaac_ros::nitros;

class ManagedGpuImage : public GpuImage {
private:
    // inUse is marked when the ManagedGpuImage is being used by someone in the application
    // inUseByNitros can only be marked when inUse is marked, and indicates that
    // Nitros is using the internal message for a message. This is for the very rare
    // case that the deconstructor is called on the object but it is still being used
    // by Nitros, then we can just let Nitros manage the buffer
    bool inUse;
    bool inUseByNitros;

    void delegateToNitros();
public:
    ManagedGpuImage(int x, int y, int devId) : inUse(false), inUseByNitros(false), GpuImage(x, y, devId) {}

    void releaseClaim();
    void claim();
    bool isBeingUsed() const;
    bool isBeingUsedByNitros() const;
    std::unique_ptr<sensor_msgs::msg::Image> createRos2ImageMessage(const std::string& frameName, rclcpp::Time broadcastTime) override;
    nitros::NitrosImage createNitrosImageMessage(const std::string& frameName, rclcpp::Time broadcastTime) override;

    // Delete copy, implement move
    ManagedGpuImage(const ManagedGpuImage&) = delete;
    ManagedGpuImage& operator=(const ManagedGpuImage&) = delete;
    ManagedGpuImage(ManagedGpuImage&& other) noexcept;
    ManagedGpuImage& operator=(ManagedGpuImage&& other) noexcept;
    ~ManagedGpuImage() override;
};
