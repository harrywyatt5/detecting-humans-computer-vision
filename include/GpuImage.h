#pragma once

#include "CudaDevice.h"
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <isaac_ros_nitros_image_type/nitros_image.hpp>
#include <rclcpp/rclcpp.hpp>
#include <memory>
#include <cstdint>

namespace nitros = nvidia::isaac_ros::nitros;

// GpuImage is just a composition of cv::cuda::GpuMat which allows for additional commands to be
// ran on it
// In some cases, such as when a GpuImage is spawned by OutImgSchedulerSingleton, a GpuImage may
// actually maintain its own internal buffer, rather than the underlying GpuMat
class GpuImage {
private:
    uint8_t* gpuBuffer;
    cv::cuda::GpuMat internalData;
    std::shared_ptr<CudaDevice> cudaDevice;

    void freeMemory();
    void throwIfDimensionMismatch(int cols, int rows) const;
public:
    GpuImage(cv::cuda::GpuMat mat, int devId);
    GpuImage(int width, int height, int devId);
    void uploadCpuImage(const cv::Mat& cpuImage);
    void copyFrom(const cv::cuda::GpuMat& gpuImage);
    void toNewColourTarget(cv::ColorConversionCodes code);
    void download(cv::Mat& target) const;
    int getWidth() const;
    int getHeight() const;
    std::unique_ptr<sensor_msgs::msg::Image> createRos2ImageMessage(const std::string& frameName, rclcpp::Time broadcastTime) const;
    nitros::NitrosImage createNitrosImageMessage(const std::string& frameName, rclcpp::Time broadcastTime) const;

    const cv::cuda::GpuMat& getConstGpuMat() const;
    cv::cuda::GpuMat& getMutableGpuMat();

    // Delete copy, implement move
    GpuImage(const GpuImage&) = delete;
    GpuImage& operator=(const GpuImage&) = delete;
    GpuImage(GpuImage&& other) noexcept;
    GpuImage& operator=(GpuImage&& other) noexcept;
    ~GpuImage();
};
