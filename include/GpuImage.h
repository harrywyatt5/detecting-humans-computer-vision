#pragma once

#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <rclcpp/rclcpp.hpp>
#include <memory>

// GpuImage is just a composition of cv::cuda::GpuMat which allows for additional commands to be
// ran on it
class GpuImage {
private:
    cv::cuda::GpuMat internalData;
    int deviceId;

    void setCudaDevice() const;
public:
    GpuImage(cv::cuda::GpuMat mat, int devId) : internalData(std::move(mat)), deviceId(devId) {}
    void uploadCpuImage(const cv::Mat& cpuImage, std::shared_ptr<cv::cuda::Stream> stream = nullptr);
    void toNewColourTarget(const cv::ColorConversionCodes code);
    void download(cv::Mat& target) const;
    int getWidth() const;
    int getHeight() const;
    std::unique_ptr<sensor_msgs::msg::Image> createRos2ImageMessage(const std::string& frameName, rclcpp::Time broadcastTime) const;
    const cv::cuda::GpuMat& getConstGpuMat() const;
    cv::cuda::GpuMat& getMutableGpuMat();
};
