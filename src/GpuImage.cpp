#include "GpuImage.h"

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <rclcpp/rclcpp.hpp>
#include <memory>
#include <stdexcept>

void GpuImage::uploadCpuImage(const cv::Mat& cpuImage, std::shared_ptr<cv::cuda::Stream> stream) {
    setCudaDevice();

    if (stream != nullptr) {
        internalData.upload(cpuImage, *stream);
    } else {
        internalData.upload(cpuImage);
    }

}

void GpuImage::setCudaDevice() const {
    cv::cuda::setDevice(deviceId);
}

void GpuImage::toNewColourTarget(cv::ColorConversionCodes code) {
    setCudaDevice();
    cv::cuda::GpuMat tempMat;
    cv::cuda::cvtColor(internalData, tempMat, code);

    internalData.swap(tempMat);
}

void GpuImage::download(cv::Mat& target) const {
    setCudaDevice();
    internalData.download(target);
}

int GpuImage::getHeight() const {
    return internalData.rows;
}

int GpuImage::getWidth() const {
    return internalData.cols;
}

const cv::cuda::GpuMat& GpuImage::getConstGpuMat() const {
    return internalData;
}

cv::cuda::GpuMat& GpuImage::getMutableGpuMat() {
    return internalData;
}

std::unique_ptr<sensor_msgs::msg::Image> GpuImage::createRos2ImageMessage(const std::string& frameName, rclcpp::Time broadcastTime) const {
    setCudaDevice();
    auto msg = std::make_unique<sensor_msgs::msg::Image>();

    msg->header.stamp = broadcastTime;
    msg->header.frame_id = frameName;
    msg->height = internalData.rows;
    msg->width = internalData.cols;
    msg->step = internalData.cols * 3; // Each pixel has three values. Don't have to care about padding because download strips it
    msg->encoding = sensor_msgs::image_encodings::RGB8;

    msg->data.resize(msg->height * msg->step);

    cv::Mat cpuImage(
        internalData.rows,
        internalData.cols,
        CV_8UC3,
        msg->data.data()
    );
    internalData.download(cpuImage);

    return msg;
}
