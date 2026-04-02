#pragma once

#include "GpuImage.h"
#include "CudaDevice.h"
#include <opencv2/opencv.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <isaac_ros_nitros_image_type/nitros_image.hpp>
#include <std_msgs/msg/header.hpp>
#include <cstdint>
#include <memory>

namespace nitros = nvidia::isaac_ros::nitros;

class OutputImage {
private:
    int imageX;
    int imageY;
    std::shared_ptr<CudaDevice> device;
    uint8_t* data;
    bool isInUse;
public:
    OutputImage(int x, int y, int devId);

    int getX() const;
    int getY() const;
    uint8_t* getData();
    const uint8_t* getData() const;
    cv::cuda::GpuMat getDataAsGpuImage();
    nitros::NitrosImage createNitrosMessage(const std_msgs::msg::Header& header, const std::string& encoding);

    bool isClaimable() const;
    void claimBuffer();
    void releaseBuffer();
    ~OutputImage();
};
