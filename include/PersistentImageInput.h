#pragma once

#include "CudaTensor.h"
#include "GpuImage.h"
#include "ImageProvider.h"
#include <sensor_msgs/msg/image.hpp>
#include <memory>
#include <vector>
#include <cstdint>
#include <string>
#include <optional>
#include <opencv2/opencv.hpp>

class PersistentImageInput : public ImageProvider {
private:
    std::shared_ptr<GpuImage> gpuImage;
    cv::cuda::GpuMat resizedImage;
    std::shared_ptr<cv::cuda::Stream> stream;
    int x;
    int y;
    int resizedX;
    int resizedY;

    bool hasUploadedImage;
public:
    PersistentImageInput(int imageX, int imageY, int resizedX, int resizedY, int deviceId);
    void uploadImageFromDisk(const std::string& path);
    void uploadImageFromSensorMsg(const sensor_msgs::msg::Image& image, const std::optional<cv::ColorConversionCodes> conversion);
    void writeImageToCudaTensor(CudaTensor<float>& tensor) override;

    std::shared_ptr<GpuImage> getMutableGpuImage();
    std::shared_ptr<const GpuImage> getConstGpuImage() const;
    int getOriginalX() const override;
    int getOriginalY() const override;
};
