#pragma once

#include "CudaTensor.h"
#include "GpuImage.h"
#include "ImageProvider.h"
#include "CudaDevicesSingleton.h"
#include "CudaDevice.h"
#include "ManagedGpuImage.h"
#include "OutImgScheduler.h"
#include <sensor_msgs/msg/image.hpp>
#include <isaac_ros_nitros_image_type/nitros_image_view.hpp>
#include <memory>
#include <vector>
#include <cstdint>
#include <string>
#include <optional>
#include <opencv2/opencv.hpp>

namespace nitros = nvidia::isaac_ros::nitros;

class PersistentImageInput : public ImageProvider {
private:
    OutImgScheduler scheduler;
    std::shared_ptr<ManagedGpuImage> gpuImage;
    cv::cuda::GpuMat resizedImage;
    uint8_t* pinnedStaticPtr;
    std::shared_ptr<CudaDevice> cudaDevice;
    int x;
    int y;
    int resizedX;
    int resizedY;

    bool hasUploadedImage;
    void cycleGpuImage();
    void copyToPinnedMemory(const uint8_t* source, int step);
    void allocatePinnedMem();
public:
    PersistentImageInput(int imageX, int imageY, int resizedX, int resizedY, OutImgScheduler outImgScheduler, int deviceId);
    void uploadImageFromDisk(const std::string& path);
    void uploadImageFromSensorMsg(const sensor_msgs::msg::Image& image, const std::optional<cv::ColorConversionCodes> conversion);
    void copyImageFromNitros(const nitros::NitrosImageView& image, const std::optional<cv::ColorConversionCodes> conversion);
    void writeImageToCudaTensor(CudaTensor<float>& tensor) override;

    GpuImage& getGpuImage();
    const GpuImage& getGpuImage() const;
    int getOriginalX() const override;
    int getOriginalY() const override;

    ~PersistentImageInput();
    PersistentImageInput(PersistentImageInput&& other) noexcept;
    PersistentImageInput(const PersistentImageInput&) = delete;
    PersistentImageInput& operator=(const PersistentImageInput&) = delete;
};
