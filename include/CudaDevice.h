#pragma once

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

class CudaDevice {
private:
    int deviceId;
    cudaStream_t stream;
    cv::cuda::Stream openCVStream;
public:
    enum class CudaStreamPriority {
        LOWEST = 0,
        MEDIUM = 1,
        HIGHEST = 2,
        LET_OS_DECIDE = 3
    };
    CudaDevice(int id, CudaStreamPriority priority);

    int getDeviceId() const;
    void switchCudaDevice() const;
    void waitForCompletion();
    cudaStream_t& getCudaStream();
    cv::cuda::Stream& getOpenCVCudaStream();

    ~CudaDevice();
};
