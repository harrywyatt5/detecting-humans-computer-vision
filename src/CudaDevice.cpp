#include "CudaDevice.h"

#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <stdexcept>
    
CudaDevice::CudaDevice(int id, CudaStreamPriority priority) : deviceId(id) {
    switchCudaDevice();

    if (priority == CudaStreamPriority::LET_OS_DECIDE) {
        cudaStreamCreate(&stream);
    } else {
        int lowest, highest;
        cudaDeviceGetStreamPriorityRange(&lowest, &highest);

        int chosenPriority;
        switch (priority) {
            case CudaStreamPriority::LOWEST:
                chosenPriority = lowest;
                break;
            case CudaStreamPriority::MEDIUM:
                chosenPriority = (highest + lowest) / 2;
                break;
            case CudaStreamPriority::HIGHEST:
                chosenPriority = highest;
                break;
            default:
                // This is kinda impossible lol
                chosenPriority = lowest;
                break;
        }
        cudaStreamCreateWithPriority(&stream, cudaStreamDefault, chosenPriority);
    }

    openCVStream = cv::cuda::StreamAccessor::wrapStream(stream);
}

int CudaDevice::getDeviceId() const {
    return deviceId;
}

void CudaDevice::switchCudaDevice() const {
    auto result = cudaSetDevice(deviceId);

    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("Could not switch to cuda device. Reason: ") + cudaGetErrorString(result));
    }
}

void CudaDevice::waitForCompletion() {
    cudaStreamSynchronize(stream);
}

cudaStream_t& CudaDevice::getCudaStream() {
    return stream;
}

cv::cuda::Stream& CudaDevice::getOpenCVCudaStream() {
    return openCVStream;
}

CudaDevice::~CudaDevice() {
    auto status = cudaStreamDestroy(stream);

    if (status != cudaSuccess) {
        std::cerr << "Could not destroy a CUDA stream. Stream is likely living past its scope. Reason: " << cudaGetErrorString(status) << std::endl;
    }
}
