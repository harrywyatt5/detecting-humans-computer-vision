#include "MappedMask.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cstdint>

__global__ void createTrackedMask(
    cv::cuda::PtrStepSz<unsigned char> mask,
    const float* masksStart,
    const MappedMask* mappedMaskStart,
    const int masksCount
) {
    // Each thread is in charge of a pixel across all of the masks
    int offsetX = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetY = blockIdx.y * blockDim.y + threadIdx.y;

    if (offsetX >= mask.cols || offsetY >= mask.rows) {
        return;
    }

    // Set the pixel to black to start off with if the output is just a random bit of memory
    // If outIsImage is set, it means an image has already been copied there, so let's not clobber it...
    mask(offsetY, offsetX) = 0;

    for (int i = 0; i < masksCount; ++i) {
        if (!mappedMaskStart[i].isPresent()) {
            continue;
        }

        int index = (i * mask.cols * mask.rows) + mask.cols * offsetY + offsetX;
        if (masksStart[index] > 0.0f) {
            // Record which detection it was so we can make a nice color image
            // 0 is a non-detection so we must bump to +1. No worries - we only ever
            // expect 200
            mask(offsetY, offsetX) = mappedMaskStart[i].getRemappedTarget() + 1;
            break;
        }
    }
}

void launchCreateTrackedMask(
    cv::cuda::GpuMat& maskOutput,
    cv::cuda::Stream& stream,
    const float* masksStart,
    const MappedMask* mappedMaskStart,
    const int masks
) {
    dim3 blocks(16, 16);
    dim3 grid((maskOutput.cols + blocks.x - 1) / blocks.x, (maskOutput.rows + blocks.y - 1) / blocks.y);

    cudaStream_t cudaStream = cv::cuda::StreamAccessor::getStream(stream);
    createTrackedMask<<<grid, blocks, 0, cudaStream>>>(maskOutput, masksStart, mappedMaskStart, masks);
}
