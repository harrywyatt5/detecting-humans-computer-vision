#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cstdint>

__global__ void createMask(
    cv::cuda::PtrStepSz<unsigned char> mask,
    const float* masksStart,
    const uint8_t* masksInclude,
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
        if (masksInclude[i] == 0) {
            continue;
        }

        int index = (i * mask.cols * mask.rows) + mask.cols * offsetY + offsetX;
        if (masksStart[index] > 0.0f) {
            // Record which detection it was so we can make a nice color image
            // 0 is a non-detection so we must bump to +1. No worries - we only ever
            // expect 200
            mask(offsetY, offsetX) = i + 1;
            break;
        }
    }
}

__global__ void createImage(
    const cv::cuda::PtrStepSz<unsigned char> mask,
    cv::cuda::PtrStepSz<uchar3> image,
    const int mixPercentage,
    const int invMixPercentage
) {
    // Each thread is in charge of a pixel across all of the masks
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= mask.cols || y >= mask.rows) {
        return;
    }

    int maskValue = (int)mask(y, x);
    if (maskValue == 0) {
        return;
    }

    uchar3 mixColour = make_uchar3(
        (unsigned char)(maskValue * 137),
        (unsigned char)(maskValue * 83),
        (unsigned char)(maskValue * 211)
    );
    uchar3 currentPixelColour = image(y, x);
    // >> 8 is same as dividing through by 256, but faster
    image(y, x) = make_uchar3(
        (unsigned char)((invMixPercentage * currentPixelColour.x + mixPercentage * mixColour.x) >> 8),
        (unsigned char)((invMixPercentage * currentPixelColour.y + mixPercentage * mixColour.y) >> 8),
        (unsigned char)((invMixPercentage * currentPixelColour.z + mixPercentage * mixColour.z) >> 8)
    );
}

void launchCreateMask(
    cv::cuda::GpuMat& maskOutput,
    cv::cuda::Stream& stream,
    const float* masksStart,
    const uint8_t* masksInclude,
    const int masks
) {
    dim3 blocks(16, 16);
    dim3 grid((maskOutput.cols + blocks.x - 1) / blocks.x, (maskOutput.rows + blocks.y - 1) / blocks.y);

    cudaStream_t cudaStream = cv::cuda::StreamAccessor::getStream(stream);
    createMask<<<grid, blocks, 0, cudaStream>>>(maskOutput, masksStart, masksInclude, masks);
}

void launchCreateImage(
    const cv::cuda::GpuMat& maskInput,
    cv::cuda::GpuMat& outputImage,
    cv::cuda::Stream& stream,
    const float mixPercentage
) {
    dim3 blocks(16, 16);
    dim3 grid((maskInput.cols + blocks.x - 1) / blocks.x, (maskInput.rows + blocks.y - 1) / blocks.y);

    cudaStream_t cudaStream = cv::cuda::StreamAccessor::getStream(stream);
    int intMix = (int)std::round(mixPercentage * 256.0f);
    int invIntMix = 256 - intMix;
    createImage<<<grid, blocks, 0, cudaStream>>>(maskInput, outputImage, intMix, invIntMix);
}
