#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cstdint>

__global__ void createMask(
    cv::cuda::PtrStepSz<unsigned char> image,
    const float* masksStart,
    const uint8_t* masksInclude,
    const int masks
) {
    // Each thread is in charge of a pixel across all of the masks
    int offsetX = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetY = blockIdx.y * blockDim.y + threadIdx.y;

    if (offsetX >= image.cols || offsetY >= image.rows) {
        return;
    }

    // Set the pixel to black to start off with if the output is just a random bit of memory
    // If outIsImage is set, it means an image has already been copied there, so let's not clobber it...
    image(offsetY, offsetX) = 0;

    for (int i = 0; i < masks; ++i) {
        if (masksInclude[i] == 0) {
            continue;
        }

        int index = (i * image.cols * image.rows) + image.cols * offsetY + offsetX;
        if (masksStart[index] > 0.0f) {
            image(offsetY, offsetX) = 255;
            break;
        }
    }
}

void launchCreateImage(
    cv::cuda::GpuMat& output,
    cv::cuda::Stream& stream,
    const float* masksStart,
    const uint8_t* masksInclude,
    const int masks
) {
    dim3 blocks(16, 16);
    dim3 grid((output.cols + blocks.x - 1) / blocks.x, (output.rows + blocks.y - 1) / blocks.y);

    cudaStream_t cudaStream = cv::cuda::StreamAccessor::getStream(stream);
    createMask<<<grid, blocks, 0, cudaStream>>>(output, masksStart, masksInclude, masks);
}
// TODO: Come here and program this so it calculates a mask and applies it to the original image