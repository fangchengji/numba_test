#include <normalize.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// cuda kernel
__global__ void normalize_kernel(const float* src, float* dst, int width, int height, int channel, const float* mean, const float* std) {

    // 2D Index of current thread
    const int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    if (index_x < width && index_y < height) {
        int img_idx = (index_y * width + index_x) * channel;
        for (int i = 0; i < channel; i++) {
            dst[img_idx + i] = ((float)src[img_idx + i] - mean[i]) / std[i]; 
        } 
    }
}

void normalize(float* src, float* dst, int width, int height, int channel, float* mean, float* std) {

    const dim3 block(32, 32);
    const dim3 grid((width + block.x  - 1) / block.x, (height + block.y - 1) / block.y);
    normalize_kernel<<<grid, block>>>(src, dst, width, height, channel, mean, std);
}


