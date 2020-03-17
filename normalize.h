#include <cuda_runtime_api.h>
#include <gpu_timer.h>


struct Shape {
    Shape(unsigned int height, unsigned int width, unsigned int channel) : h(height), w(width), c(channel) {}; 
    unsigned int h;
    unsigned int w;
    unsigned int c;
};

void normalize(float* src, float* dst, int width, int height, int channel, float* mean, float* std);

