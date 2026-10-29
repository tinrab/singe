// Optional reference: nvcc --ptx -arch=sm_80 -O3 transpose.cu
extern "C" __global__ void transpose(const float* input, float* output, unsigned width) {
    unsigned x = blockIdx.x * 16 + threadIdx.x;
    unsigned y = blockIdx.y * 16 + threadIdx.y;
    if (x < width && y < width) output[x * width + y] = input[y * width + x];
}
