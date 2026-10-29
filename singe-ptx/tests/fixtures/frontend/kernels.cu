// Regenerate from this directory with:
// nvcc --ptx --gpu-architecture=sm_80 --std=c++17 -O3 kernels.cu -o kernels.ptx

extern "C" __global__ void vector_add(const unsigned *x, const unsigned *y,
                                    unsigned *out, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = x[i] + y[i];
}

extern "C" __global__ void saxpy(const float *x, float *y, float scale,
                               unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) y[i] = scale * x[i] + y[i];
}

extern "C" __global__ void block_sum(const float *input, float *output,
                                   unsigned count) {
    __shared__ float values[128];
    unsigned lane = threadIdx.x;
    unsigned i = blockIdx.x * 128 + lane;
    values[lane] = i < count ? input[i] : 0.0f;
    __syncthreads();
    for (unsigned stride = 64; stride != 0; stride >>= 1) {
        if (lane < stride) values[lane] += values[lane + stride];
        __syncthreads();
    }
    if (lane == 0) output[blockIdx.x] = values[0];
}

extern "C" __global__ void tiled_matmul(const float *a, const float *b,
                                      float *out, unsigned width) {
    __shared__ float left[16][16];
    __shared__ float right[16][16];
    unsigned x = threadIdx.x;
    unsigned y = threadIdx.y;
    unsigned row = blockIdx.y * 16 + y;
    unsigned col = blockIdx.x * 16 + x;
    float sum = 0.0f;
    for (unsigned start = 0; start < width; start += 16) {
        left[y][x] = row < width && start + x < width
            ? a[row * width + start + x] : 0.0f;
        right[y][x] = col < width && start + y < width
            ? b[(start + y) * width + col] : 0.0f;
        __syncthreads();
        for (unsigned k = 0; k < 16; ++k) sum += left[y][k] * right[k][x];
        __syncthreads();
    }
    if (row < width && col < width) out[row * width + col] = sum;
}
