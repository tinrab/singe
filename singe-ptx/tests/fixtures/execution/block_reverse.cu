// Compiler: CUDA compilation tools, release 13.4, V13.4.92.
// Target: sm_80, PTX 9.4, 64-bit addresses.
// Regenerate from this directory with:
// nvcc --ptx --gpu-architecture=sm_80 --std=c++17 -O3 block_reverse.cu -o block_reverse.ptx

extern "C" __global__ void block_reverse(const int *input, int *output,
                                        unsigned count) {
    extern __shared__ int tile[];
    unsigned lane = threadIdx.x;
    unsigned start = blockIdx.x * blockDim.x;
    unsigned index = start + lane;
    tile[lane] = index < count ? input[index] : 0;
    __syncthreads();
    if (index < count) {
        unsigned remaining = count - start;
        unsigned length = remaining < blockDim.x ? remaining : blockDim.x;
        output[index] = tile[length - 1 - lane];
    }
}
