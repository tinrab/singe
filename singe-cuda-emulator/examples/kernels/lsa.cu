#include <nccl_device.h>

extern "C" __global__ void lsa_all_reduce(ncclDevComm comm, ncclWindow_t window, size_t offset, size_t count) {
  ncclLsaBarrierSession<ncclCoopCta> barrier{ncclCoopCta(), comm, ncclTeamTagLsa(), blockIdx.x};
  barrier.sync(ncclCoopCta(), cuda::memory_order_acquire);

  const int rank = comm.lsaRank;
  const int rank_count = comm.lsaSize;
  const int thread = threadIdx.x + blockDim.x * (rank + blockIdx.x * rank_count);
  const int thread_count = blockDim.x * gridDim.x * rank_count;
  for (size_t index = thread; index < count; index += thread_count) {
    float sum = 0;
    for (int peer = 0; peer < rank_count; peer++) {
      sum += ((float*)ncclGetLsaPointer(window, offset, peer))[index];
    }
    for (int peer = 0; peer < rank_count; peer++) {
      ((float*)ncclGetLsaPointer(window, offset, peer))[index] = sum;
    }
  }

  barrier.sync(ncclCoopCta(), cuda::memory_order_release);
}

extern "C" __global__ void lsa_barrier(ncclDevComm comm, unsigned int* arrivals, unsigned int rounds) {
  ncclLsaBarrierSession<ncclCoopCta> barrier{ncclCoopCta(), comm, ncclTeamTagLsa(), blockIdx.x};
  for (unsigned int round = 0; round < rounds; round++) {
    barrier.sync(ncclCoopCta(), cuda::memory_order_acq_rel);
    if (threadIdx.x == 0) {
      arrivals[blockIdx.x] += 1;
    }
  }
}

extern "C" __global__ void add_peer(float* local, const float* peer, size_t count) {
  for (size_t index = threadIdx.x + blockIdx.x * blockDim.x; index < count; index += blockDim.x * gridDim.x) {
    local[index] += peer[index];
  }
}
