#include <nccl_device.h>

extern "C" __global__ void multimem_all_reduce(ncclDevComm comm, ncclWindow_t window, size_t offset, size_t count) {
  ncclLsaBarrierSession<ncclCoopCta> barrier{ncclCoopCta(), comm, ncclTeamTagLsa(), blockIdx.x, true};
  barrier.sync(ncclCoopCta(), cuda::memory_order_acquire);

  float* multicast = (float*)ncclGetLsaMultimemPointer(window, offset, comm);
  const int rank = comm.lsaRank;
  const int rank_count = comm.lsaSize;
  const int thread = threadIdx.x + blockDim.x * (rank + blockIdx.x * rank_count);
  const int thread_count = blockDim.x * gridDim.x * rank_count;
  for (size_t index = thread; index < count; index += thread_count) {
    float sum;
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];" : "=f"(sum) : "l"(multicast + index) : "memory");
    asm volatile("multimem.st.relaxed.sys.global.f32 [%0], %1;" ::"l"(multicast + index), "f"(sum) : "memory");
  }

  barrier.sync(ncclCoopCta(), cuda::memory_order_release);
}
