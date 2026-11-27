#include <nccl_device.h>

extern "C" __global__ void gin_all_to_all(ncclWindow_t send_window, size_t send_offset, ncclWindow_t receive_window,
                                          size_t receive_offset, size_t count, ncclDevComm comm) {
  ncclGin gin{comm, 0};
  unsigned int signal = blockIdx.x;
  uint64_t signal_value = gin.readSignal(signal);

  ncclGinBarrierSession<ncclCoopCta> barrier{ncclCoopCta(), gin, ncclTeamTagWorld(), blockIdx.x};
  barrier.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::None);

  const int thread = threadIdx.x + blockIdx.x * blockDim.x;
  const int thread_count = blockDim.x * gridDim.x;
  const size_t size = count * sizeof(float);
  for (int peer = thread; peer < comm.nRanks; peer += thread_count) {
    gin.put(ncclTeamWorld(comm), peer, receive_window, receive_offset + comm.rank * size, send_window,
            send_offset + peer * size, size, ncclGin_WeakSignalInc{signal});
  }

  const int receiving_cta = (comm.rank % thread_count) / blockDim.x;
  if (blockIdx.x == receiving_cta) {
    gin.waitSignal(ncclCoopCta(), signal, signal_value + comm.nRanks);
  }
  gin.flush(ncclCoopCta());
  barrier.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::None);
}
