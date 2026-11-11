use super::*;

use crate::backend::cuda::CudaStream;

#[test]
fn test_cuda_backend_rejects_tuned_cublaslt_warmup_during_stream_capture() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let device = Device::Cuda(0);
    let Some(stream) = skip_if_stub(CudaStream::create(device)) else {
        return;
    };
    let shape = Shape::create([1, 1]).unwrap();
    let a = Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32]).unwrap();
    let b = Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32]).unwrap();
    let output = Tensor::from_vec(Device::Cpu, &shape, vec![0.0_f32]).unwrap();

    stream
        .begin_capture(StreamCaptureMode::ThreadLocal)
        .unwrap();
    let error = cuda::tune_cublaslt_matmul(
        &a,
        &b,
        &output,
        Epilogue::Default,
        None,
        1 << 20,
        CublasLtMatmulTuningOptions::DEFAULT,
        &stream,
    )
    .unwrap_err();
    let _ = stream.end_capture();

    assert!(matches!(
        error,
        Error::Unsupported { ref operation, .. }
            if operation == "cuBLASLt tuned matmul warmup during stream capture"
    ));
}
