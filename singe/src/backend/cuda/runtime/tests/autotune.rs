use super::*;

use crate::backend::cuda::CudaStream;

#[test]
fn test_cuda_backend_runtime_autotune_cache_is_per_device_and_cleared() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let device = Device::Cuda(0);
    let Some(runtime) = skip_if_stub(runtime_for_device(device)) else {
        return;
    };
    cuda::clear_cache(device).unwrap();

    let bucket_shape = Shape::create([16, 32]).unwrap();
    let bucket_tensor = Tensor::from_vec(device, &bucket_shape, vec![0.0_f32; 16 * 32]).unwrap();
    let rhs_bucket_shape = Shape::create([32, 8]).unwrap();
    let rhs_bucket_tensor =
        Tensor::from_vec(device, &rhs_bucket_shape, vec![0.0_f32; 32 * 8]).unwrap();
    let bucket = AutotuneTensorBucket::from_tensor(&bucket_tensor);
    let matmul_buckets = AutotuneTensorBucket::from_tensors([&bucket_tensor, &rhs_bucket_tensor]);
    let manual_bucket = AutotuneTensorBucket::new(DataType::F32, vec![16, 32], vec![32, 1], 0);
    assert_eq!(bucket, manual_bucket);
    assert_eq!(matmul_buckets.len(), 2);
    assert_eq!(matmul_buckets[0], manual_bucket);
    assert_eq!(
        matmul_buckets[1],
        AutotuneTensorBucket::new(DataType::F32, vec![32, 8], vec![8, 1], 0)
    );
    let base_key = AutotuneCacheKey::from_tensors(
        Operation::MatMul,
        DataType::F32,
        [&bucket_tensor],
        CudaGraphProvider::CudnnBackend,
        "cudnn-test",
        Some(1 << 20),
    );
    let manual_base_key = AutotuneCacheKey::new(
        Operation::MatMul,
        DataType::F32,
        vec![bucket],
        CudaGraphProvider::CudnnBackend,
        "cudnn-test".to_string(),
        Some(1 << 20),
    );
    let cublas_key = AutotuneCacheKey::from_tensors(
        Operation::MatMul,
        DataType::F32,
        [&bucket_tensor],
        CudaGraphProvider::Cublas,
        "cudnn-test",
        Some(1 << 20),
    );
    let entry = AutotuneCacheEntry::new(4096);

    assert_eq!(base_key, manual_base_key);

    {
        let mut cache = runtime.autotune_cache().lock().unwrap();
        cache.insert(base_key.clone(), entry.clone());
    }

    {
        let cache = runtime.autotune_cache().lock().unwrap();
        assert_eq!(cache.len(), 1);
        assert_eq!(base_key.operation(), Operation::MatMul);
        assert_eq!(base_key.data_type(), DataType::F32);
        assert_eq!(base_key.provider(), CudaGraphProvider::CudnnBackend);
        assert_eq!(base_key.library_version(), "cudnn-test");
        assert_eq!(base_key.workspace_limit_bytes(), Some(1 << 20));
        assert_eq!(base_key.shape_layout_bucket().len(), 1);
        let cached_bucket = &base_key.shape_layout_bucket()[0];
        assert_eq!(cached_bucket.data_type(), DataType::F32);
        assert_eq!(cached_bucket.dimensions(), &[16, 32]);
        assert_eq!(cached_bucket.strides(), &[32, 1]);
        assert_eq!(cached_bucket.storage_offset(), 0);
        let cached = cache.get(&base_key).unwrap();
        assert_eq!(cached, &entry);
        assert_eq!(cached.workspace_bytes(), 4096);
        assert_eq!(cached.compute_type(), None);
        assert_eq!(cached.cublaslt_algorithm_fingerprint(), None);
        assert_eq!(cached.average_elapsed_ms(), None);
        assert_eq!(cache.get(&cublas_key), None);
    }
    let stats = cuda::autotune_cache_stats(device).unwrap();
    assert_eq!(stats.entries(), 1);
    assert_eq!(stats.cublaslt_matmul_tuning_entries(), 1);
    assert_eq!(stats.cublaslt_matmul_tuning_workspace_bytes(), 4096);
    assert_eq!(stats.cublaslt_matmul_tuning_elapsed_ms(), 0.0);
    assert_eq!(stats.cublaslt_matmul_tuning_average_elapsed_ms(), Some(0.0));

    cuda::clear_cache(device).unwrap();
    assert!(runtime.autotune_cache().lock().unwrap().is_empty());
    assert_eq!(
        cuda::autotune_cache_stats(device).unwrap(),
        AutotuneCacheStats::default()
    );
}

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
