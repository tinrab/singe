#![cfg(all(feature = "testing", feature = "cuda"))]

use singe::{
    DataType, Device, Shape, Tensor,
    backend::cuda::{self, CudaEvent, CudaStream, EventTiming},
    error::Error,
};
use singe_cuda::testing::device_lock;

#[test]
fn test_events_order_streams_and_measure_time() {
    let _lock = device_lock(0).unwrap();
    let device = Device::Cuda(0);
    let producer = CudaStream::create(device).unwrap();
    let consumer = CudaStream::create(device).unwrap();
    let start = CudaEvent::create(device, EventTiming::Recorded).unwrap();
    let end = CudaEvent::create(device, EventTiming::Recorded).unwrap();
    let input = Tensor::from_vec(device, Shape::create([4]).unwrap(), vec![1_f32; 4]).unwrap();
    let output = {
        let _guard = cuda::set_current_stream(&producer).unwrap();
        assert!(cuda::current_stream(device).unwrap().is_some());
        start.record(&producer).unwrap();
        let output = input.add(&input).unwrap();
        end.record(&producer).unwrap();
        output
    };
    assert!(cuda::current_stream(device).unwrap().is_none());
    end.wait(&consumer).unwrap();
    end.synchronize().unwrap();
    assert!(end.query().unwrap());
    assert!(end.elapsed_since(&start).unwrap().as_secs_f64() >= 0.0);
    assert_eq!(output.to_vec::<f32>().unwrap(), [2.; 4]);
}

#[test]
fn test_memory_limit_rejects_allocations_beyond_it_and_peak_resets() {
    let _lock = device_lock(0).unwrap();
    let device = Device::Cuda(0);
    cuda::clear_cache(device).unwrap();
    let active = cuda::allocator_stats(device).unwrap().active_bytes();
    cuda::set_memory_limit(device, Some(active + (1 << 20))).unwrap();
    let small = Tensor::zeroed_on(device, DataType::F32, Shape::create([1024]).unwrap());
    let large = Tensor::zeroed_on(device, DataType::F32, Shape::create([1 << 20]).unwrap());
    cuda::set_memory_limit(device, None).unwrap();
    assert!(small.is_ok());
    assert!(matches!(large, Err(Error::MemoryLimitExceeded { .. })));

    drop(small);
    cuda::reset_peak_memory(device).unwrap();
    let stats = cuda::allocator_stats(device).unwrap();
    assert_eq!(stats.peak_bytes(), stats.active_bytes());
}
