#![cfg(feature = "testing")]

use std::sync::Arc;

use singe_cuda::{
    cuda_module,
    error::Result,
    event::{EventFlags, EventRecordFlags, EventWaitFlags},
    graph::GraphInstantiateFlags,
    memory::DeviceMemory,
    module::LaunchConfig,
    stream::{StreamCaptureMode, StreamOptions},
    testing,
};

cuda_module! {
    pub mod scale_add {
        source: r#"
            extern "C" __global__ void scale_add(
                const float* input,
                float* output,
                float alpha,
                int len
            ) {
                int i = static_cast<int>(
                    blockIdx.x * blockDim.x + threadIdx.x
                );
                if (i < len) {
                    output[i] = input[i] * alpha + 1.0f;
                }
            }
        "#,
    }
}

#[tokio::test]
async fn test_async_streams_and_graphs() -> Result<()> {
    let test_device = testing::bootstrap()?;
    let context = Arc::clone(test_device.context());
    let prepare_stream = context.create_stream(StreamOptions::default())?;
    let compute_stream = context.create_stream(StreamOptions::default())?;

    let input = vec![1.0_f32, 2.0, 3.5, -4.0, 8.25];
    let mut stream_output = vec![0.0_f32; input.len()];
    let mut graph_output = vec![0.0_f32; input.len()];

    let input_device = DeviceMemory::from_slice(&context, &input)?;
    let mut stream_output_device = DeviceMemory::<f32>::zeroed(&context, input.len())?;
    let mut graph_output_device = DeviceMemory::<f32>::zeroed(&context, input.len())?;

    let module = scale_add::Module::create(&context)?;
    let config = LaunchConfig::for_1d_grid(input.len(), 128);
    let len = i32::try_from(input.len()).unwrap();

    // Ordinary stream work.
    // Enqueue a kernel and await the stream completion without blocking the async executor.
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        module.scale_add_async(
            &input_device,
            &mut stream_output_device,
            2.0,
            len,
            &config,
            &compute_stream,
        )?;
    }
    compute_stream.completion()?.await?;
    stream_output_device.copy_to_host(&mut stream_output)?;
    assert_eq!(
        stream_output,
        input
            .iter()
            .map(|value| value * 2.0 + 1.0)
            .collect::<Vec<_>>(),
    );

    // Cross-stream ordering.
    // Prepare the graph output buffer on one stream, record an event, and make the compute stream wait asynchronously before launching graph work that writes the buffer.
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        graph_output_device.set_value_async(0, &prepare_stream)?;
    }
    let prepared = context.create_event(EventFlags::DEFAULT)?;
    prepared.record(&prepare_stream, EventRecordFlags::DEFAULT)?;
    compute_stream.wait_event(&prepared, EventWaitFlags::DEFAULT)?;
    compute_stream.completion()?.await?;

    // Capture reusable compute work once.
    // The captured graph keeps the same device pointers, so user-owned buffers must remain alive for replays.
    let executable =
        // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
        compute_stream.capture_executable(StreamCaptureMode::Relaxed, GraphInstantiateFlags::empty(), |scope| unsafe {
            module.scale_add_record(
                &input_device,
                &mut graph_output_device,
                3.0,
                len,
                &config,
                scope,
            )
        })?;

    // Replay the graph and await completion.
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe { executable.launch_async(&compute_stream)? };
    compute_stream.completion()?.await?;
    graph_output_device.copy_to_host(&mut graph_output)?;
    assert_eq!(
        graph_output,
        input
            .iter()
            .map(|value| value * 3.0 + 1.0)
            .collect::<Vec<_>>(),
    );

    Ok(())
}
