#![cfg(feature = "testing")]

use singe_cuda::{
    cuda_module,
    error::{Error, Result},
    event::EventRecordFlags,
    memory::DeviceMemory,
    module::LaunchConfig,
    nvrtc::Status as NvrtcStatus,
    stream::StreamCaptureMode,
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
async fn async_streams_and_graphs() -> Result<()> {
    let (_lock, ctx) = testing::bootstrap()?;
    let prepare_stream = ctx.create_stream()?;
    let compute_stream = ctx.create_stream()?;

    let input = vec![1.0_f32, 2.0, 3.5, -4.0, 8.25];
    let mut stream_output = vec![0.0_f32; input.len()];
    let mut graph_output = vec![0.0_f32; input.len()];

    let input_device = DeviceMemory::from_slice(&input)?;
    let mut stream_output_device = DeviceMemory::<f32>::zeroes(input.len())?;
    let mut graph_output_device = DeviceMemory::<f32>::zeroes(input.len())?;

    let module = match scale_add::Module::create(&ctx) {
        Ok(module) => module,
        Err(Error::Nvrtc {
            code: NvrtcStatus::BuiltinOperationFailure,
            ..
        }) => {
            eprintln!(
                "warning: skipping async CUDA module test because NVRTC builtins are unavailable"
            );
            return Ok(());
        }
        Err(error) => return Err(error),
    };
    let config = LaunchConfig::for_1d_grid(input.len(), 128);
    let len = input.len() as i32;

    // Ordinary stream work.
    // Enqueue a kernel and await the stream completion without blocking the async executor.
    unsafe {
        module.scale_add_with_memory_on(
            &config,
            &compute_stream,
            &input_device,
            &mut stream_output_device,
            2.0,
            len,
        )?;
    }
    compute_stream.synchronize_async().await?;
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
    unsafe {
        graph_output_device.set_value_async_unchecked(0, &prepare_stream)?;
    }
    let prepared = ctx.create_event()?;
    prepared.record(&prepare_stream, EventRecordFlags::DEFAULT)?;
    prepared.synchronize_async_on(&compute_stream).await?;

    // Capture reusable compute work once.
    // The captured graph keeps the same device pointers, so user-owned buffers must remain alive for replays.
    let executable =
        compute_stream.capture_executable(StreamCaptureMode::Relaxed, |scope| unsafe {
            module.scale_add_with_memory_record(
                scope,
                &config,
                &input_device,
                &mut graph_output_device,
                3.0,
                len,
            )
        })?;

    // Replay the graph and await completion.
    executable.launch_async(&compute_stream)?.await?;
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
