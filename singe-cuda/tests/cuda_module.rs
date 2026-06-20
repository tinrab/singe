#![cfg(feature = "testing")]

use singe_cuda::{
    cuda_module, error::Error, memory::DeviceMemory, module::LaunchConfig,
    stream::StreamCaptureMode,
};

cuda_module! {
    pub mod scale_kernel {
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

cuda_module! {
    pub mod header_kernel {
        source: r#"
        #include "common.cuh"

        extern "C" __global__ void add_bias(float* values, int len) {
            int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            if (i < len) {
                values[i] += CUDA_MODULE_BIAS;
            }
        }
        "#,
        headers: {
            "common.cuh" => r#"
            #pragma once
            #define CUDA_MODULE_BIAS 3.0f
            "#,
        },
        compile: {
            nvcc_args: ["--std=c++20"],
            nvrtc_args: ["--std=c++20"],
        },
    }
}

cuda_module! {
    pub mod exported_kernel {
        source: r#"
        extern "C" __global__ void keep_me(float* values, int len) {
            int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            if (i < len) {
                values[i] *= 2.0f;
            }
        }

        extern "C" __global__ void skip_me(float* values, int len) {
            int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            if (i < len) {
                values[i] = -999.0f;
            }
        }
        "#,
        exports: {
            keep_me as double_values,
        },
    }
}

cuda_module! {
    pub mod typed_kernel {
        source: r#"
        extern "C" __global__ void typed_params(
            const float* input,
            float* output,
            size_t len,
            unsigned int flags
        ) {
            size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < len && flags != 0) {
                output[i] = input[i];
            }
        }
        "#,
    }
}

#[test]
fn launches_kernel_from_generated_module() {
    let (_lock, ctx) = singe_cuda::testing::bootstrap().unwrap();

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let alpha = 2.5f32;
    let length = input.len() as i32;

    let input_device = DeviceMemory::from_slice(&input).unwrap();
    let mut output_device = DeviceMemory::<f32>::zeroes(output.len()).unwrap();

    let module = scale_kernel::Module::create(&ctx).unwrap();
    let config = LaunchConfig::for_1d_grid(input.len(), 128);
    unsafe {
        module
            .scale_add_with_memory(&config, &input_device, &mut output_device, alpha, length)
            .unwrap();
    }

    output_device.copy_to_host(&mut output).unwrap();

    let expected = input
        .iter()
        .map(|value| value * alpha + 1.0)
        .collect::<Vec<_>>();
    assert_eq!(output, expected);
}

#[test]
fn launches_raw_pointer_kernel_from_generated_module() {
    let (_lock, ctx) = singe_cuda::testing::bootstrap().unwrap();

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let alpha = 2.5f32;
    let length = input.len() as i32;

    let input_device = DeviceMemory::from_slice(&input).unwrap();
    let output_device = DeviceMemory::<f32>::zeroes(output.len()).unwrap();

    let module = scale_kernel::Module::create(&ctx).unwrap();
    let config = LaunchConfig::for_1d_grid(input.len(), 128);
    unsafe {
        module
            .scale_add(
                &config,
                input_device.as_ptr(),
                output_device.as_mut_ptr(),
                alpha,
                length,
            )
            .unwrap();
    }

    output_device.copy_to_host(&mut output).unwrap();

    let expected = input
        .iter()
        .map(|value| value * alpha + 1.0)
        .collect::<Vec<_>>();
    assert_eq!(output, expected);
}

#[test]
fn launches_kernel_on_stream_from_generated_module() {
    let (_lock, ctx) = singe_cuda::testing::bootstrap().unwrap();
    let stream = ctx.create_stream().unwrap();

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let alpha = 2.5f32;
    let length = input.len() as i32;

    let input_device = DeviceMemory::from_slice(&input).unwrap();
    let mut output_device = DeviceMemory::<f32>::zeroes(output.len()).unwrap();

    let module = scale_kernel::Module::create(&ctx).unwrap();
    let config = LaunchConfig::for_1d_grid(input.len(), 128);
    unsafe {
        module
            .scale_add_with_memory_on(
                &config,
                &stream,
                &input_device,
                &mut output_device,
                alpha,
                length,
            )
            .unwrap();
    }
    stream.synchronize().unwrap();

    output_device.copy_to_host(&mut output).unwrap();

    let expected = input
        .iter()
        .map(|value| value * alpha + 1.0)
        .collect::<Vec<_>>();
    assert_eq!(output, expected);
}

#[test]
fn updates_kernel_node_params_from_generated_module() {
    let (_lock, ctx) = singe_cuda::testing::bootstrap().unwrap();
    let stream = ctx.create_stream().unwrap();

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let alpha = 2.5f32;
    let length = input.len() as i32;

    let input_device = DeviceMemory::from_slice(&input).unwrap();
    let mut output_device = DeviceMemory::<f32>::zeroes(output.len()).unwrap();

    let module = scale_kernel::Module::create(&ctx).unwrap();
    let config = LaunchConfig::for_1d_grid(input.len(), 128);
    let mut graph = ctx.create_graph().unwrap();
    let node = unsafe {
        module
            .scale_add_with_memory_node(
                &mut graph,
                &[],
                &config,
                &input_device,
                &mut output_device,
                0.0,
                length,
            )
            .unwrap()
    };
    let topology = graph.topology_summary().unwrap();
    assert_eq!(topology.nodes, 1);
    assert_eq!(topology.root_nodes, 1);
    assert_eq!(topology.edges, 0);
    assert_eq!(topology.kernel_nodes, 1);
    assert_eq!(topology.child_graph_nodes, 0);

    let mut executable = graph.instantiate().unwrap();
    drop(graph);
    unsafe {
        module
            .scale_add_with_memory_set_node_params(
                &mut executable,
                node,
                &config,
                &input_device,
                &mut output_device,
                alpha,
                length,
            )
            .unwrap();
    }
    executable.launch(&stream).unwrap();
    stream.synchronize().unwrap();

    output_device.copy_to_host(&mut output).unwrap();

    let expected = input
        .iter()
        .map(|value| value * alpha + 1.0)
        .collect::<Vec<_>>();
    assert_eq!(output, expected);
}

#[test]
fn rejects_node_from_different_graph() {
    let (_lock, ctx) = singe_cuda::testing::bootstrap().unwrap();

    let mut graph = ctx.create_graph().unwrap();
    let node = graph.add_empty_node(&[]).unwrap();
    let mut other_graph = ctx.create_graph().unwrap();

    assert!(matches!(
        other_graph.add_empty_node(&[node]),
        Err(Error::GraphNodeMismatch)
    ));

    let mut executable = graph.instantiate().unwrap();
    let other_node = other_graph.add_empty_node(&[]).unwrap();
    assert!(matches!(
        executable.disable_node(other_node),
        Err(Error::GraphNodeMismatch)
    ));
}

#[test]
fn child_graph_returns_borrowed_graph() {
    let (_lock, ctx) = singe_cuda::testing::bootstrap().unwrap();

    let mut child = ctx.create_graph().unwrap();
    child.add_empty_node(&[]).unwrap();

    let mut parent = ctx.create_graph().unwrap();
    let child_node = parent.add_child_graph_node(&[], &child).unwrap();
    let borrowed_child = child_node.child_graph().unwrap();

    assert_eq!(borrowed_child.topology_summary().unwrap().nodes, 1);
    assert_eq!(borrowed_child.context(), Some(ctx.as_ref()));
}

#[test]
fn graph_kernel_node_copies_borrowed_scalar_arguments() {
    let (_lock, ctx) = singe_cuda::testing::bootstrap().unwrap();
    let stream = ctx.create_stream().unwrap();

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let mut alpha = 2.5f32;
    let mut length = input.len() as i32;

    let input_device = DeviceMemory::from_slice(&input).unwrap();
    let mut output_device = DeviceMemory::<f32>::zeroes(output.len()).unwrap();

    let module = scale_kernel::Module::create(&ctx).unwrap();
    let function = module.raw().function("scale_add").unwrap();
    let config = LaunchConfig::for_1d_grid(input.len(), 128);
    let mut params = singe_cuda::module::KernelParameters::new();
    params
        .device_slice(&input_device)
        .device_slice_mut(&mut output_device)
        .arg(&alpha)
        .arg(&length);

    let mut graph = ctx.create_graph().unwrap();
    unsafe {
        function
            .add_to_graph(&mut graph, &[], &config, &mut params)
            .unwrap();
    }
    drop(params);
    alpha = -100.0;
    length = 0;

    let executable = graph.instantiate().unwrap();
    executable.launch(&stream).unwrap();
    stream.synchronize().unwrap();

    output_device.copy_to_host(&mut output).unwrap();

    let expected = input
        .iter()
        .map(|value| value * 2.5 + 1.0)
        .collect::<Vec<_>>();
    assert_eq!(output, expected);
    assert_eq!(alpha, -100.0);
    assert_eq!(length, 0);
}

#[test]
fn records_kernel_operation_during_capture() {
    let (_lock, ctx) = singe_cuda::testing::bootstrap().unwrap();
    let stream = ctx.create_stream().unwrap();

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let alpha = 2.5f32;
    let length = input.len() as i32;

    let input_device = DeviceMemory::from_slice(&input).unwrap();
    let mut output_device = DeviceMemory::<f32>::zeroes(output.len()).unwrap();

    let module = scale_kernel::Module::create(&ctx).unwrap();
    let config = LaunchConfig::for_1d_grid(input.len(), 128);

    let executable = stream
        .capture_executable(StreamCaptureMode::Relaxed, |scope| unsafe {
            module.scale_add_with_memory_record(
                scope,
                &config,
                &input_device,
                &mut output_device,
                alpha,
                length,
            )
        })
        .unwrap();

    executable.launch(&stream).unwrap();
    stream.synchronize().unwrap();

    output_device.copy_to_host(&mut output).unwrap();

    let expected = input
        .iter()
        .map(|value| value * alpha + 1.0)
        .collect::<Vec<_>>();
    assert_eq!(output, expected);
}

#[test]
fn adds_kernel_node_from_generated_module() {
    let (_lock, ctx) = singe_cuda::testing::bootstrap().unwrap();
    let stream = ctx.create_stream().unwrap();

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let alpha = 2.5f32;
    let length = input.len() as i32;

    let input_device = DeviceMemory::from_slice(&input).unwrap();
    let mut output_device = DeviceMemory::<f32>::zeroes(output.len()).unwrap();

    let module = scale_kernel::Module::create(&ctx).unwrap();
    let config = LaunchConfig::for_1d_grid(input.len(), 128);
    let mut graph = ctx.create_graph().unwrap();
    unsafe {
        module
            .scale_add_with_memory_node(
                &mut graph,
                &[],
                &config,
                &input_device,
                &mut output_device,
                alpha,
                length,
            )
            .unwrap();
    }

    let executable = graph.instantiate().unwrap();
    executable.launch(&stream).unwrap();
    stream.synchronize().unwrap();

    output_device.copy_to_host(&mut output).unwrap();
    let expected = input
        .iter()
        .map(|value| value * alpha + 1.0)
        .collect::<Vec<_>>();
    assert_eq!(output, expected);
}

#[test]
fn launches_kernel_with_headers() {
    let (_lock, ctx) = singe_cuda::testing::bootstrap().unwrap();

    let mut values = vec![1.0f32, 2.5, -3.0, 0.25];
    let length = values.len() as i32;
    let mut values_device = DeviceMemory::from_slice(&values).unwrap();

    let module = header_kernel::Module::create(&ctx).unwrap();
    let config = LaunchConfig::for_1d_grid(values.len(), 128);
    unsafe {
        module
            .add_bias_with_memory(&config, &mut values_device, length)
            .unwrap();
    }

    values_device.copy_to_host(&mut values).unwrap();
    assert_eq!(values, vec![4.0f32, 5.5, 0.0, 3.25]);
}

#[test]
fn exports_selected_kernel_with_renamed_method() {
    let (_lock, ctx) = singe_cuda::testing::bootstrap().unwrap();

    let mut values = vec![1.0f32, 2.5, -3.0, 0.25];
    let length = values.len() as i32;
    let mut values_device = DeviceMemory::from_slice(&values).unwrap();

    let module = exported_kernel::Module::create(&ctx).unwrap();
    let config = LaunchConfig::for_1d_grid(values.len(), 128);
    unsafe {
        module
            .double_values_with_memory(&config, &mut values_device, length)
            .unwrap();
    }

    values_device.copy_to_host(&mut values).unwrap();
    assert_eq!(values, vec![2.0f32, 5.0, -6.0, 0.5]);
}

#[test]
fn generated_module_uses_source_pointer_and_scalar_types() {
    let _raw_method: unsafe fn(
        &typed_kernel::Module,
        &LaunchConfig,
        *const f32,
        *mut f32,
        usize,
        u32,
    ) -> singe_cuda::error::Result<()> = typed_kernel::Module::typed_params;
    let _memory_method: unsafe fn(
        &typed_kernel::Module,
        &LaunchConfig,
        &DeviceMemory<f32>,
        &mut DeviceMemory<f32>,
        usize,
        u32,
    ) -> singe_cuda::error::Result<()> = typed_kernel::Module::typed_params_with_memory;
    let _memory_record_method: unsafe fn(
        &typed_kernel::Module,
        &singe_cuda::stream::StreamCaptureScope<'_>,
        &LaunchConfig,
        &DeviceMemory<f32>,
        &mut DeviceMemory<f32>,
        usize,
        u32,
    ) -> singe_cuda::error::Result<()> = typed_kernel::Module::typed_params_with_memory_record;
}
