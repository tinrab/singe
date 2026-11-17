#![cfg(feature = "testing")]

use std::sync::Arc;

use singe_cuda::{
    cuda_module,
    error::{Error, Result},
    graph::GraphInstantiateFlags,
    kernel::launch::{CoordinateWidth, LaunchRank},
    memory::DeviceMemory,
    module::{KernelParameters, LaunchConfig},
    stream::{Stream, StreamCaptureMode, StreamCaptureScope, StreamOptions},
    testing::bootstrap,
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
        exports: {
            scale_add {
                parameters: { input: read, output: write },
                requirements: LaunchRequirements {
                    rank: super::LaunchRank::One,
                    axis_coordinates: Some(super::CoordinateWidth::I32),
                    ..LaunchRequirements::DEFAULT
                },
            },
        },
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
            "common.cuh" => r"
            #pragma once
            #define CUDA_MODULE_BIAS 3.0f
            ",
        },
        compile: {
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

cuda_module! {
    pub mod cxx_linkage_kernel {
        source: r"
        __global__ void cxx_scale(float* values, float factor, int len) {
            int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            if (i < len) {
                values[i] *= factor;
            }
        }
        ",
    }
}

cuda_module! {
    pub mod alias_scalar_kernel {
        source: r#"
        using index_t = int;
        typedef unsigned int flags_t;

        extern "C" __global__ void alias_scalar(
            const float* input,
            float* output,
            index_t len,
            flags_t flags
        ) {
            index_t i = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
            if (i < len && flags != 0) {
                output[i] = input[i] + 2.0f;
            }
        }
        "#,
    }
}

#[test]
fn test_launches_cxx_linkage_kernel_from_generated_module() {
    let test_device = bootstrap().unwrap();
    let context = Arc::clone(test_device.context());

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let factor = 3.0f32;
    let length = i32::try_from(input.len()).unwrap();

    let mut values_device = DeviceMemory::from_slice(&context, &input).unwrap();

    let module = cxx_linkage_kernel::Module::create(&context).unwrap();
    let config = LaunchConfig::for_1d_grid(input.len(), 128);
    let stream = context.create_stream(StreamOptions::default()).unwrap();
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        module
            .cxx_scale_async(&mut values_device, factor, length, &config, &stream)
            .unwrap();
    }
    stream.synchronize().unwrap();

    values_device.copy_to_host(&mut output).unwrap();

    let expected = input.iter().map(|value| value * factor).collect::<Vec<_>>();
    assert_eq!(output, expected);
}

#[test]
fn test_launches_kernel_with_scalar_alias_parameters() {
    let test_device = bootstrap().unwrap();
    let context = Arc::clone(test_device.context());

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let length = i32::try_from(input.len()).unwrap();
    let flags = 1_u32;

    let input_device = DeviceMemory::from_slice(&context, &input).unwrap();
    let mut output_device = DeviceMemory::<f32>::zeroed(&context, output.len()).unwrap();

    let module = alias_scalar_kernel::Module::create(&context).unwrap();
    let config = LaunchConfig::for_1d_grid(input.len(), 128);
    let stream = context.create_stream(StreamOptions::default()).unwrap();
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        module
            .alias_scalar_async(
                &input_device,
                &mut output_device,
                length,
                flags,
                &config,
                &stream,
            )
            .unwrap();
    }
    stream.synchronize().unwrap();

    output_device.copy_to_host(&mut output).unwrap();

    let expected = input.iter().map(|value| value + 2.0).collect::<Vec<_>>();
    assert_eq!(output, expected);
}

#[test]
fn test_launches_kernel_on_stream_from_generated_module() {
    let test_device = bootstrap().unwrap();
    let context = Arc::clone(test_device.context());
    let stream = context.create_stream(StreamOptions::default()).unwrap();

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let alpha = 2.5f32;
    let length = i32::try_from(input.len()).unwrap();

    let input_device = DeviceMemory::from_slice(&context, &input).unwrap();
    let mut output_device = DeviceMemory::<f32>::zeroed(&context, output.len()).unwrap();

    let module = scale_kernel::Module::create(&context).unwrap();
    let config = LaunchConfig::for_1d_grid(input.len(), 128);
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        module
            .scale_add_async(
                &input_device,
                &mut output_device,
                alpha,
                length,
                &config,
                &stream,
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
fn test_updates_kernel_node_params_from_generated_module() {
    let test_device = bootstrap().unwrap();
    let context = Arc::clone(test_device.context());
    let stream = context.create_stream(StreamOptions::default()).unwrap();

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let alpha = 2.5f32;
    let length = i32::try_from(input.len()).unwrap();

    let input_device = DeviceMemory::from_slice(&context, &input).unwrap();
    let mut output_device = DeviceMemory::<f32>::zeroed(&context, output.len()).unwrap();

    let module = scale_kernel::Module::create(&context).unwrap();
    let config = LaunchConfig::for_1d_grid(input.len(), 128);
    let mut graph = context.create_graph().unwrap();
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    let node = unsafe {
        module
            .scale_add_node(
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

    let mut executable = graph.instantiate(GraphInstantiateFlags::empty()).unwrap();
    drop(graph);
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        module
            .scale_add_set_node_params(
                &mut executable,
                &node,
                config.geometry(),
                &input_device,
                &mut output_device,
                alpha,
                length,
            )
            .unwrap();
    }
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        executable.launch_async(&stream).unwrap();
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
fn test_rejects_node_from_different_graph() {
    let test_device = bootstrap().unwrap();
    let context = Arc::clone(test_device.context());

    let mut graph = context.create_graph().unwrap();
    let node = graph.add_empty_node(&[]).unwrap();
    let mut other_graph = context.create_graph().unwrap();

    assert!(matches!(
        other_graph.add_empty_node(&[node]),
        Err(Error::GraphNodeMismatch)
    ));

    let mut executable = graph.instantiate(GraphInstantiateFlags::empty()).unwrap();
    let other_node = other_graph.add_empty_node(&[]).unwrap();
    assert!(matches!(
        executable.disable_node(&other_node),
        Err(Error::GraphNodeMismatch)
    ));
}

#[test]
fn test_child_graph_returns_borrowed_graph() {
    let test_device = bootstrap().unwrap();
    let context = Arc::clone(test_device.context());

    let mut child = context.create_graph().unwrap();
    child.add_empty_node(&[]).unwrap();

    let mut parent = context.create_graph().unwrap();
    let child_node = parent.add_child_graph_node(&[], &child).unwrap();
    let borrowed_child = parent.child_graph(&child_node).unwrap();

    assert_eq!(borrowed_child.topology_summary().unwrap().nodes, 1);
    assert_eq!(borrowed_child.context(), &context);
}

#[test]
fn test_graph_kernel_node_copies_borrowed_scalar_arguments() {
    let test_device = bootstrap().unwrap();
    let context = Arc::clone(test_device.context());
    let stream = context.create_stream(StreamOptions::default()).unwrap();

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let mut alpha = 2.5f32;
    let mut length = i32::try_from(input.len()).unwrap();

    let input_device = DeviceMemory::from_slice(&context, &input).unwrap();
    let mut output_device = DeviceMemory::<f32>::zeroed(&context, output.len()).unwrap();

    let module = scale_kernel::Module::create(&context).unwrap();
    let function = module.module().function("scale_add").unwrap();
    let config = LaunchConfig::for_1d_grid(input.len(), 128);
    let mut params = KernelParameters::new();
    params
        .device_input(&input_device)
        .device_slice_mut(&mut output_device)
        .arg(&alpha)
        .arg(&length);

    let mut graph = context.create_graph().unwrap();
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        function
            .add_to_graph(&mut graph, &[], &config, &mut params)
            .unwrap();
    }
    drop(params);
    alpha = -100.0;
    length = 0;

    let executable = graph.instantiate(GraphInstantiateFlags::empty()).unwrap();
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        executable.launch_async(&stream).unwrap();
    }
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
fn test_records_kernel_operation_during_capture() {
    let test_device = bootstrap().unwrap();
    let context = Arc::clone(test_device.context());
    let stream = context.create_stream(StreamOptions::default()).unwrap();

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let alpha = 2.5f32;
    let length = i32::try_from(input.len()).unwrap();

    let input_device = DeviceMemory::from_slice(&context, &input).unwrap();
    let mut output_device = DeviceMemory::<f32>::zeroed(&context, output.len()).unwrap();

    let module = scale_kernel::Module::create(&context).unwrap();
    let config = LaunchConfig::for_1d_grid(input.len(), 128);

    let executable = stream
        .capture_executable(
            StreamCaptureMode::Relaxed,
            GraphInstantiateFlags::empty(),
            // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
            |scope| unsafe {
                module.scale_add_record(
                    &input_device,
                    &mut output_device,
                    alpha,
                    length,
                    &config,
                    scope,
                )
            },
        )
        .unwrap();

    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        executable.launch_async(&stream).unwrap();
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
fn test_adds_kernel_node_from_generated_module() {
    let test_device = bootstrap().unwrap();
    let context = Arc::clone(test_device.context());
    let stream = context.create_stream(StreamOptions::default()).unwrap();

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let alpha = 2.5f32;
    let length = i32::try_from(input.len()).unwrap();

    let input_device = DeviceMemory::from_slice(&context, &input).unwrap();
    let mut output_device = DeviceMemory::<f32>::zeroed(&context, output.len()).unwrap();

    let module = scale_kernel::Module::create(&context).unwrap();
    let config = LaunchConfig::for_1d_grid(input.len(), 128);
    let mut graph = context.create_graph().unwrap();
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        module
            .scale_add_node(
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

    let executable = graph.instantiate(GraphInstantiateFlags::empty()).unwrap();
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        executable.launch_async(&stream).unwrap();
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
fn test_launches_kernel_with_headers() {
    let test_device = bootstrap().unwrap();
    let context = Arc::clone(test_device.context());

    let mut values = vec![1.0f32, 2.5, -3.0, 0.25];
    let length = i32::try_from(values.len()).unwrap();
    let mut values_device = DeviceMemory::from_slice(&context, &values).unwrap();

    let module = header_kernel::Module::create(&context).unwrap();
    let config = LaunchConfig::for_1d_grid(values.len(), 128);
    let stream = context.create_stream(StreamOptions::default()).unwrap();
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        module
            .add_bias_async(&mut values_device, length, &config, &stream)
            .unwrap();
    }
    stream.synchronize().unwrap();

    values_device.copy_to_host(&mut values).unwrap();
    assert_eq!(values, vec![4.0f32, 5.5, 0.0, 3.25]);
}

#[test]
fn test_exports_selected_kernel_with_renamed_method() {
    let test_device = bootstrap().unwrap();
    let context = Arc::clone(test_device.context());

    let mut values = vec![1.0f32, 2.5, -3.0, 0.25];
    let length = i32::try_from(values.len()).unwrap();
    let mut values_device = DeviceMemory::from_slice(&context, &values).unwrap();

    let module = exported_kernel::Module::create(&context).unwrap();
    let config = LaunchConfig::for_1d_grid(values.len(), 128);
    let stream = context.create_stream(StreamOptions::default()).unwrap();
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        module
            .double_values_async(&mut values_device, length, &config, &stream)
            .unwrap();
    }
    stream.synchronize().unwrap();

    values_device.copy_to_host(&mut values).unwrap();
    assert_eq!(values, vec![2.0f32, 5.0, -6.0, 0.5]);
}

type TypedParamsAsync = unsafe fn(
    &typed_kernel::Module,
    &DeviceMemory<f32>,
    &mut DeviceMemory<f32>,
    usize,
    u32,
    &LaunchConfig,
    &Stream,
) -> Result<()>;

type TypedParamsRecord = unsafe fn(
    &typed_kernel::Module,
    &DeviceMemory<f32>,
    &mut DeviceMemory<f32>,
    usize,
    u32,
    &LaunchConfig,
    &StreamCaptureScope<'_>,
) -> Result<()>;

#[test]
fn test_generated_module_uses_source_pointer_and_scalar_types() {
    let _: TypedParamsAsync = typed_kernel::Module::typed_params_async;
    let _: TypedParamsRecord = typed_kernel::Module::typed_params_record;
}
