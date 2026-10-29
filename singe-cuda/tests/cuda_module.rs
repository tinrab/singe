#![cfg(feature = "testing")]

use singe_cuda::{
    cuda_module,
    error::{Error, Result},
    memory::DeviceMemory,
    module::{KernelParameters, LaunchConfig},
    nvrtc::Status as NvrtcStatus,
    stream::{StreamCaptureMode, StreamCaptureScope},
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
                requirements: singe_cuda::kernel::launch::LaunchRequirements {
                    rank: singe_cuda::kernel::launch::LaunchRank::One,
                    axis_coordinates: Some(singe_cuda::kernel::launch::CoordinateWidth::I32),
                    ..singe_cuda::kernel::launch::LaunchRequirements::DEFAULT
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
            "common.cuh" => r#"
            #pragma once
            #define CUDA_MODULE_BIAS 3.0f
            "#,
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
        source: r#"
        __global__ void cxx_scale(float* values, float factor, int len) {
            int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            if (i < len) {
                values[i] *= factor;
            }
        }
        "#,
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

fn create_module_or_skip<T>(result: Result<T>) -> Option<T> {
    match result {
        Ok(module) => Some(module),
        Err(Error::Nvrtc {
            code: NvrtcStatus::BuiltinOperationFailure,
            ..
        }) => {
            eprintln!("warning: skipping CUDA module test because NVRTC builtins are unavailable");
            None
        }
        Err(error) => panic!("CUDA module creation failed: {error:?}"),
    }
}

#[test]
fn test_launches_kernel_from_generated_module() {
    let (_lock, context) = bootstrap().unwrap();

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let alpha = 2.5f32;
    let length = input.len() as i32;

    let input_device = DeviceMemory::from_slice(&context, &input).unwrap();
    let mut output_device = DeviceMemory::<f32>::zeroed(&context, output.len()).unwrap();

    let Some(module) = create_module_or_skip(scale_kernel::Module::create(&context)) else {
        return;
    };
    let config = LaunchConfig::for_1d_grid(input.len(), 128);
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        module
            .scale_add(&config, &input_device, &mut output_device, alpha, length)
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
fn test_launches_cxx_linkage_kernel_from_generated_module() {
    let (_lock, context) = bootstrap().unwrap();

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let factor = 3.0f32;
    let length = input.len() as i32;

    let mut values_device = DeviceMemory::from_slice(&context, &input).unwrap();

    let Some(module) = create_module_or_skip(cxx_linkage_kernel::Module::create(&context)) else {
        return;
    };
    let config = LaunchConfig::for_1d_grid(input.len(), 128);
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        module
            .cxx_scale(&config, &mut values_device, factor, length)
            .unwrap();
    }

    values_device.copy_to_host(&mut output).unwrap();

    let expected = input.iter().map(|value| value * factor).collect::<Vec<_>>();
    assert_eq!(output, expected);
}

#[test]
fn test_launches_kernel_with_scalar_alias_parameters() {
    let (_lock, context) = bootstrap().unwrap();

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let length = input.len() as i32;
    let flags = 1_u32;

    let input_device = DeviceMemory::from_slice(&context, &input).unwrap();
    let mut output_device = DeviceMemory::<f32>::zeroed(&context, output.len()).unwrap();

    let Some(module) = create_module_or_skip(alias_scalar_kernel::Module::create(&context)) else {
        return;
    };
    let config = LaunchConfig::for_1d_grid(input.len(), 128);
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        module
            .alias_scalar(&config, &input_device, &mut output_device, length, flags)
            .unwrap();
    }

    output_device.copy_to_host(&mut output).unwrap();

    let expected = input.iter().map(|value| value + 2.0).collect::<Vec<_>>();
    assert_eq!(output, expected);
}

#[test]
fn test_launches_raw_pointer_kernel_from_generated_module() {
    let (_lock, context) = bootstrap().unwrap();

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let alpha = 2.5f32;
    let length = input.len() as i32;

    let input_device = DeviceMemory::from_slice(&context, &input).unwrap();
    let output_device = DeviceMemory::<f32>::zeroed(&context, output.len()).unwrap();

    let Some(module) = create_module_or_skip(scale_kernel::Module::create(&context)) else {
        return;
    };
    let config = LaunchConfig::for_1d_grid(input.len(), 128);
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        module
            .scale_add_raw(
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
fn test_launches_kernel_on_stream_from_generated_module() {
    let (_lock, context) = bootstrap().unwrap();
    let stream = context.create_stream().unwrap();

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let alpha = 2.5f32;
    let length = input.len() as i32;

    let input_device = DeviceMemory::from_slice(&context, &input).unwrap();
    let mut output_device = DeviceMemory::<f32>::zeroed(&context, output.len()).unwrap();

    let Some(module) = create_module_or_skip(scale_kernel::Module::create(&context)) else {
        return;
    };
    let config = LaunchConfig::for_1d_grid(input.len(), 128);
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        module
            .scale_add_on(
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
fn test_updates_kernel_node_params_from_generated_module() {
    let (_lock, context) = bootstrap().unwrap();
    let stream = context.create_stream().unwrap();

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let alpha = 2.5f32;
    let length = input.len() as i32;

    let input_device = DeviceMemory::from_slice(&context, &input).unwrap();
    let mut output_device = DeviceMemory::<f32>::zeroed(&context, output.len()).unwrap();

    let Some(module) = create_module_or_skip(scale_kernel::Module::create(&context)) else {
        return;
    };
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

    let mut executable = graph.instantiate().unwrap();
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
        executable.launch(&stream).unwrap();
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
    let (_lock, context) = bootstrap().unwrap();

    let mut graph = context.create_graph().unwrap();
    let node = graph.add_empty_node(&[]).unwrap();
    let mut other_graph = context.create_graph().unwrap();

    assert!(matches!(
        other_graph.add_empty_node(&[node]),
        Err(Error::GraphNodeMismatch)
    ));

    let mut executable = graph.instantiate().unwrap();
    let other_node = other_graph.add_empty_node(&[]).unwrap();
    assert!(matches!(
        executable.disable_node(&other_node),
        Err(Error::GraphNodeMismatch)
    ));
}

#[test]
fn test_child_graph_returns_borrowed_graph() {
    let (_lock, context) = bootstrap().unwrap();

    let mut child = context.create_graph().unwrap();
    child.add_empty_node(&[]).unwrap();

    let mut parent = context.create_graph().unwrap();
    let child_node = parent.add_child_graph_node(&[], &child).unwrap();
    let borrowed_child = child_node.child_graph().unwrap();

    assert_eq!(borrowed_child.topology_summary().unwrap().nodes, 1);
    assert_eq!(borrowed_child.context(), Some(&context));
}

#[test]
fn test_graph_kernel_node_copies_borrowed_scalar_arguments() {
    let (_lock, context) = bootstrap().unwrap();
    let stream = context.create_stream().unwrap();

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let mut alpha = 2.5f32;
    let mut length = input.len() as i32;

    let input_device = DeviceMemory::from_slice(&context, &input).unwrap();
    let mut output_device = DeviceMemory::<f32>::zeroed(&context, output.len()).unwrap();

    let Some(module) = create_module_or_skip(scale_kernel::Module::create(&context)) else {
        return;
    };
    let function = module.raw().function("scale_add").unwrap();
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

    let executable = graph.instantiate().unwrap();
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        executable.launch(&stream).unwrap();
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
    let (_lock, context) = bootstrap().unwrap();
    let stream = context.create_stream().unwrap();

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let alpha = 2.5f32;
    let length = input.len() as i32;

    let input_device = DeviceMemory::from_slice(&context, &input).unwrap();
    let mut output_device = DeviceMemory::<f32>::zeroed(&context, output.len()).unwrap();

    let Some(module) = create_module_or_skip(scale_kernel::Module::create(&context)) else {
        return;
    };
    let config = LaunchConfig::for_1d_grid(input.len(), 128);

    let executable = stream
        // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
        .capture_executable(StreamCaptureMode::Relaxed, |scope| unsafe {
            module.scale_add_record(
                scope,
                &config,
                &input_device,
                &mut output_device,
                alpha,
                length,
            )
        })
        .unwrap();

    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        executable.launch(&stream).unwrap();
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
    let (_lock, context) = bootstrap().unwrap();
    let stream = context.create_stream().unwrap();

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let alpha = 2.5f32;
    let length = input.len() as i32;

    let input_device = DeviceMemory::from_slice(&context, &input).unwrap();
    let mut output_device = DeviceMemory::<f32>::zeroed(&context, output.len()).unwrap();

    let Some(module) = create_module_or_skip(scale_kernel::Module::create(&context)) else {
        return;
    };
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

    let executable = graph.instantiate().unwrap();
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        executable.launch(&stream).unwrap();
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
    let (_lock, context) = bootstrap().unwrap();

    let mut values = vec![1.0f32, 2.5, -3.0, 0.25];
    let length = values.len() as i32;
    let mut values_device = DeviceMemory::from_slice(&context, &values).unwrap();

    let Some(module) = create_module_or_skip(header_kernel::Module::create(&context)) else {
        return;
    };
    let config = LaunchConfig::for_1d_grid(values.len(), 128);
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        module
            .add_bias(&config, &mut values_device, length)
            .unwrap();
    }

    values_device.copy_to_host(&mut values).unwrap();
    assert_eq!(values, vec![4.0f32, 5.5, 0.0, 3.25]);
}

#[test]
fn test_exports_selected_kernel_with_renamed_method() {
    let (_lock, context) = bootstrap().unwrap();

    let mut values = vec![1.0f32, 2.5, -3.0, 0.25];
    let length = values.len() as i32;
    let mut values_device = DeviceMemory::from_slice(&context, &values).unwrap();

    let Some(module) = create_module_or_skip(exported_kernel::Module::create(&context)) else {
        return;
    };
    let config = LaunchConfig::for_1d_grid(values.len(), 128);
    // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        module
            .double_values(&config, &mut values_device, length)
            .unwrap();
    }

    values_device.copy_to_host(&mut values).unwrap();
    assert_eq!(values, vec![2.0f32, 5.0, -6.0, 0.5]);
}

#[test]
fn test_generated_module_uses_source_pointer_and_scalar_types() {
    let _raw_method: unsafe fn(
        &typed_kernel::Module,
        &LaunchConfig,
        *const f32,
        *mut f32,
        usize,
        u32,
    ) -> Result<()> = typed_kernel::Module::typed_params_raw;
    let _memory_method: unsafe fn(
        &typed_kernel::Module,
        &LaunchConfig,
        &DeviceMemory<f32>,
        &mut DeviceMemory<f32>,
        usize,
        u32,
    ) -> Result<()> = typed_kernel::Module::typed_params;
    let _memory_record_method: unsafe fn(
        &typed_kernel::Module,
        &StreamCaptureScope<'_>,
        &LaunchConfig,
        &DeviceMemory<f32>,
        &mut DeviceMemory<f32>,
        usize,
        u32,
    ) -> Result<()> = typed_kernel::Module::typed_params_record;
}
