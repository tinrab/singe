#![cfg(feature = "testing")]

use singe_cuda::{
    context::Context, cuda_module, graph::Graph, memory::DeviceMemory, module::LaunchConfig,
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
    let _lock = singe_cuda::testing::device_lock(0).unwrap();
    let ctx = match Context::create() {
        Ok(ctx) => ctx,
        Err(error) if singe_cuda::testing::is_stub_library(&error) => return,
        Err(error) => panic!("{error:?}"),
    };

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let alpha = 2.5f32;
    let length = input.len() as i32;

    let input_device = match DeviceMemory::from_slice(&input) {
        Ok(input_device) => input_device,
        Err(error) if singe_cuda::testing::is_stub_library(&error) => return,
        Err(error) => panic!("{error:?}"),
    };
    let mut output_device = match DeviceMemory::<f32>::zeroes(output.len()) {
        Ok(output_device) => output_device,
        Err(error) if singe_cuda::testing::is_stub_library(&error) => return,
        Err(error) => panic!("{error:?}"),
    };

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
    let _lock = singe_cuda::testing::device_lock(0).unwrap();
    let ctx = match Context::create() {
        Ok(ctx) => ctx,
        Err(error) if singe_cuda::testing::is_stub_library(&error) => return,
        Err(error) => panic!("{error:?}"),
    };

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let alpha = 2.5f32;
    let length = input.len() as i32;

    let input_device = match DeviceMemory::from_slice(&input) {
        Ok(input_device) => input_device,
        Err(error) if singe_cuda::testing::is_stub_library(&error) => return,
        Err(error) => panic!("{error:?}"),
    };
    let output_device = match DeviceMemory::<f32>::zeroes(output.len()) {
        Ok(output_device) => output_device,
        Err(error) if singe_cuda::testing::is_stub_library(&error) => return,
        Err(error) => panic!("{error:?}"),
    };

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
    let _lock = singe_cuda::testing::device_lock(0).unwrap();
    let ctx = match Context::create() {
        Ok(ctx) => ctx,
        Err(error) if singe_cuda::testing::is_stub_library(&error) => return,
        Err(error) => panic!("{error:?}"),
    };
    let stream = ctx.create_stream().unwrap();

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let alpha = 2.5f32;
    let length = input.len() as i32;

    let input_device = match DeviceMemory::from_slice(&input) {
        Ok(input_device) => input_device,
        Err(error) if singe_cuda::testing::is_stub_library(&error) => return,
        Err(error) => panic!("{error:?}"),
    };
    let mut output_device = match DeviceMemory::<f32>::zeroes(output.len()) {
        Ok(output_device) => output_device,
        Err(error) if singe_cuda::testing::is_stub_library(&error) => return,
        Err(error) => panic!("{error:?}"),
    };

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
    let _lock = singe_cuda::testing::device_lock(0).unwrap();
    let ctx = match Context::create() {
        Ok(ctx) => ctx,
        Err(error) if singe_cuda::testing::is_stub_library(&error) => return,
        Err(error) => panic!("{error:?}"),
    };
    let stream = ctx.create_stream().unwrap();

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let alpha = 2.5f32;
    let length = input.len() as i32;

    let input_device = match DeviceMemory::from_slice(&input) {
        Ok(input_device) => input_device,
        Err(error) if singe_cuda::testing::is_stub_library(&error) => return,
        Err(error) => panic!("{error:?}"),
    };
    let mut output_device = match DeviceMemory::<f32>::zeroes(output.len()) {
        Ok(output_device) => output_device,
        Err(error) if singe_cuda::testing::is_stub_library(&error) => return,
        Err(error) => panic!("{error:?}"),
    };

    let module = scale_kernel::Module::create(&ctx).unwrap();
    let config = LaunchConfig::for_1d_grid(input.len(), 128);
    let mut graph = Graph::create().unwrap();
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
fn adds_kernel_node_from_generated_module() {
    let _lock = singe_cuda::testing::device_lock(0).unwrap();
    let ctx = match Context::create() {
        Ok(ctx) => ctx,
        Err(error) if singe_cuda::testing::is_stub_library(&error) => return,
        Err(error) => panic!("{error:?}"),
    };
    let stream = ctx.create_stream().unwrap();

    let input = vec![1.0f32, 2.0, 3.5, -4.0, 8.25];
    let mut output = vec![0.0f32; input.len()];
    let alpha = 2.5f32;
    let length = input.len() as i32;

    let input_device = match DeviceMemory::from_slice(&input) {
        Ok(input_device) => input_device,
        Err(error) if singe_cuda::testing::is_stub_library(&error) => return,
        Err(error) => panic!("{error:?}"),
    };
    let mut output_device = match DeviceMemory::<f32>::zeroes(output.len()) {
        Ok(output_device) => output_device,
        Err(error) if singe_cuda::testing::is_stub_library(&error) => return,
        Err(error) => panic!("{error:?}"),
    };

    let module = scale_kernel::Module::create(&ctx).unwrap();
    let config = LaunchConfig::for_1d_grid(input.len(), 128);
    let mut graph = Graph::create().unwrap();
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
    let _lock = singe_cuda::testing::device_lock(0).unwrap();
    let ctx = match Context::create() {
        Ok(ctx) => ctx,
        Err(error) if singe_cuda::testing::is_stub_library(&error) => return,
        Err(error) => panic!("{error:?}"),
    };

    let mut values = vec![1.0f32, 2.5, -3.0, 0.25];
    let length = values.len() as i32;
    let mut values_device = match DeviceMemory::from_slice(&values) {
        Ok(values_device) => values_device,
        Err(error) if singe_cuda::testing::is_stub_library(&error) => return,
        Err(error) => panic!("{error:?}"),
    };

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
    let _lock = singe_cuda::testing::device_lock(0).unwrap();
    let ctx = match Context::create() {
        Ok(ctx) => ctx,
        Err(error) if singe_cuda::testing::is_stub_library(&error) => return,
        Err(error) => panic!("{error:?}"),
    };

    let mut values = vec![1.0f32, 2.5, -3.0, 0.25];
    let length = values.len() as i32;
    let mut values_device = match DeviceMemory::from_slice(&values) {
        Ok(values_device) => values_device,
        Err(error) if singe_cuda::testing::is_stub_library(&error) => return,
        Err(error) => panic!("{error:?}"),
    };

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
}
