use std::{error::Error, result::Result as StdResult};

mod support;

use singe_ptx::emulator::{Emulator, EmulatorOptions, LaunchConfig, ParsedModule, ShutdownMode};
use support::Observe;

type Result = StdResult<(), Box<dyn Error>>;

const KERNELS: &str = include_str!("fixtures/frontend/kernels.ptx");

#[test]
fn test_nvcc_block_reverse_uses_dynamic_shared_storage_and_partial_blocks() -> Result {
    let source = include_str!("fixtures/execution/block_reverse.ptx");
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let module = device
            .load(ParsedModule::parse(source)?)?
            .observe(asynchronous)?;
        let kernel = module.kernel("block_reverse")?;
        for count in [0usize, 1, 17, 31, 32, 33, 127, 128, 129, 257] {
            let input: Vec<i32> = (0..count)
                .map(|index| i32::try_from(index).unwrap() * 17 - 103)
                .collect();
            let expected: Vec<_> = input
                .chunks(128)
                .flat_map(|block| block.iter().rev().copied())
                .collect();
            let input = device.upload(&input)?.observe(asynchronous)?;
            let output = device.zeroed::<i32>(count)?.observe(asynchronous)?;
            kernel
                .launch(
                    LaunchConfig::create(
                        [u32::try_from(count.max(1).div_ceil(128)).unwrap(), 1, 1],
                        [128, 1, 1],
                        128 * size_of::<i32>(),
                    )?,
                    [
                        (&input).into(),
                        (&output).into(),
                        u32::try_from(count).unwrap().into(),
                    ],
                )?
                .observe(asynchronous)?;
            assert_eq!(
                device.download(&output)?.observe(asynchronous)?,
                expected,
                "count={count}, async={asynchronous}"
            );
        }
        device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
    }
    Ok(())
}

#[test]
fn test_dynamic_shared_aliases_span_module_entry_and_callee_declarations() -> Result {
    let source = ".version 9.3 .target sm_80 .address_size 64
        .extern .shared .align 4 .b8 module_shared[];
        .global .u32 output[8];
        .func (.param .u32 result) helper() {
            .extern .shared .align 64 .b8 helper_shared[];
            .reg .u32 r; ld.shared.u32 r,[helper_shared]; st.param.u32 [result],r; ret;
        }
        .entry k() {
            .extern .shared .align 16 .b8 entry_shared[];
            .shared .u32 separate; .param .u32 returned;
            .reg .u32 cta; .reg .u32 r; .reg .u64 address; .reg .u64 offset;
            mov.u32 cta,%ctaid.x;
            add.u32 r,cta,42; st.shared.u32 [module_shared],r;
            add.u32 r,cta,17; st.shared.u32 [separate],r;
            mul.wide.u32 offset,cta,16; mov.u64 address,output; add.u64 address,address,offset;
            ld.shared.u32 r,[entry_shared]; st.global.u32 [address],r;
            call (returned),helper,(); ld.param.u32 r,[returned]; st.global.u32 [address+4],r;
            ld.shared.u32 r,[separate]; st.global.u32 [address+8],r;
            mov.u64 offset,module_shared; and.b64 offset,offset,63;
            cvt.u32.u64 r,offset; st.global.u32 [address+12],r; ret;
        }";
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let module = device
            .load(ParsedModule::parse(source)?)?
            .observe(asynchronous)?;
        module
            .kernel("k")?
            .launch(LaunchConfig::create([2, 1, 1], [1; 3], 4)?, [])?
            .observe(asynchronous)?;
        let output = device
            .download(&module.global("output")?)?
            .observe(asynchronous)?;
        let actual: Vec<_> = output
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| u32::from_le_bytes(*v))
            .collect();
        assert_eq!(actual, [42, 42, 17, 0, 43, 43, 18, 0]);
        device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
    }
    Ok(())
}

#[test]
fn test_ptx_94_vector_add_with_both_waiting_styles() -> Result {
    let source = KERNELS.replace(".version 9.3", ".version 9.4");
    assert!(source.contains(".version 9.4"));
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let module = device
            .load(ParsedModule::parse(&source)?)?
            .observe(asynchronous)?;
        let a = device
            .upload(&[1i32, -2, i32::MAX])?
            .observe(asynchronous)?;
        let b = device.upload(&[2i32, 3, 1])?.observe(asynchronous)?;
        let out = device.zeroed::<i32>(3)?.observe(asynchronous)?;
        module
            .kernel("vector_add")?
            .launch(
                LaunchConfig::create([1, 1, 1], [32, 1, 1], 0)?,
                [(&a).into(), (&b).into(), (&out).into(), 3u32.into()],
            )?
            .observe(asynchronous)?;
        assert_eq!(
            device.download(&out)?.observe(asynchronous)?,
            [3, 1, i32::MIN]
        );
        device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
    }
    Ok(())
}

#[test]
fn test_nvcc_vector_add_and_saxpy() -> Result {
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let module = device
            .load(ParsedModule::parse(KERNELS)?)?
            .observe(asynchronous)?;
        for n in [1usize, 17, 31, 32, 33, 127, 128, 129, 257] {
            let a: Vec<i32> = (0..n).map(|i| i32::try_from(i).unwrap() - 80).collect();
            let b: Vec<i32> = (0..n).map(|i| 23 - i32::try_from(i).unwrap()).collect();
            let x = device.upload(&a)?.observe(asynchronous)?;
            let y = device.upload(&b)?.observe(asynchronous)?;
            let out = device.zeroed::<i32>(n)?.observe(asynchronous)?;
            let config = LaunchConfig::create(
                [u32::try_from(n.div_ceil(128)).unwrap(), 1, 1],
                [128, 1, 1],
                0,
            )?;
            let report = module
                .kernel("vector_add")?
                .launch(
                    config,
                    [
                        (&x).into(),
                        (&y).into(),
                        (&out).into(),
                        u32::try_from(n).unwrap().into(),
                    ],
                )?
                .observe(asynchronous)?;
            assert_eq!(report.completed_lanes, n.div_ceil(128) as u64 * 128);
            assert_eq!(device.download(&out)?.observe(asynchronous)?, vec![-57; n]);
            let a: Vec<f32> = a.iter().map(|v| *v as f32 / 8.).collect();
            let b: Vec<f32> = b.iter().map(|v| *v as f32 / 16.).collect();
            let x = device.upload(&a)?.observe(asynchronous)?;
            let y = device.upload(&b)?.observe(asynchronous)?;
            module
                .kernel("saxpy")?
                .launch(
                    config,
                    [
                        (&x).into(),
                        (&y).into(),
                        1.25f32.into(),
                        u32::try_from(n).unwrap().into(),
                    ],
                )?
                .observe(asynchronous)?;
            assert_eq!(
                device.download(&y)?.observe(asynchronous)?,
                a.iter()
                    .zip(&b)
                    .map(|(a, b)| a * 1.25 + b)
                    .collect::<Vec<_>>()
            );
        }
        device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
    }
    Ok(())
}

#[test]
fn test_nvcc_reduction_and_tiled_matrix() -> Result {
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let module = device
            .load(ParsedModule::parse(KERNELS)?)?
            .observe(asynchronous)?;
        for n in [1usize, 31, 32, 33, 127, 128, 129, 257] {
            let values: Vec<_> = (0..n).map(|i| (i % 9) as f32 - 4.).collect();
            let x = device.upload(&values)?.observe(asynchronous)?;
            let out = device
                .zeroed::<f32>(n.div_ceil(128))?
                .observe(asynchronous)?;
            module
                .kernel("block_sum")?
                .launch(
                    LaunchConfig::create(
                        [u32::try_from(n.div_ceil(128)).unwrap(), 1, 1],
                        [128, 1, 1],
                        0,
                    )?,
                    [(&x).into(), (&out).into(), u32::try_from(n).unwrap().into()],
                )?
                .observe(asynchronous)?;
            assert_eq!(
                device.download(&out)?.observe(asynchronous)?,
                values
                    .chunks(128)
                    .map(|v| v.iter().sum::<f32>())
                    .collect::<Vec<_>>()
            );
        }
        for n in [1usize, 15, 16, 17, 31, 32, 33] {
            let a: Vec<_> = (0..n * n).map(|i| (i % 7) as f32 - 3.).collect();
            let b: Vec<_> = (0..n * n).map(|i| (i % 5) as f32 - 2.).collect();
            let x = device.upload(&a)?.observe(asynchronous)?;
            let y = device.upload(&b)?.observe(asynchronous)?;
            let out = device.zeroed::<f32>(n * n)?.observe(asynchronous)?;
            module
                .kernel("tiled_matmul")?
                .launch(
                    LaunchConfig::create(
                        [
                            u32::try_from(n.div_ceil(16)).unwrap(),
                            u32::try_from(n.div_ceil(16)).unwrap(),
                            1,
                        ],
                        [16, 16, 1],
                        0,
                    )?,
                    [
                        (&x).into(),
                        (&y).into(),
                        (&out).into(),
                        u32::try_from(n).unwrap().into(),
                    ],
                )?
                .observe(asynchronous)?;
            let actual = device.download(&out)?.observe(asynchronous)?;
            for row in 0..n {
                for col in 0..n {
                    let expected: f32 = (0..n).map(|k| a[row * n + k] * b[k * n + col]).sum();
                    assert_eq!(
                        actual[row * n + col],
                        expected,
                        "width {n}, row {row}, col {col}"
                    );
                }
            }
        }
        device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
    }
    Ok(())
}

#[test]
fn test_transpose_across_tile_boundaries() -> Result {
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let module = device
            .load(ParsedModule::parse(include_str!(
                "fixtures/execution/transpose.ptx"
            ))?)?
            .observe(asynchronous)?;
        for n in [1usize, 15, 16, 17, 31, 32, 33] {
            let values: Vec<_> = (0..n * n).map(|i| i as f32).collect();
            let input = device.upload(&values)?.observe(asynchronous)?;
            let output = device.zeroed::<f32>(values.len())?.observe(asynchronous)?;
            module
                .kernel("transpose")?
                .launch(
                    LaunchConfig::create(
                        [
                            u32::try_from(n.div_ceil(16)).unwrap(),
                            u32::try_from(n.div_ceil(16)).unwrap(),
                            1,
                        ],
                        [16, 16, 1],
                        0,
                    )?,
                    [
                        (&input).into(),
                        (&output).into(),
                        u32::try_from(n).unwrap().into(),
                    ],
                )?
                .observe(asynchronous)?;
            let actual = device.download(&output)?.observe(asynchronous)?;
            for y in 0..n {
                for x in 0..n {
                    assert_eq!(actual[x * n + y], values[y * n + x]);
                }
            }
        }
        device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
    }
    Ok(())
}
