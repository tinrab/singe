use std::{error::Error, result::Result as StdResult};

mod support;

use singe_ptx::emulator::{
    Bf16, Emulator, EmulatorOptions, F16, LaunchConfig, ParsedModule, ShutdownMode,
};
use support::Observe;

type Result = StdResult<(), Box<dyn Error>>;

const PRELUDE: &str = ".version 9.3 .target sm_80 .address_size 64";

#[test]
fn test_half_boundary_values_widen_and_roundtrip() -> Result {
    for format in ["f16", "bf16"] {
        let source = format!(
            r"{PRELUDE}
.entry k(.param .u64 input, .param .u64 output, .param .u64 wide) {{
 .reg .u32 i; .reg .u32 thread_id; .reg .b64 a; .reg .b64 b; .reg .b64 w; .reg .b64 offset;
 .reg .b16 h; .reg .f32 f;
 mov.u32 thread_id,%tid.x; mov.u32 i,%ctaid.x; mul.lo.u32 i,i,128; add.u32 i,i,thread_id;
 ld.param.u64 a,[input]; ld.param.u64 b,[output]; ld.param.u64 w,[wide];
 mul.wide.u32 offset,i,2; add.u64 a,a,offset; add.u64 b,b,offset;
 ld.global.b16 h,[a]; cvt.f32.{format} f,h;
 mul.wide.u32 offset,i,4; add.u64 w,w,offset; st.global.f32 [w],f;
 cvt.rn.{format}.f32 h,f; st.global.b16 [b],h; ret;
}}"
        );
        let device = Emulator::create(EmulatorOptions::default())?;
        let module = device.load(ParsedModule::parse(&source)?)?.wait()?;
        let magnitudes: &[u16] = if format == "f16" {
            &[
                0, 1, 0x03ff, 0x0400, 0x3bff, 0x3c00, 0x3c01, 0x7bff, 0x7c00, 0x7c01, 0x7e00,
                0x7fff,
            ]
        } else {
            &[
                0, 1, 0x007f, 0x0080, 0x3f7f, 0x3f80, 0x3f81, 0x7f7f, 0x7f80, 0x7f81, 0x7fc0,
                0x7fff,
            ]
        };
        let bits: Vec<u16> = magnitudes
            .iter()
            .flat_map(|&value| [value, value | 0x8000])
            .collect();
        let input = device.upload(&bits)?.wait()?;
        let output = device.zeroed::<u16>(bits.len())?.wait()?;
        let wide = device.zeroed::<f32>(bits.len())?.wait()?;
        module
            .kernel("k")?
            .launch(
                LaunchConfig::create([1; 3], [u32::try_from(bits.len()).unwrap(), 1, 1], 0)?,
                [(&input).into(), (&output).into(), (&wide).into()],
            )?
            .wait()?;
        let narrow = device.download(&output)?.wait()?;
        let wide = device.download(&wide)?.wait()?;
        for (i, (&narrow, &wide)) in narrow.iter().zip(&wide).enumerate() {
            let input = bits[i];
            let expected = if format == "f16" {
                F16::from_bits(input).to_f32()
            } else {
                Bf16::from_bits(input).to_f32()
            };
            if expected.is_nan() {
                assert!(wide.is_nan());
                assert!(if format == "f16" {
                    F16::from_bits(narrow).is_nan()
                } else {
                    Bf16::from_bits(narrow).is_nan()
                });
            } else {
                assert_eq!(wide.to_bits(), expected.to_bits(), "{format} {input:04x}");
                assert_eq!(narrow, input, "{format} {input:04x}");
            }
        }
        device.shutdown(ShutdownMode::Drain).wait()?;
    }
    Ok(())
}

#[test]
fn test_typed_half_transfers_scalar_arguments_and_packed_kernels() -> Result {
    for asynchronous in [false, true] {
        for format in ["f16", "bf16"] {
            let source = format!(
                r"{PRELUDE}
.entry scalar(.param .u64 input, .param .u64 output, .param .b16 alpha, .param .u32 n) {{
 .reg .u32 i; .reg .u32 thread_id; .reg .u32 size; .reg .pred p;
 .reg .b64 a; .reg .b64 b; .reg .b64 offset;
 .reg .b16 h; .reg .b16 scale; .reg .b16 zero;
 mov.u32 thread_id,%tid.x; mov.u32 i,%ctaid.x; mul.lo.u32 i,i,128; add.u32 i,i,thread_id;
 ld.param.u32 size,[n]; setp.ge.u32 p,i,size; @p ret;
 ld.param.u64 a,[input]; ld.param.u64 b,[output]; ld.param.b16 scale,[alpha];
 mul.wide.u32 offset,i,2; add.u64 a,a,offset; add.u64 b,b,offset;
 ld.global.b16 h,[a]; mov.b16 zero,0;
 fma.rn.{format} h,h,scale,zero; st.global.b16 [b],h; ret;
}}
.entry packed(.param .u64 input, .param .u64 output, .param .u32 n) {{
 .reg .u32 i; .reg .u32 thread_id; .reg .u32 size; .reg .pred p;
 .reg .b64 a; .reg .b64 b; .reg .b64 offset;
 .reg .b32 h; .reg .b32 scale; .reg .b32 zero;
 mov.u32 thread_id,%tid.x; mov.u32 i,%ctaid.x; mul.lo.u32 i,i,128; add.u32 i,i,thread_id;
 ld.param.u32 size,[n]; setp.ge.u32 p,i,size; @p ret;
 ld.param.u64 a,[input]; ld.param.u64 b,[output];
 mul.wide.u32 offset,i,4; add.u64 a,a,offset; add.u64 b,b,offset;
 ld.global.b32 h,[a]; cvt.rn.{format}x2.f32 scale,2.0,0.5; mov.b32 zero,0;
 fma.rn.{format}x2 h,h,scale,zero; st.global.b32 [b],h; ret;
}}"
            );
            let device = Emulator::create(EmulatorOptions::default())?;
            let module = device
                .load(ParsedModule::parse(&source)?)?
                .observe(asynchronous)?;
            for n in [1usize, 17, 31, 32, 33, 127, 128, 129, 257] {
                let values: Vec<f32> = (0..n).map(|i| (i % 17) as f32 / 8.).collect();
                macro_rules! scalar {
                    ($ty:ty) => {{
                        let values: Vec<$ty> = values.iter().map(|&v| <$ty>::from_f32(v)).collect();
                        let input = device.upload(&values)?.observe(asynchronous)?;
                        let output = device.zeroed::<$ty>(n)?.observe(asynchronous)?;
                        module
                            .kernel("scalar")?
                            .launch(
                                LaunchConfig::create(
                                    [u32::try_from(n.div_ceil(128)).unwrap(), 1, 1],
                                    [128, 1, 1],
                                    0,
                                )?,
                                [
                                    (&input).into(),
                                    (&output).into(),
                                    <$ty>::from_f32(2.).into(),
                                    u32::try_from(n).unwrap().into(),
                                ],
                            )?
                            .observe(asynchronous)?;
                        let actual = device.download(&output)?.observe(asynchronous)?;
                        assert_eq!(
                            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            values
                                .iter()
                                .map(|v| <$ty>::from_f32(v.to_f32() * 2.).to_bits())
                                .collect::<Vec<_>>()
                        );
                        let view = output.view(0..1)?;
                        device
                            .write(&view, &[<$ty>::from_bits(0x8000)])?
                            .observe(asynchronous)?;
                        assert_eq!(
                            device.download(&view)?.observe(asynchronous)?[0].to_bits(),
                            0x8000
                        );
                    }};
                }
                if format == "f16" {
                    scalar!(F16);
                } else {
                    scalar!(Bf16);
                }
                let encode = |v| {
                    if format == "f16" {
                        F16::from_f32(v).to_bits()
                    } else {
                        Bf16::from_f32(v).to_bits()
                    }
                };
                let pairs: Vec<u32> = values
                    .chunks(2)
                    .map(|v| {
                        u32::from(encode(v[0]))
                            | (u32::from(encode(v.get(1).copied().unwrap_or(0.))) << 16)
                    })
                    .collect();
                let expected: Vec<u32> = values
                    .chunks(2)
                    .map(|v| {
                        u32::from(encode(v[0] * 0.5))
                            | (u32::from(encode(v.get(1).copied().unwrap_or(0.) * 2.)) << 16)
                    })
                    .collect();
                let input = device.upload(&pairs)?.observe(asynchronous)?;
                let output = device.zeroed::<u32>(pairs.len())?.observe(asynchronous)?;
                module
                    .kernel("packed")?
                    .launch(
                        LaunchConfig::create(
                            [u32::try_from(pairs.len().div_ceil(128)).unwrap(), 1, 1],
                            [128, 1, 1],
                            0,
                        )?,
                        [
                            (&input).into(),
                            (&output).into(),
                            u32::try_from(pairs.len()).unwrap().into(),
                        ],
                    )?
                    .observe(asynchronous)?;
                assert_eq!(device.download(&output)?.observe(asynchronous)?, expected);
            }
            device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
        }
    }
    Ok(())
}

#[test]
fn test_carry_predication_and_call_clobbering() -> Result {
    for (body, error, expected) in [
        ("mov.pred p,0; @p addc.u32 r,0,0; mov.u32 r,7;", false, 7u32),
        (
            "mov.pred p,0; add.cc.u32 r,0xffffffff,1; @p add.cc.u32 r,0,0; addc.u32 r,0,0;",
            false,
            1,
        ),
        ("addc.u32 r,0,0;", true, 0),
        (
            "add.cc.u32 r,0xffffffff,1; call helper,(); addc.u32 r,0,0;",
            true,
            0,
        ),
        ("call helper,(); addc.u32 r,0,0;", true, 0),
    ] {
        let source = format!(
            "{PRELUDE} .global .u32 output; .func helper() {{.reg .u32 r;add.cc.u32 r,0xffffffff,1;ret;}} .entry k() {{.reg .u32 r;.reg .pred p;{body} st.global.u32 [output],r;ret;}}"
        );
        let device = Emulator::create(EmulatorOptions::default())?;
        let module = device.load(ParsedModule::parse(&source)?)?.wait()?;
        let result = module
            .kernel("k")?
            .launch(LaunchConfig::create([1, 1, 1], [1, 1, 1], 0)?, [])?
            .wait();
        if error {
            assert!(result.unwrap_err().to_string().contains("carry flag"));
        } else {
            result?;
            assert_eq!(
                device.download(&module.global("output")?)?.wait()?,
                expected.to_le_bytes()
            );
        }
        let shutdown = device.shutdown(ShutdownMode::Drain).wait();
        if error {
            assert!(shutdown.is_err());
        } else {
            shutdown?;
        }
    }
    Ok(())
}

#[test]
fn test_fp64_vector_add_saxpy_and_reduction_and_multiword_arithmetic() -> Result {
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let module = device
            .load(ParsedModule::parse(include_str!(
                "fixtures/execution/scalar.ptx"
            ))?)?
            .observe(asynchronous)?;
        for n in [1usize, 17, 31, 32, 33, 127, 128, 129, 257] {
            let config = LaunchConfig::create(
                [u32::try_from(n.div_ceil(128)).unwrap(), 1, 1],
                [128, 1, 1],
                0,
            )?;
            let a: Vec<f64> = (0..n).map(|i| i as f64 / 8. - 20.).collect();
            let b: Vec<f64> = (0..n).map(|i| i as f64 / 16. + 3.).collect();
            let x = device.upload(&a)?.observe(asynchronous)?;
            let y = device.upload(&b)?.observe(asynchronous)?;
            let output = device.zeroed::<f64>(n)?.observe(asynchronous)?;
            module
                .kernel("vector_add_f64")?
                .launch(
                    config,
                    [
                        (&x).into(),
                        (&y).into(),
                        (&output).into(),
                        u32::try_from(n).unwrap().into(),
                    ],
                )?
                .observe(asynchronous)?;
            assert_eq!(
                device.download(&output)?.observe(asynchronous)?,
                a.iter().zip(&b).map(|(a, b)| a + b).collect::<Vec<_>>()
            );
            module
                .kernel("saxpy_f64")?
                .launch(
                    config,
                    [
                        (&x).into(),
                        (&y).into(),
                        1.25f64.into(),
                        u32::try_from(n).unwrap().into(),
                    ],
                )?
                .observe(asynchronous)?;
            assert_eq!(
                device.download(&y)?.observe(asynchronous)?,
                a.iter()
                    .zip(&b)
                    .map(|(a, b)| a.mul_add(1.25, *b))
                    .collect::<Vec<_>>()
            );
            let output = device
                .zeroed::<f64>(n.div_ceil(128))?
                .observe(asynchronous)?;
            module
                .kernel("block_sum_f64")?
                .launch(
                    config,
                    [
                        (&x).into(),
                        (&output).into(),
                        u32::try_from(n).unwrap().into(),
                    ],
                )?
                .observe(asynchronous)?;
            assert_eq!(
                device.download(&output)?.observe(asynchronous)?,
                a.chunks(128)
                    .map(|v| v.iter().sum::<f64>())
                    .collect::<Vec<_>>()
            );
            let a: Vec<u64> = (0..n)
                .map(|i| {
                    (i as u64)
                        .wrapping_mul(0x0dee_dfac_e123_4567)
                        .wrapping_add(u64::MAX)
                })
                .collect();
            let b: Vec<u64> = (0..n)
                .map(|i| {
                    (i as u64)
                        .wrapping_mul(0x00ba_da55_cafe_d00d)
                        .wrapping_add(u64::MAX)
                })
                .collect();
            let x = device.upload(&a)?.observe(asynchronous)?;
            let y = device.upload(&b)?.observe(asynchronous)?;
            let output = device.zeroed::<u32>(n * 10)?.observe(asynchronous)?;
            module
                .kernel("multiword")?
                .launch(
                    config,
                    [
                        (&x).into(),
                        (&y).into(),
                        (&output).into(),
                        u32::try_from(n).unwrap().into(),
                    ],
                )?
                .observe(asynchronous)?;
            let word = |value: u128| u32::try_from(value & u128::from(u32::MAX)).unwrap();
            let expected: Vec<u32> = a
                .iter()
                .zip(&b)
                .flat_map(|(&a, &b)| {
                    let sum = u128::from(a) + u128::from(b);
                    let difference = a.wrapping_sub(b);
                    let product = u128::from(a) * u128::from(b);
                    [
                        word(sum),
                        word(sum >> 32),
                        word(sum >> 64),
                        word(u128::from(difference)),
                        word(u128::from(difference >> 32)),
                        u32::from(a < b),
                        word(product),
                        word(product >> 32),
                        word(product >> 64),
                        word(product >> 96),
                    ]
                })
                .collect();
            assert_eq!(device.download(&output)?.observe(asynchronous)?, expected);
        }
        device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
    }
    Ok(())
}

#[test]
fn test_floating_special_values_nan_selection_and_signed_zero() -> Result {
    for (format, width, nan, one, negative_zero, positive_infinity) in [
        ("f16", 16, 0x7e01u64, 0x3c00, 0x8000, 0x7c00),
        ("bf16", 16, 0x7fc1, 0x3f80, 0x8000, 0x7f80),
        (
            "f32",
            32,
            0x7fc0_0001,
            0x3f80_0000,
            0x8000_0000,
            0x7f80_0000,
        ),
        (
            "f64",
            64,
            0x7ff8_0000_0000_0001,
            0x3ff0_0000_0000_0000,
            0x8000_0000_0000_0000,
            0x7ff0_0000_0000_0000,
        ),
    ] {
        let reg = if width == 16 {
            "h"
        } else if width == 32 {
            "r"
        } else {
            "d"
        };
        let is_nan = |bits: u64| match format {
            "f16" => F16::from_bits(u16::try_from(bits & 0xffff).unwrap()).is_nan(),
            "bf16" => Bf16::from_bits(u16::try_from(bits & 0xffff).unwrap()).is_nan(),
            "f32" => f32::from_bits(u32::try_from(bits & 0xffff_ffff).unwrap()).is_nan(),
            _ => f64::from_bits(bits).is_nan(),
        };
        let operations = vec![
            (
                format!("min.{format} {reg}0,{reg}1,{reg}2;"),
                nan,
                one,
                Some(one),
            ),
            (
                format!("max.{format} {reg}0,{reg}1,{reg}2;"),
                nan,
                one,
                Some(one),
            ),
            (
                format!("min.{format} {reg}0,{reg}1,{reg}2;"),
                nan,
                nan,
                None,
            ),
            (
                format!("min.{format} {reg}0,{reg}1,{reg}2;"),
                negative_zero,
                0,
                Some(negative_zero),
            ),
            (
                format!("max.{format} {reg}0,{reg}1,{reg}2;"),
                negative_zero,
                0,
                Some(0),
            ),
            (
                format!("fma.rn.{format} {reg}0,{reg}1,{reg}2,{reg}2;"),
                positive_infinity,
                0,
                None,
            ),
        ];
        let mut operations = operations;
        if format != "f64" {
            operations.push((
                format!("min.NaN.{format} {reg}0,{reg}1,{reg}2;"),
                nan,
                one,
                None,
            ));
            operations.push((
                format!("max.NaN.{format} {reg}0,{reg}1,{reg}2;"),
                one,
                nan,
                None,
            ));
        }
        if width == 16 {
            operations.push((
                format!("fma.rn.relu.{format} {reg}0,{reg}1,{reg}2,{reg}2;"),
                nan,
                one,
                None,
            ));
        }
        for (instruction, a, b, expected) in operations {
            let source = format!(
                "{PRELUDE} .global .u64 output;.entry k() {{.reg .b{width} {reg}<3>;mov.b{width} {reg}1,{a};mov.b{width} {reg}2,{b};{instruction}st.global.b{width} [output],{reg}0;ret;}}"
            );
            let device = Emulator::create(EmulatorOptions::default())?;
            let module = device.load(ParsedModule::parse(&source)?)?.wait()?;
            module
                .kernel("k")?
                .launch(LaunchConfig::create([1, 1, 1], [1, 1, 1], 0)?, [])?
                .wait()?;
            let bits = u64::from_le_bytes(
                device
                    .download(&module.global("output")?)?
                    .wait()?
                    .try_into()
                    .unwrap(),
            );
            if let Some(expected) = expected {
                assert_eq!(bits, expected, "{instruction}");
            } else {
                assert!(is_nan(bits), "{instruction}: {bits:x}");
            }
            device.shutdown(ShutdownMode::Drain).wait()?;
        }
    }
    Ok(())
}

#[test]
fn test_half_declarations_use_bit_storage_for_initialization_and_immediates() -> Result {
    for declaration in [
        ".global .f16 input = 1.5;",
        ".global .f16x2 input = 0x40003e00;",
    ] {
        let parsed = ParsedModule::parse(&format!("{PRELUDE} {declaration} .entry k() {{ret;}}"))?;
        assert!(
            parsed
                .compile("k")
                .unwrap_err()
                .to_string()
                .contains("half storage cannot have an initializer")
        );
    }
    let parsed = ParsedModule::parse(&format!(
        "{PRELUDE} .entry k() {{.reg .f16 h;add.f16 h,1.5,1.0;ret;}}"
    ))?;
    assert!(parsed.compile("k").is_err());
    Ok(())
}

#[test]
fn test_calls_do_not_pass_carry_to_the_callee_and_faults_remain_inspectable() -> Result {
    let source = format!(
        "{PRELUDE} .global .u32 output; .func helper() {{.reg .u32 r;addc.u32 r,0,0;ret;}} .entry k() {{.reg .u32 r;st.global.u32 [output],33;add.cc.u32 r,0xffffffff,1;call helper,();ret;}}"
    );
    let device = Emulator::create(EmulatorOptions::default())?;
    let module = device.load(ParsedModule::parse(&source)?)?.wait()?;
    let error = module
        .kernel("k")?
        .launch(LaunchConfig::create([1, 1, 1], [1, 1, 1], 0)?, [])?
        .wait()
        .unwrap_err();
    assert!(error.to_string().contains("carry flag"));
    assert!(error.to_string().contains("helper"));
    assert_eq!(
        device.download(&module.global("output")?)?.wait()?,
        33u32.to_le_bytes()
    );
    assert!(device.shutdown(ShutdownMode::Drain).wait().is_err());
    Ok(())
}
