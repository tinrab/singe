use singe_ptx::emulator::{Emulator, EmulatorOptions, LaunchConfig, ParsedModule, ShutdownMode};

fn execute(body: &str) -> u64 {
    let source = format!(
        ".version 9.3 .target sm_80 .address_size 64 .global .u64 output; .entry k() {{ .reg .b32 r; .reg .b64 d; .reg .pred p; {body} ret; }}"
    );
    let device = Emulator::create(EmulatorOptions::default()).unwrap();
    let module = device
        .load(ParsedModule::parse(&source).unwrap())
        .unwrap()
        .wait()
        .unwrap();
    module
        .kernel("k")
        .unwrap()
        .launch(LaunchConfig::create([1, 1, 1], [1, 1, 1], 0).unwrap(), [])
        .unwrap()
        .wait()
        .unwrap();
    let bytes = device
        .download(&module.global("output").unwrap())
        .unwrap()
        .wait()
        .unwrap();
    device.shutdown(ShutdownMode::Drain).wait().unwrap();
    u64::from_le_bytes(bytes.try_into().unwrap())
}

#[test]
fn test_binary64_arithmetic_preserves_single_nan_payloads() {
    for input in [
        0x7ff8_0000_0000_1234_u64,
        0xfff8_0000_0000_5678,
        0x7ff0_0000_0000_1234,
        0xfff0_0000_0000_5678,
    ] {
        for instruction in [
            "add.rn.f64 d,d,0d0000000000000000;",
            "sub.rn.f64 d,d,0d3ff0000000000000;",
            "mul.rn.f64 d,d,0d3ff0000000000000;",
            "div.rn.f64 d,d,0d3ff0000000000000;",
            "fma.rn.f64 d,d,0d3ff0000000000000,0d0000000000000000;",
            "rcp.rn.f64 d,d;",
            "sqrt.rn.f64 d,d;",
            "rsqrt.approx.f64 d,d;",
        ] {
            let actual = execute(&format!(
                "mov.b64 d,{input}; {instruction} st.global.b64 [output],d;"
            ));
            assert_eq!(
                actual & 0x7fff_ffff_ffff_ffff,
                (input | 0x0008_0000_0000_0000) & 0x7fff_ffff_ffff_ffff,
                "{instruction}, input {input:016x}"
            );
        }
        let input = input | (1 << 32);
        for operation in ["rcp", "rsqrt"] {
            assert_eq!(
                execute(&format!(
                    "mov.b64 d,{input}; {operation}.approx.ftz.f64 d,d; st.global.b64 [output],d;"
                )),
                0x7fff_ffff_0000_0000
            );
        }
    }
}

#[test]
fn test_rounding_ftz_signed_zero_saturation_and_fused_arithmetic() {
    for (instruction, expected) in [
        ("add.rn.f32 r,1.0,0f33800000;", 0x3f80_0000),
        ("add.rz.f32 r,1.0,0f33800000;", 0x3f80_0000),
        ("add.rm.f32 r,1.0,0f33800000;", 0x3f80_0000),
        ("add.rp.f32 r,1.0,0f33800000;", 0x3f80_0001),
        ("add.rn.ftz.f32 r,0f80000001,0f80000000;", 0x8000_0000),
        ("add.rn.f32 r,0f00000001,0f00000001;", 2),
        ("min.f32 r,0f00000000,0f80000000;", 0x8000_0000),
        ("max.f32 r,0f80000000,0f00000000;", 0),
        ("div.rn.f32 r,1.0,0.0;", 0x7f80_0000),
        ("add.rn.sat.f32 r,0f7fc12345,1.0;", 0),
        ("fma.rn.f32 r,0f3f800001,0f3f7ffffe,-1.0;", 0xa880_0000),
    ] {
        assert_eq!(
            execute(&format!("{instruction} st.global.b32 [output],r;")),
            expected,
            "{instruction}"
        );
    }
    let nan = u32::try_from(
        execute("div.rn.f32 r,0.0,0.0; st.global.b32 [output],r;") & u64::from(u32::MAX),
    )
    .unwrap();
    assert!(f32::from_bits(nan).is_nan());
}

#[test]
fn test_binary64_conversions_nans_and_integer_extension() {
    for (instruction, expected) in [
        ("cvt.rp.f64.u64 d,9007199254740993;", 0x4340_0000_0000_0001),
        ("cvt.rn.f64.u64 d,9007199254740993;", 0x4340_0000_0000_0000),
        ("cvt.s8.s32 d,128;", 0xffff_ffff_ffff_ff80),
        ("cvt.u8.s32 d,-1;", 255),
        ("cvt.rzi.u64.f32 d,0f7fc00000;", 0x8000_0000_0000_0000),
        (
            "cvt.rzi.s32.f64 d,0d7ff8000000000000;",
            0xffff_ffff_8000_0000,
        ),
        ("cvt.rzi.u32.f64 d,0d7ff8000000000000;", 0x8000_0000),
        ("cvt.rzi.u32.f32 d,0f7fc00000;", 0),
        ("cvt.rzi.s32.f32 d,0f7f800000;", 2_147_483_647),
    ] {
        assert_eq!(
            execute(&format!("{instruction} st.global.b64 [output],d;")),
            expected,
            "{instruction}"
        );
    }
    assert_eq!(
        execute(
            "setp.ne.f64 p,0d7ff8000000000000,0.0; selp.u32 r,1,0,p; st.global.u32 [output],r;"
        ),
        0
    );
    assert_eq!(
        execute(
            "setp.neu.f64 p,0d7ff8000000000000,0.0; selp.u32 r,1,0,p; st.global.u32 [output],r;"
        ),
        1
    );
}

#[test]
fn test_ptx_normal_classification_includes_both_zeros_but_excludes_subnormals() {
    for (ty, literal, expected) in [
        ("f32", "0f00000000", 1),
        ("f32", "0f80000000", 1),
        ("f32", "0f00000001", 0),
        ("f32", "0f007fffff", 0),
        ("f32", "0f00800000", 1),
        ("f32", "0f80800000", 1),
        ("f32", "0f7f7fffff", 1),
        ("f32", "0f7f800000", 0),
        ("f32", "0f7fc00000", 0),
        ("f64", "0d0000000000000000", 1),
        ("f64", "0d8000000000000000", 1),
        ("f64", "0d0000000000000001", 0),
        ("f64", "0d000fffffffffffff", 0),
        ("f64", "0d0010000000000000", 1),
        ("f64", "0d8010000000000000", 1),
        ("f64", "0d7fefffffffffffff", 1),
        ("f64", "0d7ff0000000000000", 0),
        ("f64", "0d7ff8000000000000", 0),
    ] {
        assert_eq!(
            execute(&format!(
                "testp.normal.{ty} p,{literal}; selp.u32 r,1,0,p; st.global.u32 [output],r;"
            )),
            expected,
            "{ty}: {literal}"
        );
    }
}

#[test]
fn test_integer_wrapping_products_and_shift_clipping() {
    for (instruction, expected) in [
        ("add.u64 d,18446744073709551615,1;", 0),
        ("sub.u64 d,0,1;", u64::MAX),
        (
            "mul.hi.u64 d,18446744073709551615,18446744073709551615;",
            u64::MAX - 1,
        ),
        ("shr.s64 d,-1,100;", u64::MAX),
        ("shl.b64 d,1,64;", 0),
        ("mul.wide.s32 d,-2147483648,2;", 0xffff_ffff_0000_0000),
        ("div.s64 d,-19,7;", (-2i64) as u64),
        ("rem.s64 d,-19,7;", (-5i64) as u64),
    ] {
        assert_eq!(
            execute(&format!("{instruction} st.global.b64 [output],d;")),
            expected,
            "{instruction}"
        );
    }
}
