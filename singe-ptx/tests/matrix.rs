//! The selected `sm_80` matrix path against independent host computations.

use std::{env, fs};

use anyhow::Result;
use singe_ptx::emulator::{
    Emulator, EmulatorOptions, ExecutionError, F16, LaunchConfig, ParsedModule, ShutdownMode,
};

mod support;

use support::Observe;

const MMA: &str = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32";
const ROWS: &str = "ldmatrix.sync.aligned.m8n8.x4.shared.b16";
const COLUMNS: &str = "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16";

#[test]
fn test_tiled_half_matrix_multiplication_matches_exact_and_bounded_references() -> Result<()> {
    let parsed = ParsedModule::parse(include_str!("fixtures/execution/matrix_m16n8k16.ptx"))?;
    let shapes = [
        (0u32, 9u32, 17u32),
        (17, 0, 17),
        (17, 9, 0),
        (1, 1, 1),
        (16, 8, 16),
        (17, 9, 17),
        (33, 17, 33),
        (15, 7, 15),
        (16, 8, 1),
        (17, 9, 31),
        (31, 17, 32),
        (33, 33, 33),
    ];
    let mut comparisons = Vec::new();
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let module = device.load(parsed.clone())?.observe(asynchronous)?;
        for (m, n, k) in shapes {
            for exact in [true, false] {
                let a: Vec<_> = (0..m * k)
                    .map(|i| {
                        if exact {
                            F16::from_f32(((i * 17 + 3) % 31) as f32 / 8.0 - 15.0 / 8.0)
                        } else {
                            let bits = i.wrapping_mul(0x9e37_79b9).wrapping_add(0x3012);
                            F16::from_bits(
                                (u16::try_from(bits >> 16).unwrap() & 0x8000)
                                    | (u16::try_from(bits & 0xffff).unwrap() % 0x7c00),
                            )
                        }
                    })
                    .collect();
                let b: Vec<_> = (0..k * n)
                    .map(|i| {
                        if exact {
                            F16::from_f32(((i * 13 + 7) % 29) as f32 / 8.0 - 14.0 / 8.0)
                        } else {
                            let bits = i.wrapping_mul(0x85eb_ca6b).wrapping_add(0x8071);
                            F16::from_bits(
                                (u16::try_from(bits >> 16).unwrap() & 0x8000)
                                    | (u16::try_from(bits & 0xffff).unwrap() % 0x7c00),
                            )
                        }
                    })
                    .collect();
                let input_a = device.upload(&a)?.observe(asynchronous)?;
                let input_b = device.upload(&b)?.observe(asynchronous)?;
                let output = device
                    .zeroed::<f32>((m * n) as usize)?
                    .observe(asynchronous)?;
                module
                    .kernel("matrix")?
                    .launch(
                        LaunchConfig::create(
                            [n.div_ceil(8).max(1), m.div_ceil(16).max(1), 1],
                            [32, 1, 1],
                            0,
                        )?,
                        [
                            (&input_a).into(),
                            (&input_b).into(),
                            (&output).into(),
                            m.into(),
                            n.into(),
                            k.into(),
                        ],
                    )?
                    .observe(asynchronous)?;
                let actual = device.download(&output)?.observe(asynchronous)?;
                if exact {
                    comparisons.push(serde_json::json!({"m":m,"n":n,"k":k,"asynchronous":asynchronous,"values":actual}));
                }
                let operations = f64::from(k.div_ceil(16) * 16);
                let unit = 2f64.powi(-24);
                let gamma = operations * unit / f64::mul_add(operations, -unit, 1.0);
                for row in 0..m as usize {
                    for col in 0..n as usize {
                        let mut reference = 0.0f64;
                        let mut magnitude = 0.0f64;
                        for inner in 0..k as usize {
                            let product = f64::from(a[row * k as usize + inner].to_f32())
                                * f64::from(b[inner * n as usize + col].to_f32());
                            reference += product;
                            magnitude += product.abs();
                        }
                        let value = f64::from(actual[row * n as usize + col]);
                        if exact {
                            assert_eq!(
                                value, reference,
                                "{m}x{n}x{k}, ({row},{col}), async={asynchronous}"
                            );
                        } else {
                            let bound = f64::mul_add(
                                operations,
                                f64::mul_add(f64::EPSILON, magnitude, 2f64.powi(-149)),
                                gamma * magnitude,
                            );
                            assert!(
                                value.is_finite() && (value - reference).abs() <= bound,
                                "{m}x{n}x{k}, ({row},{col}): {value} != {reference}, bound={bound}"
                            );
                        }
                    }
                }
            }
        }
        device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
    }
    if let Some(path) = env::var_os("SINGE_PTX_MATRIX_RESULTS") {
        fs::write(path, serde_json::to_vec(&comparisons)?)?;
    }
    Ok(())
}

fn source(body: &str) -> String {
    format!(
        ".version 9.4 .target sm_80 .address_size 64
        .global .align 16 .u32 output[512];
        .entry k() {{ .shared .align 16 .b8 tile[2048];
            .reg .u32 t; .reg .u32 r; .reg .u32 count;
            .reg .u64 address; .reg .u64 offset; .reg .u64 base;
            .reg .b32 a<4>; .reg .b32 b<2>; .reg .f32 c<4>;
            .reg .b16 h; .reg .pred p; .reg .pred q;
            mov.u32 t,%tid.x;
            {body}
            ret; }}"
    )
}

fn initialized() -> &'static str {
    "mov.b32 a0,0x3c003c00; mov.b32 a1,0x3c003c00;
     mov.b32 a2,0x3c003c00; mov.b32 a3,0x3c003c00;
     mov.b32 b0,0x3c003c00; mov.b32 b1,0x3c003c00;
     mov.f32 c0,0f00000000; mov.f32 c1,0f00000000;
     mov.f32 c2,0f00000000; mov.f32 c3,0f00000000;"
}

fn store_accumulators() -> &'static str {
    "mul.wide.u32 offset,t,16; mov.u64 base,output; add.u64 address,base,offset;
     st.global.v4.f32 [address],{c0,c1,c2,c3};"
}

fn run(body: &str, block: u32, asynchronous: bool) -> Result<Vec<u32>> {
    let device = Emulator::create(EmulatorOptions::default())?;
    let module = device
        .load(ParsedModule::parse(&source(body))?)?
        .observe(asynchronous)?;
    module
        .kernel("k")?
        .launch(LaunchConfig::create([1; 3], [block, 1, 1], 0)?, [])?
        .observe(asynchronous)?;
    let bytes = device
        .download(&module.global("output")?)?
        .observe(asynchronous)?;
    device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| u32::from_le_bytes(*bytes))
        .collect())
}

#[test]
fn test_matrix_inputs_are_snapshotted_before_writing_aliased_destinations() -> Result<()> {
    let body = format!(
        "{} add.cc.u32 r,0xffffffff,1;
        {MMA} {{a0,a1,a2,a3}},{{a0,a1,a2,a3}},{{b0,b1}},{{c0,c1,c2,c3}};
        mov.b32 c0,a0; mov.b32 c1,a1; mov.b32 c2,a2; mov.b32 c3,a3;
        {MMA} {{c0,c1,c2,c3}},{{a0,a1,a2,a3}},{{b0,b1}},{{c0,c1,c2,c3}};
        addc.u32 r,0,0; st.global.u32 [output+2044],r;
        {}",
        initialized(),
        store_accumulators()
    );
    // 16.0f32 has halfwords [0,2.75]. The second operation therefore adds
    // eight 2.75 products to its original 16.0 accumulator.
    for asynchronous in [false, true] {
        let values = run(&body, 32, asynchronous)?;
        assert!(values[..128].iter().all(|&value| value == 38f32.to_bits()));
        assert_eq!(values[511], 1);
    }
    Ok(())
}

#[test]
fn test_matrix_sources_accept_literals_including_mixed_register_vectors() -> Result<()> {
    for asynchronous in [false, true] {
        let body = format!(
            "{MMA} {{c0,c1,c2,c3}},
            {{0x3c003c00,0x3c003c00,0x3c003c00,0x3c003c00}},{{0x3c003c00,0x3c003c00}},
            {{0f3f800000,0f3f800000,0f3f800000,0f3f800000}};
            {}",
            store_accumulators()
        );
        assert!(
            run(&body, 32, asynchronous)?[..128]
                .iter()
                .all(|&value| value == 17f32.to_bits())
        );
        let body = format!(
            "{} {MMA} {{c0,c1,c2,c3}},{{0,a1,a2,a3}},{{b0,b1}},{{c0,c1,c2,c3}};
            {}",
            initialized(),
            store_accumulators()
        );
        let values = run(&body, 32, asynchronous)?;
        for lane in 0..32 {
            assert_eq!(
                &values[lane * 4..lane * 4 + 4],
                &[
                    8f32.to_bits(),
                    8f32.to_bits(),
                    16f32.to_bits(),
                    16f32.to_bits()
                ]
            );
        }
    }
    Ok(())
}

#[test]
fn test_discarded_matrix_results_preserve_other_destinations_and_synchronize() -> Result<()> {
    for asynchronous in [false, true] {
        let body = format!(
            "{} {MMA} {{_,c1,c2,c3}},{{a0,a1,a2,a3}},{{b0,b1}},{{c0,c1,c2,c3}};
            {}",
            initialized(),
            store_accumulators()
        );
        let values = run(&body, 32, asynchronous)?;
        for lane in 0..32 {
            assert_eq!(
                &values[lane * 4..lane * 4 + 4],
                &[0, 16f32.to_bits(), 16f32.to_bits(), 16f32.to_bits()]
            );
        }
        for (head, destination, store) in [(ROWS, "{_,a1,a2,a3}", "v4"), (COLUMNS, "{_,a1}", "v2")]
        {
            let body = format!(
                "mul.wide.u32 offset,t,16; mov.u64 base,tile; add.u64 address,base,offset;
                mov.u32 r,0x00010001; st.shared.v4.u32 [address],{{r,r,r,r}}; bar.sync 0;
                mov.b32 a0,77; {head} {destination},[address];
                mov.u64 base,output; add.u64 address,base,offset;
                st.global.{store}.b32 [address],{};",
                if store == "v4" {
                    "{a0,a1,a2,a3}"
                } else {
                    "{a0,a1}"
                }
            );
            let values = run(&body, 32, asynchronous)?;
            for lane in 0..32 {
                assert_eq!(values[lane * 4], 77);
                assert_eq!(values[lane * 4 + 1], 0x0001_0001);
                if store == "v4" {
                    assert_eq!(&values[lane * 4 + 2..lane * 4 + 4], &[0x0001_0001; 2]);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn test_matrix_loads_map_noncontiguous_rows_and_ignore_unused_addresses() -> Result<()> {
    for asynchronous in [false, true] {
        for block in [32, 64] {
            for transpose in [false, true] {
                let body = format!(
                    "
                    {}
                    xor.b32 count,t,7; mul.wide.u32 offset,count,32;
                    mov.u64 base,tile; add.u64 address,base,offset; mov.u32 count,0;
                    fill:
                    mad.lo.u32 r,t,8,count; add.u32 r,r,1; cvt.u16.u32 h,r;
                    mul.wide.u32 offset,count,2; add.u64 base,address,offset;
                    st.shared.b16 [base],h;
                    add.u32 count,count,1; setp.lt.u32 p,count,8; @p bra fill;
                    cvt.u32.u64 a0,address;
                    ready: bar.sync 0; add.cc.u32 r,0xffffffff,1;
                    {} {},[a0];
                    addc.u32 r,0,0; st.global.u32 [output+2044],r;
                    mul.wide.u32 offset,t,16; mov.u64 base,output; add.u64 address,base,offset;
                    st.global.{}.b32 [address],{};",
                    if transpose {
                        "and.b32 r,t,31; setp.ge.u32 p,r,16; @p bra ready;"
                    } else {
                        ""
                    },
                    if transpose { COLUMNS } else { ROWS },
                    if transpose {
                        "{a0,a1}"
                    } else {
                        "{a0,a1,a2,a3}"
                    },
                    if transpose { "v2" } else { "v4" },
                    if transpose {
                        "{a0,a1}"
                    } else {
                        "{a0,a1,a2,a3}"
                    }
                );
                let result = run(&body, block, asynchronous)?;
                for lane in 0..block as usize {
                    let warp = lane / 32 * 32;
                    let group = lane % 32 / 4;
                    let pair = lane % 4 * 2;
                    for matrix in 0..if transpose { 2 } else { 4 } {
                        let (low, high) = if transpose {
                            (
                                (warp + matrix * 8 + pair) * 8 + group + 1,
                                (warp + matrix * 8 + pair + 1) * 8 + group + 1,
                            )
                        } else {
                            let low = (warp + matrix * 8 + group) * 8 + pair + 1;
                            (low, low + 1)
                        };
                        assert_eq!(
                            result[lane * 4 + matrix],
                            u32::try_from(low).unwrap() | u32::try_from(high).unwrap() << 16,
                            "transpose={transpose}, block={block}, lane={lane}, matrix={matrix}"
                        );
                    }
                }
                assert_eq!(result[511], 1);
            }
        }
    }
    Ok(())
}

#[test]
fn test_matrix_rendezvous_handles_divergent_arrival_times_and_repeated_generations() -> Result<()> {
    let body = format!(
        "{}
        cvt.rn.f32.u32 c0,t; mov.f32 c1,c0; mov.f32 c2,c0; mov.f32 c3,c0;
        setp.lt.u32 p,t,16; @p bra join;
        mov.u32 count,t;
        delay: sub.u32 count,count,1; setp.ne.u32 p,count,0; @p bra delay;
        join: mov.u32 count,3;
        repeat: {MMA} {{c0,c1,c2,c3}},{{a0,a1,a2,a3}},{{b0,b1}},{{c0,c1,c2,c3}};
        sub.u32 count,count,1; setp.ne.u32 p,count,0; @p bra repeat;
        {}",
        initialized(),
        store_accumulators()
    );
    for asynchronous in [false, true] {
        let values = run(&body, 32, asynchronous)?;
        for lane in 0..32 {
            assert_eq!(
                &values[lane * 4..lane * 4 + 4],
                &[(48.0 + lane as f32).to_bits(); 4]
            );
        }
        let disabled = format!(
            "mov.pred p,0; add.cc.u32 r,0xffffffff,1;
            @p {MMA} {{c0,c1,c2,c3}},{{a0,a1,a2,a3}},{{b0,b1}},{{c0,c1,c2,c3}};
            @p {ROWS} {{a0,a1,a2,a3}},[address]; @p {COLUMNS} {{b0,b1}},[address];
            addc.u32 r,0,0; st.global.u32 [output],r;"
        );
        assert_eq!(run(&disabled, 17, asynchronous)?[0], 1);
    }
    Ok(())
}

#[test]
fn test_matrix_special_values_follow_the_documented_software_accumulation() -> Result<()> {
    let cases: [(u16, u16, u16, u16, u32); 15] = [
        (0, 0, 0x3c00, 0x3c00, 0x0000_0001),
        (0, 0, 0x3c00, 0x3c00, 0x8000_0001),
        (0x7c00, 0x7c00, 0, 0, 0),
        (0x0001, 0x8001, 0x3c00, 0xbc00, 0),
        (0x0001, 0x0001, 0x0001, 0x0001, 0x0000_0001),
        (0x8000, 0x8000, 0x3c00, 0x3c00, 0x8000_0000),
        (0, 0, 0x3c00, 0xbc00, 0x8000_0000),
        (0x7bff, 0x7bff, 0x7bff, 0x7bff, 0),
        (0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x4b80_0000),
        (0x3c00, 0xbc00, 0x3c00, 0x3c00, 0x3f80_0000),
        (0x7c00, 0x7c00, 0x3c00, 0x3c00, 0),
        (0xfc00, 0xfc00, 0x3c00, 0x3c00, 0),
        (0x7c00, 0xfc00, 0x3c00, 0x3c00, 0),
        (0x7c01, 0x7e55, 0x3c00, 0x3c00, 0),
        (0x3c00, 0x3c00, 0, 0, 0x7fc1_2345),
    ];
    for asynchronous in [false, true] {
        for (al, ah, bl, bh, seed) in cases {
            let a = u32::from(al) | u32::from(ah) << 16;
            let b = u32::from(bl) | u32::from(bh) << 16;
            let body = format!(
                "mov.b32 a0,{a}; mov.b32 a1,{a}; mov.b32 a2,{a}; mov.b32 a3,{a};
                mov.b32 b0,{b}; mov.b32 b1,{b};
                mov.f32 c0,0f{seed:08x}; mov.f32 c1,c0; mov.f32 c2,c0; mov.f32 c3,c0;
                {MMA} {{c0,c1,c2,c3}},{{a0,a1,a2,a3}},{{b0,b1}},{{c0,c1,c2,c3}};
                {}",
                store_accumulators()
            );
            let mut expected = f32::from_bits(seed);
            for inner in 0..16 {
                let a = F16::from_bits(if inner % 2 == 0 { al } else { ah }).to_f32();
                let b = F16::from_bits(if inner % 2 == 0 { bl } else { bh }).to_f32();
                expected = a.mul_add(b, expected);
            }
            let values = run(&body, 32, asynchronous)?;
            for &value in &values[..128] {
                if expected.is_nan() {
                    assert_eq!(value, 0x7fc0_0000);
                } else {
                    assert_eq!(
                        value,
                        expected.to_bits(),
                        "a={a:08x}, b={b:08x}, seed={seed:08x}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn test_invalid_matrix_participants_and_memory_produce_located_faults() -> Result<()> {
    let mma = format!("{MMA} {{c0,c1,c2,c3}},{{a0,a1,a2,a3}},{{b0,b1}},{{c0,c1,c2,c3}};");
    let mut cases = vec![
        (format!("{}{mma}", initialized()), 17, "full 32-lane"),
        (
            format!("{} setp.eq.u32 p,t,17; @p ret; {mma}", initialized()),
            32,
            "exited",
        ),
        (
            format!(
                "{} setp.lt.u32 p,t,16; @p bra left; {mma} bra done; left: {mma} done:",
                initialized()
            ),
            32,
            "different instructions",
        ),
        (
            format!(
                "{} setp.lt.u32 p,t,16; @p bra matrix; bar.warp.sync 0xffffffff; bra done; matrix: {mma} done:",
                initialized()
            ),
            32,
            "stalled",
        ),
        (
            format!("{ROWS} {{a0,a1,a2,a3}},[tile];"),
            32,
            "indeterminate",
        ),
        (
            format!("{ROWS} {{a0,a1,a2,a3}},[address];"),
            32,
            "indeterminate",
        ),
        (
            format!("{ROWS} {{a0,a1,a2,a3}},[tile+2];"),
            32,
            "misaligned",
        ),
        (format!("{COLUMNS} {{b0,b1}},[tile+4096];"), 32, "bounds"),
    ];
    for missing in ["mov.b32 a2,0x3c003c00;", "mov.f32 c3,0f00000000;"] {
        cases.push((
            format!("{}{mma}", initialized().replace(missing, "")),
            32,
            "indeterminate",
        ));
    }
    for asynchronous in [false, true] {
        for (body, block, expected) in &cases {
            let text = source(&format!("st.global.u32 [output],77; {body}"));
            let device = Emulator::create(EmulatorOptions::default())?;
            let module = device
                .load(ParsedModule::parse(&text)?)?
                .observe(asynchronous)?;
            let error = module
                .kernel("k")?
                .launch(LaunchConfig::create([1; 3], [*block, 1, 1], 0)?, [])?
                .observe(asynchronous)
                .unwrap_err();
            assert!(error.to_string().contains(expected), "{body}: {error}");
            assert!(error.launch_report().is_some());
            if !matches!(error.root_cause(), ExecutionError::Stall(_)) {
                let ExecutionError::Launch { cause, .. } = &error else {
                    panic!("{error}");
                };
                let ExecutionError::Lane { location, .. } = cause.as_ref() else {
                    panic!("{error}");
                };
                assert_eq!(location.function, "k");
                assert!(!text[location.span.start..location.span.end].is_empty());
            }
            let output = device
                .download(&module.global("output")?)?
                .observe(asynchronous)?;
            assert_eq!(&output[..4], &77u32.to_le_bytes());
            assert!(
                device
                    .shutdown(ShutdownMode::Drain)
                    .observe(asynchronous)
                    .is_err()
            );
        }
    }
    Ok(())
}

#[test]
fn test_matrix_compilation_limits_shapes_layouts_types_and_targets() -> Result<()> {
    let text = include_str!("fixtures/execution/matrix_m16n8k16.ptx");
    ParsedModule::parse(&text.replace(".version 9.4", ".version 7.0"))?.compile("matrix")?;
    for text in [
        text.replace(".target sm_80", ".target sm_75"),
        text.replace(".target sm_80", ".target sm_90"),
    ] {
        assert!(ParsedModule::parse(&text)?.compile("matrix").is_err());
    }
    for instruction in [
        "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {a0},[tile];",
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {b0,b1},[tile];",
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {a0,a1,a2,a3},[tile];",
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {c0,c1,c2,c3},{a0,a1},{b0},{c0,c1,c2,c3};",
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {c0,c1,c2,c3},{a0,a1,a2,a3},{b0,b1},{c0,c1,c2,c3};",
    ] {
        assert!(
            ParsedModule::parse(&source(instruction))?
                .compile("k")
                .is_err(),
            "{instruction}"
        );
    }
    let explicit = source("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {a0,a1,a2,a3},[tile];");
    ParsedModule::parse(&explicit.replace(".version 9.4", ".version 7.8"))?.compile("k")?;
    assert!(
        ParsedModule::parse(&explicit.replace(".version 9.4", ".version 7.7"))?
            .compile("k")
            .unwrap_err()
            .to_string()
            .contains("requires PTX 7.8")
    );
    Ok(())
}

#[test]
fn test_matrix_rendezvous_uses_instruction_identity_across_call_paths() -> Result<()> {
    let text = format!(
        ".version 9.4 .target sm_80 .address_size 64 .global .f32 output[128];
        .func leaf() {{ .reg .f32 c<4>; .reg .u32 t;
            .reg .u64 offset; .reg .u64 base; .reg .u64 address;
            {MMA} {{c0,c1,c2,c3}},{{0x3c003c00,0x3c003c00,0x3c003c00,0x3c003c00}},
                {{0x3c003c00,0x3c003c00}},{{0f00000000,0f00000000,0f00000000,0f00000000}};
            mov.u32 t,%tid.x; {} ret;
        }}
        .func nested() {{ call leaf; ret; }}
        .entry k() {{ .reg .u32 t; .reg .pred p; mov.u32 t,%tid.x;
            setp.lt.u32 p,t,16; @p bra direct;
            call nested; bra done;
            direct: call leaf;
            done: ret;
        }}",
        store_accumulators()
    );
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let module = device
            .load(ParsedModule::parse(&text)?)?
            .observe(asynchronous)?;
        module
            .kernel("k")?
            .launch(LaunchConfig::create([1; 3], [32, 1, 1], 0)?, [])?
            .observe(asynchronous)?;
        let output = device
            .download(&module.global("output")?)?
            .observe(asynchronous)?;
        assert_eq!(
            output,
            (0..128)
                .flat_map(|_| 16f32.to_le_bytes())
                .collect::<Vec<_>>()
        );
        device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
    }
    Ok(())
}

#[test]
fn test_matrix_membership_uses_linear_lanes_in_multidimensional_blocks() -> Result<()> {
    let body = format!(
        "mov.u32 t,%tid.z; mov.u32 r,%ntid.y; mul.lo.u32 t,t,r;
        mov.u32 r,%tid.y; add.u32 t,t,r;
        mov.u32 r,%ntid.x; mul.lo.u32 t,t,r;
        mov.u32 r,%tid.x; add.u32 t,t,r;
        cvt.rn.f32.u32 c0,t; mov.f32 c1,c0; mov.f32 c2,c0; mov.f32 c3,c0;
        {MMA} {{c0,c1,c2,c3}},{{0x3c003c00,0x3c003c00,0x3c003c00,0x3c003c00}},
            {{0x3c003c00,0x3c003c00}},{{c0,c1,c2,c3}};
        {}",
        store_accumulators()
    );
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let module = device
            .load(ParsedModule::parse(&source(&body))?)?
            .observe(asynchronous)?;
        for block in [[8, 4, 1], [4, 4, 4]] {
            let lanes = block.into_iter().product::<u32>() as usize;
            module
                .kernel("k")?
                .launch(LaunchConfig::create([1; 3], block, 0)?, [])?
                .observe(asynchronous)?;
            let output = device
                .download(&module.global("output")?)?
                .observe(asynchronous)?;
            let expected: Vec<_> = (0..lanes)
                .flat_map(|lane| (0..4).flat_map(move |_| (16.0 + lane as f32).to_le_bytes()))
                .collect();
            assert_eq!(&output[..expected.len()], expected);
        }
        device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
    }
    Ok(())
}

#[cfg(feature = "cuda_tests")]
#[test]
fn test_selected_matrix_fixture_assembles_for_sm80() -> Result<()> {
    use std::{
        env, fs,
        process::{self, Command},
    };
    let output = env::temp_dir().join(format!("singe-ptx-matrix-{}.cubin", process::id()));
    let result = Command::new("ptxas")
        .args([
            "-arch=sm_80",
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/execution/matrix_m16n8k16.ptx"
            ),
            "-o",
        ])
        .arg(&output)
        .output()?;
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    fs::remove_file(output)?;
    Ok(())
}
