use std::{array, error::Error, fmt::Write, result::Result as StdResult};

mod support;

use singe_ptx::emulator::{
    Emulator, EmulatorOptions, F16, LaunchConfig, ParsedModule, ShutdownMode,
};
use support::Observe;

type Result = StdResult<(), Box<dyn Error>>;

const PRELUDE: &str = ".version 9.3 .target sm_80 .address_size 64";

fn config(n: usize) -> LaunchConfig {
    LaunchConfig::create(
        [u32::try_from(n.div_ceil(128)).unwrap(), 1, 1],
        [128, 1, 1],
        0,
    )
    .unwrap()
}

#[test]
fn test_dot_products_cover_signed_boundaries_and_packed_positions() -> Result {
    let bytes = [0u8, 1, 2, 0x7e, 0x7f, 0x80, 0x81, 0xff];
    let halves = [0u16, 1, 2, 0x7ffe, 0x7fff, 0x8000, 0x8001, 0xffff];
    let (a, b): (Vec<_>, Vec<_>) = bytes
        .iter()
        .flat_map(|&a| {
            bytes.iter().map(move |&b| {
                (
                    u32::from_le_bytes([
                        a,
                        a.wrapping_add(37),
                        a.wrapping_add(74),
                        a.wrapping_add(111),
                    ]),
                    u32::from_le_bytes([
                        b,
                        b.wrapping_add(53),
                        b.wrapping_add(106),
                        b.wrapping_add(159),
                    ]),
                )
            })
        })
        .chain(halves.iter().flat_map(|&a| {
            halves.iter().map(move |&b| {
                (
                    u32::from(a) | (u32::from(a.rotate_left(1)) << 16),
                    u32::from(b) | (u32::from(b.rotate_left(1)) << 16),
                )
            })
        }))
        .unzip();
    let count = a.len();
    let device = Emulator::create(EmulatorOptions::default())?;
    let input_a = device.upload(&a)?.wait()?;
    let input_b = device.upload(&b)?.wait()?;
    let output = device.zeroed::<u32>(count)?.wait()?;
    for mode in ["dp4a", "dp2a.lo", "dp2a.hi"] {
        for a_signed in [false, true] {
            for b_signed in [false, true] {
                let head = format!(
                    "{mode}.{}.{}",
                    if a_signed { "s32" } else { "u32" },
                    if b_signed { "s32" } else { "u32" }
                );
                let source = format!(
                    r"{PRELUDE}
.entry k(.param .u64 input_a,.param .u64 input_b,.param .u64 output) {{
 .reg .u32 i; .reg .u32 thread_id; .reg .b32 a; .reg .b32 b; .reg .b32 result;
 .reg .b64 ap; .reg .b64 bp; .reg .b64 op; .reg .b64 offset;
 mov.u32 thread_id,%tid.x;mov.u32 i,%ctaid.x;mad.lo.u32 i,i,128,thread_id;
 ld.param.u64 ap,[input_a];ld.param.u64 bp,[input_b];ld.param.u64 op,[output];
 mul.wide.u32 offset,i,4;add.u64 ap,ap,offset;add.u64 bp,bp,offset;add.u64 op,op,offset;
 ld.global.b32 a,[ap];ld.global.b32 b,[bp];{head} result,a,b,0x7ffffff0;st.global.b32 [op],result;ret;
}}"
                );
                let module = device.load(ParsedModule::parse(&source)?)?.wait()?;
                module
                    .kernel("k")?
                    .launch(
                        config(count),
                        [(&input_a).into(), (&input_b).into(), (&output).into()],
                    )?
                    .wait()?;
                let actual = device.download(&output)?.wait()?;
                for (i, ((&a, &b), &actual)) in a.iter().zip(&b).zip(&actual).enumerate() {
                    let bytes_a = a.to_le_bytes();
                    let bytes_b = b.to_le_bytes();
                    let mut expected = 0x7fff_fff0_i64;
                    for lane in 0..if mode == "dp4a" { 4 } else { 2 } {
                        let av = if mode == "dp4a" {
                            if a_signed {
                                i64::from(bytes_a[lane].cast_signed())
                            } else {
                                i64::from(bytes_a[lane])
                            }
                        } else {
                            let pair = [bytes_a[lane * 2], bytes_a[lane * 2 + 1]];
                            if a_signed {
                                i64::from(i16::from_le_bytes(pair))
                            } else {
                                i64::from(u16::from_le_bytes(pair))
                            }
                        };
                        let byte = bytes_b[lane + if mode == "dp2a.hi" { 2 } else { 0 }];
                        let bv = if b_signed {
                            i64::from(byte.cast_signed())
                        } else {
                            i64::from(byte)
                        };
                        expected += av * bv;
                    }
                    assert_eq!(
                        actual,
                        u32::try_from(expected.cast_unsigned() & u64::from(u32::MAX)).unwrap(),
                        "{head} input {i}"
                    );
                }
            }
        }
    }
    device.shutdown(ShutdownMode::Drain).wait()?;
    Ok(())
}

#[test]
fn test_int8_matrix_multiplication_handles_odd_reductions_and_multiple_ctas() -> Result {
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let mixed = include_str!("fixtures/execution/quantized.ptx");
        for signed_b in [false, true] {
            let source = if signed_b {
                mixed.replace("dp4a.s32.u32", "dp4a.s32.s32")
            } else {
                mixed.to_owned()
            };
            let module = device
                .load(ParsedModule::parse(&source)?)?
                .observe(asynchronous)?;
            for (m, n, k) in [
                (1usize, 1usize, 1usize),
                (3, 5, 3),
                (5, 3, 4),
                (3, 5, 5),
                (17, 17, 17),
                (31, 33, 31),
                (32, 32, 32),
                (33, 17, 33),
                (129, 1, 17),
            ] {
                let a: Vec<i8> = (0..m * k)
                    .map(|i| u8::try_from((i * 37 + 11) % 256).unwrap().cast_signed())
                    .collect();
                let b: Vec<u8> = (0..k * n)
                    .map(|i| u8::try_from((i * 53 + 7) % 256).unwrap())
                    .collect();
                let words = k.div_ceil(4);
                let packed_a: Vec<u32> = (0..m)
                    .flat_map(|row| {
                        let a = &a;
                        (0..words).map(move |word| {
                            u32::from_le_bytes(array::from_fn(|i| {
                                a.get(row * k + word * 4 + i)
                                    .filter(|_| word * 4 + i < k)
                                    .copied()
                                    .unwrap_or(0) as u8
                            }))
                        })
                    })
                    .collect();
                let packed_b: Vec<u32> = (0..n)
                    .flat_map(|col| {
                        let b = &b;
                        (0..words).map(move |word| {
                            u32::from_le_bytes(array::from_fn(|i| {
                                if word * 4 + i < k {
                                    b[(word * 4 + i) * n + col]
                                } else {
                                    0
                                }
                            }))
                        })
                    })
                    .collect();
                let input_a = device.upload(&packed_a)?.observe(asynchronous)?;
                let input_b = device.upload(&packed_b)?.observe(asynchronous)?;
                let output = device.zeroed::<i32>(m * n)?.observe(asynchronous)?;
                module
                    .kernel("int8_matmul")?
                    .launch(
                        config(m * n),
                        [
                            (&input_a).into(),
                            (&input_b).into(),
                            (&output).into(),
                            u32::try_from(m).unwrap().into(),
                            u32::try_from(n).unwrap().into(),
                            u32::try_from(words).unwrap().into(),
                        ],
                    )?
                    .observe(asynchronous)?;
                let actual = device.download(&output)?.observe(asynchronous)?;
                for row in 0..m {
                    for col in 0..n {
                        let expected: i64 = (0..k)
                            .map(|i| {
                                i64::from(a[row * k + i])
                                    * if signed_b {
                                        i64::from(b[i * n + col].cast_signed())
                                    } else {
                                        i64::from(b[i * n + col])
                                    }
                            })
                            .sum();
                        assert_eq!(
                            actual[row * n + col],
                            i32::try_from(expected).unwrap(),
                            "{m}x{n}x{k}, signed={signed_b}"
                        );
                    }
                }
            }
        }
        device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
    }
    Ok(())
}

#[test]
fn test_requantization_packs_signed_and_unsigned_subbyte_byte_and_halfword_results() -> Result {
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        for bits in [2u32, 4, 8, 16] {
            for signed in [false, true] {
                let kind = if signed { "s" } else { "u" };
                let per_word = 32 / bits;
                let mut body = String::new();
                for lane in 0..per_word {
                    writeln!(
                        body,
                        "add.u32 index,base,{lane};mov.s32 v{lane},0;setp.lt.u32 p,index,n;@!p bra skip{lane};mul.wide.u32 offset,index,4;add.u64 address,input,offset;ld.global.s32 v{lane},[address];shr.s32 v{lane},v{lane},3;skip{lane}:"
                    )
                    .unwrap();
                }
                body += "mov.b32 result,0;";
                for pair in (0..per_word / 2).rev() {
                    write!(
                        body,
                        "cvt.pack.sat.{kind}{bits}.s32{} result,v{},v{}{};",
                        if bits == 16 { "" } else { ".b32" },
                        pair * 2 + 1,
                        pair * 2,
                        if bits == 16 { "" } else { ",result" }
                    )
                    .unwrap();
                }
                let source = format!(
                    r"{PRELUDE}
.entry k(.param .u64 data,.param .u64 out,.param .u32 count) {{
 .reg .u32 thread; .reg .u32 thread_id; .reg .u32 base; .reg .u32 index; .reg .u32 n; .reg .u32 words;
 .reg .s32 v<{per_word}>; .reg .b32 result; .reg .pred p;
 .reg .b64 input; .reg .b64 output; .reg .b64 offset; .reg .b64 address;
 mov.u32 thread_id,%tid.x;mov.u32 thread,%ctaid.x;mad.lo.u32 thread,thread,128,thread_id;
 ld.param.u32 n,[count];add.u32 words,n,{tail};div.u32 words,words,{per_word};setp.ge.u32 p,thread,words;@p ret;
 ld.param.u64 input,[data];ld.param.u64 output,[out];mul.lo.u32 base,thread,{per_word};
 {body}
 mul.wide.u32 offset,thread,4;add.u64 output,output,offset;st.global.b32 [output],result;ret;
}}",
                    tail = per_word - 1
                );
                let module = device
                    .load(ParsedModule::parse(&source)?)?
                    .observe(asynchronous)?;
                for n in [1usize, 2, 3, 4, 5, 17, 31, 32, 33, 127, 128, 129, 257] {
                    let values: Vec<i32> = (0..n)
                        .map(|i| match i % 7 {
                            0 => i32::MIN,
                            1 => i32::MAX,
                            2 => -1,
                            3 => 0,
                            _ => i32::try_from(i).unwrap() * 137 - 10000,
                        })
                        .collect();
                    let input = device.upload(&values)?.observe(asynchronous)?;
                    let words = n.div_ceil(per_word as usize);
                    let output = device.zeroed::<u32>(words)?.observe(asynchronous)?;
                    module
                        .kernel("k")?
                        .launch(
                            config(words),
                            [
                                (&input).into(),
                                (&output).into(),
                                u32::try_from(n).unwrap().into(),
                            ],
                        )?
                        .observe(asynchronous)?;
                    let expected: Vec<u32> = values
                        .chunks(per_word as usize)
                        .map(|chunk| {
                            chunk.iter().enumerate().fold(0u32, |word, (i, &value)| {
                                let lo = if signed { -(1i64 << (bits - 1)) } else { 0 };
                                let hi = if signed {
                                    (1i64 << (bits - 1)) - 1
                                } else {
                                    (1i64 << bits) - 1
                                };
                                let value = i32::try_from((i64::from(value) >> 3).clamp(lo, hi))
                                    .unwrap()
                                    .cast_unsigned();
                                word | ((value & ((1 << bits) - 1))
                                    << (u32::try_from(i).unwrap() * bits))
                            })
                        })
                        .collect();
                    assert_eq!(
                        device.download(&output)?.observe(asynchronous)?,
                        expected,
                        "{kind}{bits}, n={n}"
                    );
                }
            }
        }
        device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
    }
    Ok(())
}

#[test]
fn test_fp16_masks_and_predicates_handle_nan_ftz_and_independent_packed_lanes() -> Result {
    for asynchronous in [false, true] {
        for ftz in [false, true] {
            let flag = if ftz { ".ftz" } else { "" };
            let source = format!(
                r"{PRELUDE}
.entry k(.param .u64 data,.param .u64 masks,.param .u64 predicates,.param .u32 count) {{
 .reg .u32 index; .reg .u32 thread_id; .reg .u32 n; .reg .pred p0; .reg .pred p1; .reg .pred tail;
 .reg .b32 value; .reg .b32 zero; .reg .b32 mask; .reg .b32 low; .reg .b32 high;
 .reg .b64 input; .reg .b64 output; .reg .b64 pred; .reg .b64 offset;
 mov.u32 thread_id,%tid.x;mov.u32 index,%ctaid.x;mad.lo.u32 index,index,128,thread_id;ld.param.u32 n,[count];setp.ge.u32 tail,index,n;@tail ret;
 ld.param.u64 input,[data];ld.param.u64 output,[masks];ld.param.u64 pred,[predicates];
 mul.wide.u32 offset,index,4;add.u64 input,input,offset;add.u64 output,output,offset;add.u64 pred,pred,offset;
 ld.global.b32 value,[input];mov.b32 zero,0;set.gt{flag}.u32.f16x2 mask,value,zero;st.global.u32 [output],mask;
 setp.gt{flag}.f16x2 p0|p1,value,zero;selp.u32 low,1,0,p0;selp.u32 high,1,0,p1;shl.b32 high,high,16;or.b32 low,low,high;st.global.b32 [pred],low;ret;
}}"
            );
            let device = Emulator::create(EmulatorOptions::default())?;
            let module = device
                .load(ParsedModule::parse(&source)?)?
                .observe(asynchronous)?;
            for n in [1usize, 3, 17, 31, 32, 33, 127, 128, 129, 257] {
                let values: Vec<F16> = (0..n)
                    .map(|i| {
                        F16::from_bits(
                            [
                                0, 0x8000, 1, 0x8001, 0x3c00, 0x4000, 0xbc00, 0x7e01, 0x7c00,
                                0xfc00,
                            ][i % 10],
                        )
                    })
                    .collect();
                let words: Vec<u32> = values
                    .chunks(2)
                    .map(|v| {
                        u32::from(v[0].to_bits())
                            | (u32::from(v.get(1).copied().unwrap_or(F16::ZERO).to_bits()) << 16)
                    })
                    .collect();
                let input = device.upload(&words)?.observe(asynchronous)?;
                let masks = device.zeroed::<u32>(words.len())?.observe(asynchronous)?;
                let predicates = device.zeroed::<u32>(words.len())?.observe(asynchronous)?;
                module
                    .kernel("k")?
                    .launch(
                        config(words.len()),
                        [
                            (&input).into(),
                            (&masks).into(),
                            (&predicates).into(),
                            u32::try_from(words.len()).unwrap().into(),
                        ],
                    )?
                    .observe(asynchronous)?;
                let truth = |v: F16| v.to_f32() > 0. && !(ftz && v.to_bits() & 0x7c00 == 0);
                let expected: Vec<u32> = values
                    .chunks(2)
                    .map(|v| {
                        u32::from(truth(v[0]))
                            | (u32::from(v.get(1).is_some_and(|&v| truth(v))) << 16)
                    })
                    .collect();
                assert_eq!(
                    device.download(&predicates)?.observe(asynchronous)?,
                    expected
                );
                assert_eq!(
                    device.download(&masks)?.observe(asynchronous)?,
                    expected
                        .iter()
                        .map(|v| (if v & 1 != 0 { 65535 } else { 0 })
                            | (if v & 0x10000 != 0 { 0xffff_0000 } else { 0 }))
                        .collect::<Vec<_>>()
                );
            }
            device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
        }
    }
    Ok(())
}

#[test]
fn test_metadata_enforces_instruction_versions_in_ordinary_compilation() -> Result {
    for (version, body) in [
        ("7.5", "bmsk.clamp.b32 r0,1,2;"),
        ("7.5", "szext.wrap.u32 r0,7,3;"),
        ("8.1", "lop3.and.b32 r0|p0,1,2,3,7,p1;"),
        ("9.2", "clmad.lo.u64 d0,1,2,3;"),
    ] {
        let source = format!(
            ".version {version} .target sm_80 .address_size 64 .entry k() {{.reg .b32 r0;.reg .b64 d0;.reg .pred p0;.reg .pred p1;{body}ret;}}"
        );
        assert!(
            ParsedModule::parse(&source)?
                .compile("k")
                .unwrap_err()
                .to_string()
                .contains("requires PTX")
        );
    }
    Ok(())
}

#[test]
fn test_new_operations_preserve_carry_and_skip_indeterminate_operands_when_predicated_off() -> Result
{
    for body in [
        "mad.hi.sat.s32 r0,r1,r2,r3",
        "mul24.hi.s32 r0,r1,r2",
        "mad24.hi.sat.s32 r0,r1,r2,r3",
        "sad.s64 d0,d1,d2,d3",
        "clmad.hi.u64 d0,d1,d2,d3",
        "fns.b32 r0,r1,r2,r3",
        "bmsk.wrap.b32 r0,r1,r2",
        "szext.clamp.s32 r0,r1,r2",
        "cnot.b64 d0,d1",
        "lop3.and.b32 r0|p0,r1,r2,r3,7,p1",
        "dp4a.s32.u32 r0,r1,r2,r3",
        "dp2a.hi.u32.s32 r0,r1,r2,r3",
        "cvt.pack.sat.s4.s32.b32 r0,r1,r2,r3",
        "set.eq.u32.s64 r0,d1,d2",
        "set.lt.f16x2.f16x2 r0,r1,r2",
        "setp.eq.f16 p0,h1,h2",
        "setp.gt.f16x2 p0|p1,r1,r2",
        "slct.b32.s32 r0,r1,r2,r3",
        "copysign.f64 d0,d1,d2",
    ] {
        for enabled in [false, true] {
            let setup = if enabled {
                "mov.b32 r1,1;mov.b32 r2,2;mov.b32 r3,3;mov.b64 d1,1;mov.b64 d2,2;mov.b64 d3,3;mov.b16 h1,0x3c00;mov.b16 h2,0x4000;mov.pred p1,1;"
            } else {
                ""
            };
            let source = format!(
                "{PRELUDE} .global .u32 output; .entry k() {{.reg .b32 r<4>;.reg .b64 d<4>;.reg .b16 h<3>;.reg .pred p<4>;{setup}mov.pred p3,{};add.cc.u32 r0,0xffffffff,1;@p3 {body};addc.u32 r0,0,0;st.global.u32 [output],r0;ret;}}",
                u8::from(enabled)
            );
            let device = Emulator::create(EmulatorOptions::default())?;
            let module = device.load(ParsedModule::parse(&source)?)?.wait()?;
            module
                .kernel("k")?
                .launch(LaunchConfig::create([1, 1, 1], [1, 1, 1], 0)?, [])?
                .wait()?;
            assert_eq!(
                device.download(&module.global("output")?)?.wait()?,
                1u32.to_le_bytes(),
                "{body}, enabled={enabled}"
            );
            device.shutdown(ShutdownMode::Drain).wait()?;
        }
    }
    Ok(())
}

#[test]
fn test_comparisons_use_source_precision_and_selection_preserves_payload_bits() -> Result {
    // Each unequal source pair would become equal if first rounded to binary16.
    // The final selector is negative zero, followed by negative subnormal and NaN.
    let source = format!(
        r"{PRELUDE} .global .align 16 .b8 output[32];
.entry k() {{.reg .b16 h; .reg .b32 r; .reg .b32 s; .reg .b64 a; .reg .b64 b;
 mov.b64 a,0x3ff0000000000001;mov.b64 b,0x3ff0000000000000;
 set.gt.f16.f64 h,a,b;st.global.b16 [output],h;
 set.gt.f16.u64 h,9007199254740993,9007199254740992;st.global.b16 [output+2],h;
 mov.b32 r,0x3f800001;mov.b32 s,0x3f800000;
 set.gt.f16.f32 h,r,s;st.global.b16 [output+4],h;
 mov.b32 s,0x80000000;slct.b32.f32 r,0xffc12345,0x80000000,s;st.global.b32 [output+8],r;
 mov.b32 s,0x80000001;slct.b32.f32 r,0xffc12345,0x80000000,s;st.global.b32 [output+12],r;
 slct.ftz.b32.f32 r,0xffc12345,0x80000000,s;st.global.b32 [output+16],r;
 mov.b32 s,0xffa12345;slct.b32.f32 r,0xffc12345,0x80000000,s;st.global.b32 [output+20],r;
 copysign.f32 r,s,r;st.global.b32 [output+24],r;
 mov.b32 s,0;copysign.f32 r,s,0fFFC12345;st.global.b32 [output+28],r;ret;
}}"
    );
    let device = Emulator::create(EmulatorOptions::default())?;
    let module = device.load(ParsedModule::parse(&source)?)?.wait()?;
    module
        .kernel("k")?
        .launch(LaunchConfig::create([1, 1, 1], [1, 1, 1], 0)?, [])?
        .wait()?;
    let mut expected = vec![0u8; 32];
    for offset in [0, 2, 4] {
        expected[offset..offset + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
    }
    for (offset, value) in [
        (8, 0xffc1_2345_u32),
        (12, 0x8000_0000),
        (16, 0xffc1_2345),
        (20, 0x8000_0000),
        (24, 0x8000_0000),
        (28, 0x7fc1_2345),
    ] {
        expected[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    assert_eq!(
        device.download(&module.global("output")?)?.wait()?,
        expected
    );
    device.shutdown(ShutdownMode::Drain).wait()?;
    Ok(())
}

#[test]
fn test_out_of_range_fns_fault_keeps_completed_quantized_writes_inspectable() -> Result {
    let device = Emulator::create(EmulatorOptions::default())?;
    let source = format!(
        "{PRELUDE} .global .u32 output;.entry k() {{.reg .b32 r;dp4a.s32.u32 r,0x01010101,0x02020202,0;st.global.u32 [output],r;fns.b32 r,0xffffffff,32,1;ret;}}"
    );
    let module = device.load(ParsedModule::parse(&source)?)?.wait()?;
    let error = module
        .kernel("k")?
        .launch(LaunchConfig::create([1, 1, 1], [1, 1, 1], 0)?, [])?
        .wait()
        .unwrap_err();
    assert!(error.to_string().contains("fns base"));
    let output = module.global("output")?;
    assert_eq!(device.download(&output)?.wait()?, 8u32.to_le_bytes());
    assert!(device.write(&output, &[0u8; 4]).is_err());
    assert!(device.shutdown(ShutdownMode::Drain).wait().is_err());
    Ok(())
}
