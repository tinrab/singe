//! Warp counts and contended weighted sums, checked against host references.

use anyhow::Result;
use singe_ptx::emulator::{
    Emulator, EmulatorOptions, ExecutionError, LaunchConfig, ParsedModule, ShutdownMode,
};

mod support;

use support::Observe;

fn collective_source(body: &str) -> String {
    format!(
        ".version 9.4 .target sm_80 .address_size 64
         .global .u32 output[32];
         .entry k() {{
             .reg .u32 t; .reg .u32 r; .reg .u32 b;
             .reg .u64 a; .reg .u64 base; .reg .pred p; .reg .pred q;
             mov.u32 t, %tid.x;
             {body}
             mul.wide.u32 a, t, 4;
             mov.u64 base, output; add.u64 a, a, base;
             st.global.u32 [a], r; ret;
         }}"
    )
}

#[test]
fn test_redux_snapshots_wrapping_sums_and_matches_live_participants_across_positions() -> Result<()>
{
    let cases = [
        (
            "redux.sync.add.u32 r, t, 0xffffffff; redux.sync.add.u32 r, r, 0xffffffff;",
            vec![496 * 32; 32],
        ),
        (
            "redux.sync.add.u32 r, 0xffffffff, 0xffffffff;",
            vec![0xffff_ffe0; 32],
        ),
        (
            "setp.lt.u32 p,t,16; @p bra left;
             mov.u32 r,2; redux.sync.add.u32 r,r,0xffffffff; bra joined;
             left: mov.u32 r,1; redux.sync.add.u32 r,r,0xffffffff;
             joined:",
            vec![48; 32],
        ),
        (
            "setp.ge.u32 p,t,17; @p ret; redux.sync.add.u32 r,t,0xffffffff;",
            (0..32)
                .map(|lane| if lane < 17 { 136 } else { 0 })
                .collect(),
        ),
        (
            "mov.u32 r,77; mov.pred q,0; @q redux.sync.add.u32 r,b,b;
             setp.lt.u32 p,t,17; @p redux.sync.add.u32 r,t,0x1ffff;",
            (0..32)
                .map(|lane| if lane < 17 { 136 } else { 77 })
                .collect(),
        ),
        (
            "setp.lt.u32 p,t,16; selp.b32 b,0xffff,0xffff0000,p;
             redux.sync.add.u32 r,t,b; redux.sync.add.u32 r,r,0xffffffff;",
            vec![7936; 32],
        ),
    ];
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        for (body, expected) in &cases {
            let parsed = ParsedModule::parse(&collective_source(body))?;
            let module = device.load(parsed)?.observe(asynchronous)?;
            module
                .kernel("k")?
                .launch(LaunchConfig::create([1; 3], [32, 1, 1], 0)?, [])?
                .observe(asynchronous)?;
            let actual = device
                .download(&module.global("output")?)?
                .observe(asynchronous)?;
            let expected: Vec<_> = expected
                .iter()
                .flat_map(|value| u32::to_le_bytes(*value))
                .collect();
            assert_eq!(actual, expected, "{body}, async={asynchronous}");
        }
        device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
    }
    Ok(())
}

#[test]
fn test_invalid_redux_membership_and_incompatible_rendezvous_preserve_prior_writes() -> Result<()> {
    for asynchronous in [false, true] {
        for (body, stalls) in [
            ("redux.sync.add.u32 r,1,0;", false),
            (
                "setp.lt.u32 p,t,16; selp.b32 b,0x7fffffff,0xffffffff,p;
                 redux.sync.add.u32 r,1,b;",
                true,
            ),
            (
                "setp.lt.u32 p,t,16; @p bra reduce;
                 vote.sync.ballot.b32 r,p,0xffffffff; bra joined;
                 reduce: redux.sync.add.u32 r,1,0xffffffff;
                 joined:",
                true,
            ),
        ] {
            let body = format!(
                "mul.wide.u32 a,t,4; mov.u64 base,output; add.u64 a,a,base;
                 st.global.u32 [a],42; {body}"
            );
            let device = Emulator::create(EmulatorOptions::default())?;
            let module = device
                .load(ParsedModule::parse(&collective_source(&body))?)?
                .observe(asynchronous)?;
            let error = module
                .kernel("k")?
                .launch(LaunchConfig::create([1; 3], [32, 1, 1], 0)?, [])?
                .observe(asynchronous)
                .unwrap_err();
            if stalls {
                assert!(
                    matches!(error.root_cause(), ExecutionError::Stall(_)),
                    "{error}"
                );
            } else {
                assert!(
                    matches!(error.root_cause(), ExecutionError::Invalid(_)),
                    "{error}"
                );
            }
            assert!(error.launch_report().is_some());
            let actual = device
                .download(&module.global("output")?)?
                .observe(asynchronous)?;
            assert_eq!(&actual[..4], &42u32.to_le_bytes());
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
fn test_global_float_atomic_add_rounds_flushes_and_returns_original_bits() -> Result<()> {
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        for (initial, addend, expected) in [
            (0x3f80_0000_u32, 0x4000_0000_u32, Some(0x4040_0000_u32)),
            (0x3f80_0000, 0x3380_0000, Some(0x3f80_0000)),
            (0x3f80_0000, 0x3440_0000, Some(0x3f80_0002)),
            (0x7f7f_ffff, 0x7f7f_ffff, Some(0x7f80_0000)),
            (0x7f80_0000, 0xff80_0000, None),
            (0x7fa0_0001, 0x3f80_0000, None),
            (0x8000_0000, 0x8000_0000, Some(0x8000_0000)),
            (0x8000_0000, 0, Some(0)),
            (1, 1, Some(0)),
            (0x8000_0001, 0x8000_0000, Some(0x8000_0000)),
            (0x0080_0001, 0x8080_0000, Some(0)),
            (0x0080_0000, 0x8080_0001, Some(0x8000_0000)),
        ] {
            let source = format!(
                ".version 9.4 .target sm_80 .address_size 64
                 .global .u32 cell = {initial}; .global .u32 previous;
                 .entry k() {{
                     .reg .b32 old; .reg .b32 value;
                     mov.b32 value,{addend};
                     atom.global.add.f32 old,[cell],value;
                     st.global.b32 [previous],old; ret;
                 }}"
            );
            let module = device
                .load(ParsedModule::parse(&source)?)?
                .observe(asynchronous)?;
            module
                .kernel("k")?
                .launch(LaunchConfig::create([1; 3], [1; 3], 0)?, [])?
                .observe(asynchronous)?;
            assert_eq!(
                device
                    .download(&module.global("previous")?)?
                    .observe(asynchronous)?,
                initial.to_le_bytes()
            );
            let actual = device
                .download(&module.global("cell")?)?
                .observe(asynchronous)?;
            if let Some(expected) = expected {
                assert_eq!(
                    actual,
                    expected.to_le_bytes(),
                    "{initial:08x} + {addend:08x}"
                );
            } else {
                let bytes: [u8; 4] = actual.try_into().unwrap();
                assert!(f32::from_le_bytes(bytes).is_nan());
            }
        }
        device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
    }
    Ok(())
}

#[test]
fn test_contended_float_atomics_return_each_preceding_sum_and_skip_predicated_inputs() -> Result<()>
{
    let source = ".version 9.4 .target sm_80 .address_size 64
        .global .f32 total = 0f00000000; .global .u32 previous[32];
        .entry k() {
            .reg .u32 lane; .reg .u32 uninitialized;
            .reg .u64 address; .reg .u64 offset; .reg .u64 uninitialized_address;
            .reg .f32 old; .reg .f32 uninitialized_float; .reg .pred skip;
            mov.pred skip,0;
            @skip atom.global.add.f32 old,[uninitialized_address],uninitialized_float;
            @skip red.global.add.u32 [uninitialized_address],uninitialized;
            mov.u32 lane,%tid.x;
            atom.global.add.f32 old,[total],0f3F800000;
            mov.u64 address,previous; mul.wide.u32 offset,lane,4;
            add.u64 address,address,offset; st.global.f32 [address],old;
            ret;
        }";
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let module = device
            .load(ParsedModule::parse(source)?)?
            .observe(asynchronous)?;
        module
            .kernel("k")?
            .launch(LaunchConfig::create([1; 3], [32, 1, 1], 0)?, [])?
            .observe(asynchronous)?;
        let bytes = device
            .download(&module.global("previous")?)?
            .observe(asynchronous)?;
        let mut actual: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| u32::from_le_bytes(*bytes))
            .collect();
        actual.sort_unstable();
        let expected: Vec<_> = (0..32).map(|value| (value as f32).to_bits()).collect();
        assert_eq!(actual, expected);
        assert_eq!(
            device
                .download(&module.global("total")?)?
                .observe(asynchronous)?,
            32f32.to_bits().to_le_bytes()
        );
        device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
    }
    Ok(())
}

#[test]
fn test_reduction_wraps_and_atomic_address_faults_preserve_completed_updates() -> Result<()> {
    for asynchronous in [false, true] {
        for offset in [1, 4] {
            let source = format!(
                ".version 9.4 .target sm_80 .address_size 64
                 .global .u32 count = 0xffffffff; .global .u32 weight;
                 .entry k() {{
                     .reg .f32 old;
                     red.global.add.u32 [count],1;
                     atom.global.add.f32 _,[weight],0f42280000;
                     atom.global.add.f32 old,[weight+{offset}],0f3F800000;
                     ret;
                 }}"
            );
            let device = Emulator::create(EmulatorOptions::default())?;
            let module = device
                .load(ParsedModule::parse(&source)?)?
                .observe(asynchronous)?;
            let error = module
                .kernel("k")?
                .launch(LaunchConfig::create([1; 3], [1; 3], 0)?, [])?
                .observe(asynchronous)
                .unwrap_err();
            assert!(
                matches!(error.root_cause(), ExecutionError::Memory(_)),
                "{error}"
            );
            assert_eq!(
                device
                    .download(&module.global("count")?)?
                    .observe(asynchronous)?,
                0u32.to_le_bytes()
            );
            assert_eq!(
                device
                    .download(&module.global("weight")?)?
                    .observe(asynchronous)?,
                42f32.to_bits().to_le_bytes()
            );
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
fn test_weighted_reduction_counts_partial_warps_and_sums_contended_weights() -> Result<()> {
    let parsed = ParsedModule::parse(include_str!("fixtures/execution/weighted_reduce.ptx"))?;
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let module = device.load(parsed.clone())?.observe(asynchronous)?;
        let kernel = module.kernel("weighted_reduce")?;
        let count_output = module.global("total_count")?;
        let weight_output = module.global("total_weight")?;
        for block in [1u32, 17, 32, 33, 64, 127, 128] {
            for count in [0u32, 1, 17, 31, 32, 33, 127, 128, 129, 257] {
                // Small multiples of 1/4 have an exact sum in any issue order.
                let values: Vec<f32> = (0..count)
                    .map(|index| ((index * 37 % 101).cast_signed() - 50) as f32 * 0.25)
                    .collect();
                let expected = values.iter().fold(0.0f32, |sum, value| sum + value);
                let input = device.upload(&values)?.observe(asynchronous)?;
                device
                    .write(&count_output, &[0; 4])?
                    .observe(asynchronous)?;
                device
                    .write(&weight_output, &[0; 4])?
                    .observe(asynchronous)?;
                kernel
                    .launch(
                        LaunchConfig::create(
                            [count.max(1).div_ceil(block), 1, 1],
                            [block, 1, 1],
                            0,
                        )?,
                        [(&input).into(), count.into()],
                    )?
                    .observe(asynchronous)?;
                assert_eq!(
                    device.download(&count_output)?.observe(asynchronous)?,
                    count.to_le_bytes(),
                    "block={block}, count={count}, async={asynchronous}"
                );
                assert_eq!(
                    device.download(&weight_output)?.observe(asynchronous)?,
                    expected.to_bits().to_le_bytes(),
                    "block={block}, count={count}, async={asynchronous}"
                );
            }
        }
        device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
    }
    Ok(())
}
