//! Decimal and rational expectations generated independently from the runtime math.

use std::error::Error;

use serde::Deserialize;
use singe_ptx::emulator::{
    Emulator, EmulatorOptions, ExecutionError, LaunchConfig, ParsedModule, ShutdownMode,
};

mod support;

use support::Observe;

#[derive(Deserialize)]
struct Oracle {
    head: String,
    bits: u32,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    input: u64,
    expected: u64,
    comparison: String,
    reference: Option<String>,
}

fn value(bits: u64, width: u32) -> f64 {
    if width == 32 {
        f64::from(f32::from_bits(u32::try_from(bits & 0xffff_ffff).unwrap()))
    } else {
        f64::from_bits(bits)
    }
}

#[test]
fn test_exponential_preserves_carry_and_predication_and_diagnoses_uninitialized_inputs()
-> Result<(), Box<dyn Error>> {
    let parsed = ParsedModule::parse(
        ".version 9.4 .target sm_80 .address_size 64
         .global .u32 output[2];
         .entry k() {
             .reg .u32 carry; .reg .f32 value; .reg .pred skip;
             mov.pred skip, 0;
             add.cc.u32 carry, 0xffffffff, 1;
             @skip ex2.approx.f32 value, value;
             mov.f32 value, 0f3F800000;
             ex2.approx.f32 value, value;
             addc.u32 carry, 0, 0;
             st.global.u32 [output], carry;
             st.global.f32 [output+4], value;
             ret;
         }
         .entry uninitialized() {
             .reg .f32 value;
             ex2.approx.ftz.f32 value, value;
             ret;
         }",
    )?;
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let module = device.load(parsed.clone())?.observe(asynchronous)?;
        let config = LaunchConfig::create([1; 3], [1; 3], 0)?;
        module
            .kernel("k")?
            .launch(config, [])?
            .observe(asynchronous)?;
        let expected = [1u32.to_le_bytes(), 2f32.to_bits().to_le_bytes()].concat();
        assert_eq!(
            device
                .download(&module.global("output")?)?
                .observe(asynchronous)?,
            expected
        );
        let error = module
            .kernel("uninitialized")?
            .launch(config, [])?
            .observe(asynchronous)
            .unwrap_err();
        assert!(matches!(
            error.root_cause(),
            ExecutionError::Indeterminate(_)
        ));
        assert_eq!(
            device
                .download(&module.global("output")?)?
                .observe(asynchronous)?,
            expected
        );
        assert!(
            device
                .shutdown(ShutdownMode::Drain)
                .observe(asynchronous)
                .is_err()
        );
    }
    Ok(())
}

#[test]
fn test_scalar_math_matches_independent_oracles_through_both_waiting_styles()
-> Result<(), Box<dyn Error>> {
    let oracles: Vec<Oracle> =
        serde_json::from_str(include_str!("fixtures/execution/math-oracles.json"))?;
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        for oracle in &oracles {
            let head = &oracle.head;
            let bits = oracle.bits;
            let source = format!(
                ".version 9.3 .target sm_80 .address_size 64
                .entry k(.param .u64 input,.param .u64 output,.param .u32 count) {{
                    .reg .u32 index; .reg .u32 block; .reg .u32 bound; .reg .pred done;
                    .reg .u64 address; .reg .u64 offset; .reg .f{bits} r;
                    mov.u32 index,%tid.x; mov.u32 block,%ctaid.x; mad.lo.u32 index,block,32,index;
                    ld.param.u32 bound,[count]; setp.ge.u32 done,index,bound; @done ret;
                    mul.wide.u32 offset,index,8;
                    ld.param.u64 address,[input]; add.u64 address,address,offset;
                    ld.global.b{bits} r,[address]; {head} r,r;
                    ld.param.u64 address,[output]; add.u64 address,address,offset;
                    st.global.b{bits} [address],r; ret;
                }}"
            );
            let module = device
                .load(ParsedModule::parse(&source)?)?
                .observe(asynchronous)?;
            let inputs: Vec<_> = oracle.cases.iter().map(|case| case.input).collect();
            let input = device.upload(&inputs)?.observe(asynchronous)?;
            let output = device.zeroed::<u64>(inputs.len())?.observe(asynchronous)?;
            let count = u32::try_from(inputs.len()).unwrap();
            module
                .kernel("k")?
                .launch(
                    LaunchConfig::create([count.div_ceil(32), 1, 1], [32, 1, 1], 0)?,
                    [(&input).into(), (&output).into(), count.into()],
                )?
                .observe(asynchronous)?;
            let actual = device.download(&output)?.observe(asynchronous)?;
            for (case, actual) in oracle.cases.iter().zip(actual) {
                let observed = value(actual, bits);
                let label = format!("{head} input={:016x} actual={actual:016x}", case.input);
                match case.comparison.as_str() {
                    "exact" => assert_eq!(actual, case.expected, "{label}"),
                    "nan" => {
                        assert!(observed.is_nan(), "{label}");
                        if case.expected == 0x7fff_ffff_0000_0000 {
                            assert_eq!(actual, case.expected, "{label}");
                        }
                    }
                    "ulp1" => {
                        let reference: f64 = case.reference.as_ref().unwrap().parse()?;
                        let magnitude = case.expected & !(1 << (bits - 1));
                        let spacing = value(magnitude + 1, bits) - value(magnitude, bits);
                        let slack = reference.abs() * f64::EPSILON * 4.0;
                        assert!(
                            (observed - reference).abs() <= spacing + slack,
                            "{label}, reference={reference}"
                        );
                    }
                    "ulp2" => assert!(actual.abs_diff(case.expected) <= 2, "{label}"),
                    "relative_sqrt" | "relative_rsqrt" => {
                        let reference: f64 = case.reference.as_ref().unwrap().parse()?;
                        let bound = if case.comparison == "relative_sqrt" {
                            2f64.powi(-23)
                        } else {
                            (-22.9f64).exp2()
                        };
                        assert!(
                            ((observed - reference) / reference).abs() <= bound,
                            "{label}, reference={reference}"
                        );
                    }
                    other => panic!("unknown oracle comparison {other}"),
                }
            }
        }
        device.shutdown(ShutdownMode::Drain).wait()?;
    }
    Ok(())
}
