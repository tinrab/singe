//! Check special-register values and module directive validation.
mod support;

use std::{array, collections::BTreeSet};

use anyhow::{Result, ensure};
use serde::Deserialize;
use singe_ptx::emulator::{Emulator, EmulatorOptions, LaunchConfig, ParsedModule, ShutdownMode};
use support::Observe;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RegisterCase {
    head: String,
    shape: String,
    conversion: bool,
    grid: [u32; 3],
    block: [u32; 3],
    source: String,
    expected: Vec<u64>,
}

#[test]
fn test_registers_and_directives_execute_and_validate() -> Result<()> {
    let cases: Vec<RegisterCase> =
        serde_json::from_str(include_str!("fixtures/execution/environment.json"))?;
    let mut names = BTreeSet::new();
    for case in cases {
        ensure!(
            names.insert((case.head.clone(), case.shape.clone(), case.conversion)),
            "duplicate register case"
        );
        for asynchronous in [false, true] {
            let device = Emulator::create(EmulatorOptions::default())?;
            let module = device
                .load(ParsedModule::parse(&case.source)?)?
                .observe(asynchronous)?;
            // A prior accepted launch makes the tested grid id nonzero.
            let empty = device
                .load(ParsedModule::parse(
                    ".version 9.3 .target sm_80 .address_size 64 .entry empty() {ret;}",
                )?)?
                .observe(asynchronous)?;
            empty
                .kernel("empty")?
                .launch(LaunchConfig::create([1; 3], [1; 3], 0)?, [])?
                .observe(asynchronous)?;
            let output = device
                .zeroed::<u64>(case.expected.len())?
                .observe(asynchronous)?;
            module
                .kernel("kernel")?
                .launch(
                    LaunchConfig::create(case.grid, case.block, 0)?,
                    [(&output).into()],
                )?
                .observe(asynchronous)?;
            ensure!(
                device.download(&output)?.observe(asynchronous)? == case.expected,
                "incorrect register {} ({})",
                case.head,
                case.shape
            );
            device.shutdown(ShutdownMode::Drain).wait()?;
        }
    }
    directives()
}

fn directives() -> Result<()> {
    let compile =
        |source: &str| -> Result<_> { Ok(ParsedModule::parse(source)?.compile("kernel")?) };
    for (major, last) in [(7, 8), (8, 8), (9, 4)] {
        for minor in 0..=last {
            compile(&format!(
                ".version {major}.{minor} .target sm_80 .address_size 64 .entry kernel() {{ret;}}"
            ))?;
        }
    }
    for version in ["6.9", "7.9", "8.9", "9.5"] {
        ensure!(
            compile(&format!(
                ".version {version} .target sm_80 .entry kernel() {{ret;}}"
            ))
            .is_err(),
            "invalid module version accepted"
        );
    }
    compile(".version 9.3 .target sm_80 .entry kernel() {ret;}")?;
    for target in ["sm_75", "sm_90"] {
        ensure!(
            compile(&format!(
                ".version 9.3 .target {target} .entry kernel() {{ret;}}"
            ))
            .is_err(),
            "foreign target accepted"
        );
    }
    for bits in [32, 64] {
        let code = compile(&format!(
            ".version 9.3 .target sm_80 .address_size {bits}
            .entry kernel(.param .u{bits} .ptr.global input) {{ret;}}"
        ))?;
        ensure!(
            code.parameters()[0].storage().byte_len() == bits / 8,
            "address width did not set parameter layout"
        );
        ensure!(
            compile(&format!(
                ".version 9.3 .target sm_80 .address_size {bits}
            .entry kernel(.param .u{} .ptr.global input) {{ret;}}",
                96 - bits
            ))
            .is_err(),
            "pointer width mismatch accepted"
        );
    }
    for dimensions in [vec![4], vec![2, 2], vec![2, 1, 2]] {
        let text = dimensions
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        for directive in [".maxntid", ".reqntid"] {
            let code = compile(&format!(
                ".version 9.3 .target sm_80 .entry kernel() {directive} {text} {{ret;}}"
            ))?;
            let mut block = [1; 3];
            block[..dimensions.len()].copy_from_slice(&dimensions);
            code.launch_requirements().validate_block(block)?;
            ensure!(
                code.launch_requirements()
                    .validate_block([8, 1, 1])
                    .is_err(),
                "thread directive did not constrain launch"
            );
            if directive == ".maxntid" {
                code.launch_requirements().validate_block([1; 3])?;
                ensure!(
                    code.launch_requirements().maximum_threads() == Some(4),
                    "maximum thread count differs"
                );
            } else {
                ensure!(
                    code.launch_requirements().required_block() == Some(block),
                    "required block differs"
                );
                ensure!(
                    code.launch_requirements().validate_block([1; 3]).is_err(),
                    "required shape was ignored"
                );
            }
        }
    }
    for directive in [".maxnreg", ".minnctapersm"] {
        compile(&format!(
            ".version 9.3 .target sm_80 .entry kernel() {directive} 1 {{ret;}}"
        ))?;
        ensure!(
            compile(&format!(
                ".version 9.3 .target sm_80 .entry kernel() {directive} 0 {{ret;}}"
            ))
            .is_err(),
            "zero resource hint accepted"
        );
    }
    Ok(())
}

#[test]
fn test_legacy_widths_are_limited_to_the_defined_registers_and_instructions() -> Result<()> {
    for instruction in [
        "mov.u16 h,%laneid",
        "mov.u64 d,%smid",
        "cvt.u32.f32 r,%gridid",
        "mov.f32 f,%gridid",
        "st.global.u16 [output],%tid.x",
    ] {
        let source = format!(".version 9.3 .target sm_80 .address_size 64 .global .u16 output;
            .entry k() {{ .reg .u16 h; .reg .u32 r; .reg .u64 d; .reg .f32 f; {instruction}; ret; }}");
        assert!(
            ParsedModule::parse(&source)?.compile("k").is_err(),
            "{instruction}"
        );
    }
    Ok(())
}

#[test]
fn test_special_registers_require_move_or_conversion_instructions() -> Result<()> {
    for instruction in [
        "add.u32 r,%laneid,1",
        "st.global.u32 [output],%laneid",
        "selp.u32 r,%laneid,1,p",
        "bar.sync %laneid",
        "shfl.sync.idx.b32 r,%laneid,0,31,-1",
        "cvta.to.global.u32 r,%laneid",
    ] {
        let text = format!(
            ".version 9.4 .target sm_80 .address_size 64 .global .u32 output;
            .entry k() {{ .reg .u32 r; .reg .pred p; {instruction}; ret; }}"
        );
        let error = ParsedModule::parse(&text)?
            .compile("k")
            .expect_err(instruction);
        assert!(
            error
                .to_string()
                .contains("special registers require mov or cvt")
        );
        assert_eq!(&text[error.span.start..error.span.end], "%laneid");
    }
    Ok(())
}

#[test]
fn test_declared_registers_shadow_special_registers_in_their_lexical_scope() -> Result<()> {
    let text = ".version 9.4 .target sm_80 .address_size 64
        .global .u32 output[160];
        .entry k() { .reg .u32 t; .reg .u32 value; .reg .u16 half_value;
            .reg .u64 a; .reg .u64 base;
            mov.u32 t,%tid.x; mul.wide.u32 a,t,20; mov.u64 base,output; add.u64 a,a,base;
            mov.u32 value,%laneid; st.global.u32 [a],value;
            { .reg .u32 %laneid; .reg .u16 %gridid;
                mov.u32 %laneid,10; add.u32 value,%laneid,1; st.global.u32 [a+4],value;
                mov.u16 %gridid,4660; mov.u16 half_value,%gridid;
                cvt.u32.u16 value,half_value; st.global.u32 [a+8],value;
                cvt.u32.u16 value,%gridid; st.global.u32 [a+12],value;
            }
            mov.u32 value,%laneid; st.global.u32 [a+16],value;
            ret;
        }";
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let module = device
            .load(ParsedModule::parse(text)?)?
            .observe(asynchronous)?;
        module
            .kernel("k")?
            .launch(LaunchConfig::create([1; 3], [32, 1, 1], 0)?, [])?
            .observe(asynchronous)?;
        let actual = device
            .download(&module.global("output")?)?
            .observe(asynchronous)?;
        assert_eq!(actual.len(), 32 * 20);
        for (lane, bytes) in actual.as_chunks::<20>().0.iter().enumerate() {
            let values: [u32; 5] =
                array::from_fn(|index| u32::from_le_bytes(bytes.as_chunks::<4>().0[index]));
            assert_eq!(
                values,
                [
                    u32::try_from(lane).unwrap(),
                    11,
                    4660,
                    4660,
                    u32::try_from(lane).unwrap()
                ],
                "lane {lane}"
            );
        }
        device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
    }
    Ok(())
}

#[test]
fn test_percent_prefixed_memory_symbols_can_supply_addresses() -> Result<()> {
    let parsed = ParsedModule::parse(".version 9.4 .target sm_80 .address_size 64
        .global .u32 %data = 42; .global .u32 output;
        .entry k() { .reg .u64 pointer; .reg .u32 value;
            mov.u64 pointer,%data; ld.global.u32 value,[pointer]; st.global.u32 [output],value; ret; }")?;
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let module = device.load(parsed.clone())?.observe(asynchronous)?;
        module
            .kernel("k")?
            .launch(LaunchConfig::create([1; 3], [1; 3], 0)?, [])?
            .observe(asynchronous)?;
        assert_eq!(
            device
                .download(&module.global("output")?)?
                .observe(asynchronous)?,
            42u32.to_le_bytes()
        );
        device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
    }
    Ok(())
}
