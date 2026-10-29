//! Address-sized parameters, relocations, and live caller-frame pointers.

use std::{error::Error, result::Result as StdResult};

use singe_ptx::emulator::{Emulator, EmulatorOptions, LaunchConfig, ParsedModule, ShutdownMode};

type Result = StdResult<(), Box<dyn Error>>;

#[test]
fn test_address_sized_pointer_arguments_execute_odd_multiblock_kernels() -> Result {
    for address_bits in [32, 64] {
        for annotated in [false, true] {
            for count in [1u32, 31, 32, 33, 129] {
                let pointer = if annotated {
                    ".ptr .global .align 4"
                } else {
                    ""
                };
                let source = format!(
                    ".version 9.3 .target sm_80 .address_size {address_bits}
                    .entry k(.param .u{address_bits} {pointer} output,.param .u32 count) {{
                        .reg .u32 index; .reg .u32 block; .reg .u32 bound; .reg .u32 result;
                        .reg .u{address_bits} address; .reg .u{address_bits} offset;
                        .reg .pred done;
                        mov.u32 index,%tid.x; mov.u32 block,%ctaid.x;
                        mad.lo.u32 index,block,32,index;
                        ld.param.u32 bound,[count]; setp.ge.u32 done,index,bound;
                        @done ret;
                        ld.param.u{address_bits} address,[output];
                        cvt.u{address_bits}.u32 offset,index;
                        mul.lo.u{address_bits} offset,offset,4;
                        add.u{address_bits} address,address,offset;
                        mad.lo.u32 result,index,7,13;
                        st.global.u32 [address],result; ret;
                    }}"
                );
                let device = Emulator::create(EmulatorOptions::default())?;
                let module = device.load(ParsedModule::parse(&source)?)?.wait()?;
                let output = device.zeroed::<u32>(count as usize)?.wait()?;
                module
                    .kernel("k")?
                    .launch(
                        LaunchConfig::create([count.div_ceil(32), 1, 1], [32, 1, 1], 0)?,
                        [(&output).into(), count.into()],
                    )?
                    .wait()?;
                assert_eq!(
                    device.download(&output)?.wait()?,
                    (0..count).map(|index| index * 7 + 13).collect::<Vec<_>>()
                );
                device.shutdown(ShutdownMode::Drain).wait()?;
            }
        }
    }
    Ok(())
}

#[test]
fn test_relocations_and_callee_access_to_live_caller_storage_work_in_both_address_sizes() -> Result
{
    for bits in [32, 64] {
        let source = format!(
            ".version 9.3 .target sm_80 .address_size {bits}
            .const .u32 constant = 29;
            .global .u{bits} relocated = generic(constant);
            .global .u32 output;
            .func (.param .u32 result) helper(.param .u{bits} p) {{
                .reg .u{bits} address; .reg .u32 value;
                ld.param.u{bits} address,[p]; ld.u32 value,[address];
                add.u32 value,value,1; st.param.u32 [result],value; ret;
            }}
            .entry k() {{
                .local .u32 scratch; .param .u{bits} argument; .param .u32 result;
                .reg .u{bits} address; .reg .u32 value;
                ld.global.u{bits} address,[relocated]; ld.u32 value,[address];
                st.local.u32 [scratch],value;
                mov.u{bits} address,scratch; cvta.local.u{bits} address,address;
                st.param.u{bits} [argument],address;
                call (result),helper,(argument); ld.param.u32 value,[result];
                st.global.u32 [output],value; ret;
            }}"
        );
        let device = Emulator::create(EmulatorOptions::default())?;
        let module = device.load(ParsedModule::parse(&source)?)?.wait()?;
        module
            .kernel("k")?
            .launch(LaunchConfig::create([1, 1, 1], [1, 1, 1], 0)?, [])?
            .wait()?;
        assert_eq!(
            device.download(&module.global("output")?)?.wait()?,
            30u32.to_le_bytes()
        );
        device.shutdown(ShutdownMode::Drain).wait()?;
    }
    Ok(())
}

#[test]
fn test_pointer_annotations_must_match_the_module_address_size() -> Result {
    for (module_bits, parameter_bits) in [(32, 64), (64, 32)] {
        let module = ParsedModule::parse(&format!(
            ".version 9.3 .target sm_80 .address_size {module_bits} .entry k(.param .u{parameter_bits} .ptr p) {{ ret; }}"
        ))?;
        assert!(module.compile("k").is_err());
    }
    Ok(())
}
