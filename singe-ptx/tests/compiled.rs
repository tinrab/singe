use std::{error::Error, result::Result as StdResult};

use singe_ptx::emulator::{
    CompileErrorKind, Emulator, EmulatorOptions, LaunchConfig, ParsedModule, ShutdownMode,
};

const SOURCE: &str = ".version 9.4 .target sm_80 .address_size 64
.global .u32 counter = 0;
.entry increment() {
 .reg .u32 %value;
 ld.global.u32 %value, [counter];
 add.u32 %value, %value, 1;
 st.global.u32 [counter], %value;
 ret;
}";

type Result = StdResult<(), Box<dyn Error>>;

#[test]
fn test_compiled_kernel_binds_to_independent_emulators_and_module_globals() -> Result {
    let source = ParsedModule::parse(SOURCE)?;
    let compiled = source.compile("increment")?;
    let first = Emulator::create(EmulatorOptions::default())?;
    let second = Emulator::create(EmulatorOptions::default())?;
    let a = first.load(source.clone())?.wait()?;
    let b = second.load(source.clone())?.wait()?;
    let c = first.load(source)?.wait()?;
    let config = LaunchConfig::create([1; 3], [1; 3], 0)?;
    a.bind(&compiled)?.launch(config, [])?.wait()?;
    for _ in 0..2 {
        b.bind(&compiled)?.launch(config, [])?.wait()?;
    }
    assert_eq!(first.download(&a.global("counter")?)?.wait()?, [1, 0, 0, 0]);
    assert_eq!(
        second.download(&b.global("counter")?)?.wait()?,
        [2, 0, 0, 0]
    );
    assert_eq!(first.download(&c.global("counter")?)?.wait()?, [0; 4]);
    first.shutdown(ShutdownMode::Drain).wait()?;
    second.shutdown(ShutdownMode::Drain).wait()?;
    Ok(())
}

#[test]
fn test_separately_parsed_identical_sources_cannot_bind() -> Result {
    let compiled = ParsedModule::parse(SOURCE)?.compile("increment")?;
    let device = Emulator::create(EmulatorOptions::default())?;
    let module = device.load(ParsedModule::parse(SOURCE)?)?.wait()?;
    let error = module.bind(&compiled).unwrap_err();
    assert_eq!(error.kind, CompileErrorKind::ModuleMismatch);
    assert_eq!(
        &compiled.source().source()[error.span.start..error.span.end],
        SOURCE
    );
    module.kernel("increment")?;
    device.shutdown(ShutdownMode::Drain).wait()?;
    Ok(())
}
