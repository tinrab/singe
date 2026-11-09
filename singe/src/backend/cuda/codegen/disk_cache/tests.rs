use std::{fs, path::PathBuf, process};

use singe_codegen::emit::{EntryPoint, KernelSource, SourceLanguage, SourceParameter};

use crate::{
    backend::cuda::{
        self,
        codegen::{
            CompileRequirements, KernelAbi, KernelArgument, KernelArtifact, KernelCode,
            KernelDescription, KernelLaunch, KernelOutput,
            disk_cache::{CubinOrigin, cubin},
            kernel_compile_options,
        },
        runtime::runtime_for_device,
    },
    device::Device,
    graph::ComputePrecision,
    testing::{maybe_lock_device, skip_if_stub},
};

#[test]
fn test_kernel_cache_loads_cubins_it_wrote_and_recompiles_damaged_entries() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some(runtime) = skip_if_stub(runtime_for_device(Device::Cuda(0))) else {
        return;
    };
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/test-kernel-cache")
        .join(process::id().to_string());
    let _ = fs::remove_dir_all(&directory);
    cuda::enable_kernel_cache(&directory).unwrap();
    // The process id keeps the source, and so the entry, unique to this run.
    let name = "singe_cache_probe";
    let kernel = KernelArtifact {
        name: name.to_owned(),
        code: KernelCode::Source(KernelSource::new(
            SourceLanguage::CudaC,
            "",
            "",
            EntryPoint::new(
                name,
                vec![SourceParameter::new("float*", "out")],
                format!("out[0] = {}.0f;\n", process::id()),
            ),
        )),
        abi: KernelAbi {
            arguments: vec![KernelArgument::Output(0)],
            scratch: Vec::new(),
        },
        launch: KernelLaunch {
            blocks: 1,
            threads: 1,
            shared_bytes: 0,
            grid: [1, 1],
        },
        description: KernelDescription::default(),
        target: None,
        requirements: CompileRequirements::default(),
    };
    let source = kernel.render(KernelOutput::Nvrtc).unwrap();
    let options = kernel_compile_options(
        runtime.architecture(),
        CompileRequirements::default(),
        ComputePrecision::Full,
    )
    .unwrap();

    let (compiled, origin) = cubin(name, &source, &options).unwrap();
    assert_eq!(origin, CubinOrigin::Compiled);
    let (cached, origin) = cubin(name, &source, &options).unwrap();
    assert_eq!(origin, CubinOrigin::Cached);
    assert_eq!(cached, compiled);

    let entries = fs::read_dir(&directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    assert_eq!(entries.len(), 1);
    fs::write(&entries[0], b"not a cubin").unwrap();
    runtime.kernel(&kernel).unwrap();
    assert_ne!(fs::read(&entries[0]).unwrap(), b"not a cubin");

    cuda::disable_kernel_cache();
    let (_, origin) = cubin(name, &source, &options).unwrap();
    assert_eq!(origin, CubinOrigin::Compiled);
    fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn test_generated_kernels_compile_exactly_unless_numerics_are_relaxed() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some(runtime) = skip_if_stub(runtime_for_device(Device::Cuda(0))) else {
        return;
    };
    for precision in [ComputePrecision::Full, ComputePrecision::AllowTf32] {
        let arguments = kernel_compile_options(
            runtime.architecture(),
            CompileRequirements::default(),
            precision,
        )
        .unwrap()
        .as_arguments();
        for exact in ["--fmad=false", "--prec-div=true", "--prec-sqrt=true"] {
            assert!(
                arguments.iter().any(|argument| argument == exact),
                "{arguments:?}"
            );
        }
    }
    let relaxed = kernel_compile_options(
        runtime.architecture(),
        CompileRequirements::default(),
        ComputePrecision::Relaxed,
    )
    .unwrap()
    .as_arguments();
    assert!(
        relaxed.iter().any(|argument| argument == "--use_fast_math"),
        "{relaxed:?}"
    );
}
