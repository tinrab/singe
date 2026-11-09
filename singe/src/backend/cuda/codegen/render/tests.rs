use singe_codegen::emit::{
    EntryLinkage, EntryPoint, KernelSource, SourceLanguage, SourceParameter,
};
use singe_cuda::architecture::GpuArchitecture;

use crate::{
    backend::{
        cuda::{
            codegen::{
                BufferDescription, CompileRequirements, KernelAbi, KernelArgument, KernelArtifact,
                KernelCode, KernelDescription, KernelLaunch, KernelOutput, ProgramLaunch,
            },
            kernel_plan::{KernelInput, KernelPlanner},
        },
        registry::{Operation, OperationAttributes},
    },
    data_type::DataType,
    layout::Layout,
};

fn artifact(arguments: Vec<KernelArgument>, parameters: Vec<SourceParameter>) -> KernelArtifact {
    let source = KernelSource::new(
        SourceLanguage::CudaC,
        "// prelude\n",
        "__device__ int helper() { return 1; }\n",
        EntryPoint::new("singe_test", parameters, ""),
    );
    KernelArtifact {
        name: "singe_test".to_owned(),
        code: KernelCode::Source(source),
        abi: KernelAbi {
            arguments,
            scratch: Vec::new(),
        },
        launch: KernelLaunch {
            blocks: 3,
            threads: 128,
            shared_bytes: 0,
            grid: [1, 1],
        },
        description: KernelDescription::default(),
        target: None,
        requirements: CompileRequirements::default(),
    }
}

fn gather() -> KernelArtifact {
    let input = |dimensions: &[i64], data_type| {
        KernelInput::new(data_type, Layout::contiguous(dimensions).unwrap())
    };
    KernelPlanner::default()
        .create(
            Operation::IndexSelect,
            &OperationAttributes::IndexSelect(0),
            &[
                input(&[4, 3], DataType::F32).with_name("table"),
                input(&[2], DataType::I64),
            ],
        )
        .unwrap()
        .kernel
}

#[test]
fn test_default_rendering_is_the_source_nvrtc_compiles() {
    let kernel = artifact(Vec::new(), Vec::new());
    let source = kernel.code.source().unwrap();
    let text = kernel.render(KernelOutput::Nvrtc).unwrap();
    assert_eq!(
        text,
        format!(
            "{}{}{}",
            source.prelude(),
            source.declarations(),
            source.render_definition(EntryLinkage::C)
        )
    );
    assert_eq!(
        text,
        "// prelude\n__device__ int helper() { return 1; }\nextern \"C\" __global__ void singe_test() {\n}\n"
    );
    let own = kernel
        .render(KernelOutput::Listing {
            documentation: false,
        })
        .unwrap();
    assert_eq!(own, text.trim_start_matches("// prelude\n"));
}

#[test]
fn test_launchers_take_the_caller_buffers_and_pass_the_planned_values() {
    let gather = gather();
    let options = KernelOutput::File {
        documentation: false,
    };
    let text = gather.render(options).unwrap();
    let source = gather.code.source().unwrap();

    assert!(text.starts_with("#include <cuda_runtime.h>\n"));
    assert!(text.contains(&source.render_definition(EntryLinkage::C)));
    assert!(!text.contains("@param"));
    let launcher = text
        .lines()
        .find(|line| line.starts_with(&format!("inline cudaError_t launch_{}(", gather.name)))
        .unwrap();
    assert!(launcher.contains("unsigned int* status, cudaStream_t stream = 0)"));
    assert!(!launcher.contains("unsigned long long"));
    let launch = format!(
        "    {}<<<{}, {}, 0, stream>>>(",
        gather.name, gather.launch.blocks, gather.launch.threads
    );
    assert!(text.contains(&launch));
}

#[test]
fn test_documentation_states_what_the_description_records() {
    let gather = gather();
    assert_eq!(gather.description.operations, ["index_select"]);
    assert!(
        !gather
            .render(KernelOutput::Nvrtc)
            .unwrap()
            .contains("@param")
    );
    let options = KernelOutput::File {
        documentation: true,
    };
    let text = gather.render(options).unwrap();
    let parameters = gather.code.source().unwrap().entry().parameters();
    // Every parameter is documented on the kernel, and the buffers again on the launcher.
    for parameter in parameters {
        let line = format!("/// @param {} ", parameter.name());
        assert!(text.contains(&line), "{line} is missing from\n{text}");
    }
    assert!(text.contains(&format!("/// `{}` computes index_select.\n", gather.name)));
    assert!(text.contains(" input 0 `table`: f32 [4, 3], strides [3, 1]\n"));
    assert!(text.contains(" input 1: i64 [2], strides [1]\n"));
    assert!(text.contains(" output 0, the index_select result: f32 [2, 3], strides [3, 1]\n"));
    assert!(text.contains("/// @param stream the stream the kernel runs on\n"));
}

#[test]
fn test_literals_read_back_as_the_planned_values() {
    let arguments = vec![
        KernelArgument::Output(0),
        KernelArgument::I32(i32::MIN),
        KernelArgument::I64(i64::MIN),
        KernelArgument::U64(7),
        KernelArgument::F32(0.1),
        KernelArgument::F32(f32::NEG_INFINITY),
        KernelArgument::F64(1e300),
        KernelArgument::F64(f64::NAN),
        KernelArgument::Bool(true),
        KernelArgument::Size(9),
    ];
    let parameters = ["out", "a", "b", "c", "d", "e", "f", "g", "h", "s0"]
        .into_iter()
        .map(|name| SourceParameter::new("int", name))
        .collect();
    let mut kernel = artifact(arguments, parameters);
    kernel.description = KernelDescription {
        outputs: vec![BufferDescription {
            name: Some("y".to_owned()),
            producer: Some("minimum".to_owned()),
            data_type: DataType::I32,
            layout: Layout::contiguous([5]).unwrap(),
        }],
        scalars: vec!["minimum".to_owned()],
        sizes: vec!["rows".to_owned()],
        ..KernelDescription::default()
    };
    let text = kernel
        .render(KernelOutput::File {
            documentation: true,
        })
        .unwrap();

    assert!(text.contains("#include <limits>\n"));
    assert!(
        text.contains("inline cudaError_t launch_singe_test(int out, cudaStream_t stream = 0) {")
    );
    assert!(text.contains(
        "singe_test<<<3, 128, 0, stream>>>(out, (-2147483647 - 1), (-9223372036854775807LL - 1), 7ULL, 0.1f, -std::numeric_limits<float>::infinity(), 1e300, std::numeric_limits<double>::quiet_NaN(), true, 9ULL);"
    ));
    assert!(
        text.contains("/// @param out output 0 `y`, the minimum result: i32 [5], strides [1]\n")
    );
    assert!(text.contains("/// @param a scalar 0, minimum, planned as `(-2147483647 - 1)`\n"));
    assert!(text.contains("/// @param b scalar 1, planned as `(-9223372036854775807LL - 1)`\n"));
    assert!(text.contains("/// @param s0 size 0, rows, planned as `9ULL`\n"));
}

#[test]
fn test_rendering_needs_cuda_source_declaring_every_argument() {
    let mut tile = artifact(Vec::new(), Vec::new());
    tile.code = KernelCode::TileIr(vec![0; 4]);
    assert!(tile.render(KernelOutput::Nvrtc).is_err());

    let undeclared = artifact(vec![KernelArgument::Output(0)], Vec::new());
    assert!(undeclared.render(KernelOutput::Nvrtc).is_ok());
    let launcher = KernelOutput::File {
        documentation: false,
    };
    assert!(undeclared.render(launcher).is_err());
}

#[test]
fn test_launchers_of_empty_kernels_launch_nothing() {
    let mut empty = artifact(Vec::new(), Vec::new());
    empty.launch.blocks = 0;
    let options = KernelOutput::File {
        documentation: false,
    };
    let text = empty.render(options).unwrap();
    assert!(text.contains("{\n    return cudaSuccess;\n}\n"));
    assert!(!text.contains("<<<"));
}

#[test]
fn test_documentation_names_the_target_architecture() {
    let mut kernel = artifact(Vec::new(), Vec::new());
    kernel.target = Some(GpuArchitecture::Sm90);
    let options = KernelOutput::Listing {
        documentation: true,
    };
    let text = kernel.render(options).unwrap();
    assert!(text.contains(
        "/// Planned for `sm_90` (`nvcc -arch=sm_90`) to launch as 3 blocks of 128 threads.\n"
    ));
    kernel.target = None;
    assert!(
        kernel
            .render(options)
            .unwrap()
            .contains("/// Planned to launch as 3 blocks of 128 threads.\n")
    );
}

#[test]
fn test_launchers_of_three_dimensional_grids_pass_every_extent() {
    let mut kernel = artifact(Vec::new(), Vec::new());
    kernel.launch.grid = [2, 5];
    let options = KernelOutput::File {
        documentation: true,
    };
    let text = kernel.render(options).unwrap();
    assert!(
        text.contains("singe_test<<<dim3(3, 2, 5), 128, 0, stream>>>"),
        "{text}"
    );
    assert!(
        text.contains("to launch as 3 by 2 by 5 blocks of 128 threads"),
        "{text}"
    );
}

#[test]
fn test_program_units_hold_a_cpp_kernel_and_its_launcher_without_shared_text() {
    let gather = gather();
    let text = gather
        .render(KernelOutput::ProgramUnit(ProgramLaunch::Serial))
        .unwrap();
    let source = gather.code.source().unwrap();
    assert!(text.contains(&source.render_definition(EntryLinkage::Cpp)));
    assert!(!text.contains("extern \"C\""));
    assert!(!text.contains(source.prelude()));
    assert!(!text.contains("#include"));
    assert!(text.contains(&format!("inline cudaError_t launch_{}(", gather.name)));
}
