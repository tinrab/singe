use singe_codegen::emit::{EntryPoint, KernelSource, SourceLanguage, SourceParameter};

use crate::{
    backend::cuda::codegen::{
        BufferDescription, CompileRequirements, KernelAbi, KernelArgument, KernelArtifact,
        KernelCode, KernelDescription, KernelLaunch, KernelOutput,
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
