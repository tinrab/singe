//! Checks that Singe covers every `ATen` operation the reference models' forward passes call.
//!
//! `scripts/references/suites/model_operations.py` records the operations of a `Llama`, `Qwen3`, `Mistral`, `Whisper`, and `ViT` forward pass in `references/model_operations.json`.
//! Each one maps to a registry operation, a composite tensor method, a tensor creation function, or a reason it needs none, and every operation and composite it maps to must match `PyTorch` in `F32` on every execution path of the conformance harness.

use std::{collections::BTreeSet, fs, path::Path};

use serde_json::Value;
use singe::backend::registry::registered_operations;

use self::Coverage::{Composite, Creation, Nothing, Operation};

/// How Singe provides an `ATen` operation.
#[derive(Debug, Clone, Copy)]
enum Coverage {
    /// A registry operation, by name.
    Operation(&'static str),
    /// A tensor method composed from registry operations, by its conformance name.
    Composite(&'static str),
    /// A tensor creation function, which needs no kernel.
    Creation(&'static str),
    /// Nothing, for the reason given.
    Nothing(&'static str),
}

/// Every `ATen` operation the reference models call, and how Singe provides it.
const COVERAGE: &[(&str, Coverage)] = &[
    (
        "aten._scaled_dot_product_flash_attention_for_cpu",
        Operation("scaled_dot_product_attention"),
    ),
    ("aten._softmax", Operation("softmax")),
    ("aten._to_copy", Operation("cast")),
    ("aten._unsafe_view", Operation("reshape")),
    ("aten.add", Operation("add")),
    ("aten.addmm", Composite("addmm")),
    (
        "aten.alias",
        Nothing("a second name for a tensor, which Singe's views share"),
    ),
    ("aten.arange", Creation("Tensor::arange")),
    ("aten.bitwise_and", Operation("bitwise_and")),
    ("aten.bmm", Operation("matmul")),
    ("aten.cat", Operation("concat")),
    ("aten.clone", Operation("copy")),
    ("aten.convolution", Composite("conv1d")),
    ("aten.cos", Operation("cos")),
    ("aten.div", Operation("div")),
    ("aten.embedding", Operation("embedding")),
    ("aten.expand", Operation("expand")),
    ("aten.gelu", Operation("gelu_exact")),
    ("aten.gt", Operation("greater")),
    ("aten.index", Operation("index")),
    ("aten.le", Operation("less_equal")),
    (
        "aten.lift_fresh",
        Nothing("PyTorch's marker for a tensor made from Python data"),
    ),
    ("aten.mean", Operation("mean")),
    ("aten.mm", Operation("matmul")),
    ("aten.mul", Operation("mul")),
    ("aten.native_layer_norm", Operation("layer_norm")),
    ("aten.neg", Operation("neg")),
    ("aten.new_ones", Creation("Tensor::ones")),
    ("aten.permute", Operation("permute")),
    ("aten.pow", Operation("pow_scalar")),
    ("aten.repeat", Composite("repeat")),
    ("aten.rsqrt", Operation("rsqrt")),
    ("aten.scalar_tensor", Creation("Tensor::scalar")),
    ("aten.select", Composite("select")),
    ("aten.silu", Operation("silu")),
    ("aten.sin", Operation("sin")),
    ("aten.slice", Operation("slice")),
    ("aten.sub", Operation("sub")),
    ("aten.split", Composite("split")),
    ("aten.tanh", Operation("tanh")),
    ("aten.t", Operation("transpose")),
    ("aten.transpose", Operation("transpose")),
    ("aten.unsqueeze", Operation("unsqueeze")),
    ("aten.view", Operation("reshape")),
    ("aten.where", Operation("where")),
    ("aten.zeros", Creation("Tensor::zeroed_on")),
];

fn fixtures() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/references"))
}

/// Returns every `ATen` operation the fixture lists for any model.
fn model_operations() -> BTreeSet<String> {
    let text = fs::read_to_string(fixtures().join("model_operations.json")).unwrap();
    let fixture: Value = serde_json::from_str(&text).unwrap();
    fixture["models"]
        .as_object()
        .unwrap()
        .values()
        .flat_map(|operations| operations.as_object().unwrap().keys().cloned())
        .collect()
}

/// Returns the conformance harness's operation names, which name composites as well as registry operations.
fn conformance_operations() -> BTreeSet<String> {
    let text = fs::read_to_string(fixtures().join("conformance/cases.jsonl")).unwrap();
    text.lines()
        .filter_map(|line| {
            let case: Value = serde_json::from_str(line).unwrap();
            case["operation"].as_str().map(str::to_owned)
        })
        .collect()
}

#[test]
fn test_every_reference_model_operation_has_coverage() {
    let missing = model_operations()
        .into_iter()
        .filter(|operation| !COVERAGE.iter().any(|(name, _)| name == operation))
        .collect::<Vec<_>>();
    for (aten, coverage) in COVERAGE {
        if let Creation(text) | Nothing(text) = coverage {
            assert!(!text.is_empty(), "{aten} names no function or reason");
        }
    }
    assert!(
        missing.is_empty(),
        "reference model operations without coverage: {missing:?}"
    );
}

#[test]
fn test_reference_model_operations_match_pytorch_in_f32_on_every_path() {
    let registered = registered_operations()
        .iter()
        .map(|operation| operation.name())
        .collect::<BTreeSet<_>>();
    let tested = conformance_operations();
    let mut failures = Vec::new();
    for (aten, coverage) in COVERAGE {
        let name = match coverage {
            Operation(name) => {
                assert!(
                    registered.contains(name),
                    "{aten} maps to unknown operation {name}"
                );
                name
            }
            Composite(name) => name,
            Creation(_) | Nothing(_) => continue,
        };
        assert!(
            tested.contains(*name),
            "{aten}: the conformance harness has no {name} cases"
        );
        // Every `F32` case of the operation passes on every path, so no baseline lists one.
        for path in ["cpu", "cuda", "graph"] {
            let baseline = fs::read_to_string(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join(format!("tests/baselines/conformance/{path}.txt")),
            )
            .unwrap();
            let prefix = format!(" {name} F32 ");
            failures.extend(
                baseline
                    .lines()
                    .filter(|line| line.contains(&prefix) && !line.starts_with("extension"))
                    .map(|line| format!("{aten} on {path}: {line}")),
            );
        }
    }
    assert!(
        failures.is_empty(),
        "reference model operations that differ from PyTorch:\n{}",
        failures.join("\n")
    );
}
