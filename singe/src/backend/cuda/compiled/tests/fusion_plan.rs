use std::collections::HashMap;

use singe_core::assert_close;

use crate::{
    backend::{
        cuda::{
            self,
            compiled::{
                executable::CudaCompiledGraph,
                plan::fusion::{FusionGroup, GroupKind},
            },
        },
        registry::{LayerNormalizationAttributes, Operation, OperationAttributes},
    },
    data_type::DataType,
    device::Device,
    dispatch::call_one,
    error::Result,
    graph::{Graph, GraphCudaOptions, ir::OperationNode, test_support::patterned_tensor},
    shape::Shape,
    tensor::Tensor,
    testing::{maybe_lock_device, skip_if_stub},
};

/// Returns a tensor of `dimensions` on `device` with values in about `[-1, 1)`.
/// Compiles `graph`, checks every output against the eager outputs `eager`, and returns the compiled graph.
fn compiled(graph: &Graph, eager: &[Tensor], inputs: &[(&str, &Tensor)]) -> CudaCompiledGraph {
    let mut compiled = cuda::compile_graph(graph, &GraphCudaOptions::new()).unwrap();
    let inputs = inputs
        .iter()
        .map(|(name, tensor)| ((*name).to_owned(), (*tensor).clone()))
        .collect::<HashMap<_, _>>();
    let outputs = compiled.run(&inputs).unwrap();
    for (actual, expected) in outputs.iter().zip(eager) {
        assert_close!(
            &actual.to_vec::<f32>().unwrap(),
            &expected.to_vec::<f32>().unwrap(),
            1e-5
        );
    }
    compiled
}

/// Returns each group's kind and its members' operation names, in the order lowering visits them.
fn groups(compiled: &CudaCompiledGraph) -> Vec<(String, Vec<String>)> {
    let ir = compiled.ir();
    compiled
        .report()
        .fusion_plan()
        .groups()
        .iter()
        .map(|group: &FusionGroup| {
            let kind = match group.kind {
                GroupKind::Pointwise => "pointwise",
                GroupKind::Row { stored: false, .. } => "row",
                GroupKind::Row { stored: true, .. } => "row storing its input",
                GroupKind::Concat { .. } => "concat",
                GroupKind::Contraction { .. } => "contraction",
                GroupKind::Module(_) => "module",
            };
            let members = group
                .members
                .iter()
                .map(|&member| {
                    ir.required_operation_for_value(member)
                        .unwrap()
                        .opcode()
                        .unwrap()
                        .name()
                        .to_owned()
                })
                .collect();
            (kind.to_owned(), members)
        })
        .collect()
}

fn group(kind: &str, members: &[&str]) -> (String, Vec<String>) {
    (
        kind.to_owned(),
        members.iter().map(|&member| member.to_owned()).collect(),
    )
}

#[test]
fn test_fusion_plans_group_diamonds_broadcasts_and_row_prologues() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let Some(x) = skip_if_stub(Tensor::from_vec(
        device,
        Shape::create([4, 8]).unwrap(),
        (0..32)
            .map(|index| index as f32 / 16.0 - 1.0)
            .collect::<Vec<_>>(),
    )) else {
        return;
    };
    let bias = patterned_tensor(device, &[8], 2);
    // A diamond whose two branches read one value, with a broadcast bias, then a softmax that also reads the diamond's value.
    let (graph, eager) = Graph::capture([("x", x.clone()), ("bias", bias.clone())], |inputs| {
        let shifted = inputs[0].add(&inputs[1])?;
        let left = shifted.mul_scalar(2.0)?;
        let right = shifted.mul(&shifted)?;
        let joined = left.add(&right)?;
        Ok(vec![joined.softmax(-1)?, joined.sum(&[-1], true)?])
    })
    .unwrap();
    let compiled = compiled(&graph, &eager, &[("x", &x), ("bias", &bias)]);
    // The softmax computes the diamond as it loads its row and stores it for the sum, which a row group of its own reads.
    assert_eq!(
        groups(&compiled),
        [group(
            "row storing its input",
            &["add", "scale", "mul", "add", "softmax"]
        )],
        "{}",
        compiled.report().fusion_plan()
    );
}

#[test]
fn test_fusion_plans_never_wait_for_themselves_through_an_outside_operation() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let Some(x) = skip_if_stub(Tensor::from_vec(
        device,
        Shape::create([4, 8]).unwrap(),
        (0..32)
            .map(|index| index as f32 / 16.0 - 1.0)
            .collect::<Vec<_>>(),
    )) else {
        return;
    };
    // `scaled` reads `shifted` both directly and through the softmax, so a group holding both without the softmax would wait for itself.
    // The softmax's epilogue reads `shifted` from memory, since the kernel holds a prologue's values only at the row's own points.
    let (graph, eager) = Graph::capture([("x", x.clone())], |inputs| -> Result<Vec<Tensor>> {
        let shifted = inputs[0].add_scalar(1.0)?;
        let probabilities = shifted.softmax(-1)?;
        let scaled = shifted.mul(&probabilities)?;
        Ok(vec![scaled.mul_scalar(3.0)?])
    })
    .unwrap();
    let compiled = compiled(&graph, &eager, &[("x", &x)]);
    let plan = compiled.report().fusion_plan();
    assert_eq!(
        groups(&compiled),
        [group("row", &["softmax", "mul", "scale"])],
        "{plan}"
    );
}

#[test]
fn test_fusion_plans_store_rotated_halves_into_their_concatenation() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let Some(x) = skip_if_stub(Tensor::from_vec(
        device,
        Shape::create([2, 3, 8]).unwrap(),
        (0..48)
            .map(|index| index as f32 / 24.0 - 1.0)
            .collect::<Vec<_>>(),
    )) else {
        return;
    };
    let cos = patterned_tensor(device, &[3, 4], 5);
    let sin = patterned_tensor(device, &[3, 4], 6);
    // The rotary embedding: each half of the last axis rotated against the other, concatenated back.
    let (graph, eager) = Graph::capture(
        [("x", x.clone()), ("cos", cos.clone()), ("sin", sin.clone())],
        |inputs| {
            let (x, cos, sin) = (&inputs[0], &inputs[1], &inputs[2]);
            let first = x.narrow(2, 0, 4)?;
            let second = x.narrow(2, 4, 4)?;
            Ok(vec![Tensor::concat(
                &[
                    first.mul(cos)?.sub(&second.mul(sin)?)?,
                    second.mul(cos)?.add(&first.mul(sin)?)?,
                ],
                2,
            )?])
        },
    )
    .unwrap();
    let compiled = compiled(&graph, &eager, &[("x", &x), ("cos", &cos), ("sin", &sin)]);
    let plan = groups(&compiled);
    assert_eq!(plan.len(), 1, "{}", compiled.report().fusion_plan());
    assert_eq!(plan[0].0, "concat");
    assert_eq!(plan[0].1.len(), 7);
}

#[test]
fn test_fusion_plans_store_a_rotary_embedding_after_a_head_norm() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let (batch, sequence, heads, width) = (2_i64, 3, 2, 16);
    let half = width / 2;
    let Some(x) = skip_if_stub(Tensor::from_vec(
        device,
        Shape::create([batch, sequence, heads, width]).unwrap(),
        (0..batch * sequence * heads * width)
            .map(|index| (index % 13) as f32 / 6.5 - 1.0)
            .collect::<Vec<_>>(),
    )) else {
        return;
    };
    let weight = patterned_tensor(device, &[width], 7);
    let cos = patterned_tensor(device, &[sequence, half], 8);
    let sin = patterned_tensor(device, &[sequence, half], 9);
    // The q and k norms of a decoder layer: an RMS norm over each head, transposed to heads before the sequence, then rotated.
    let (graph, eager) = Graph::capture(
        [
            ("x", x.clone()),
            ("weight", weight.clone()),
            ("cos", cos.clone()),
            ("sin", sin.clone()),
        ],
        |inputs| {
            let normed = call_one(
                Operation::RmsNorm,
                OperationAttributes::LayerNormalization(LayerNormalizationAttributes {
                    normalized_shape: vec![width],
                    has_weight: true,
                    has_bias: false,
                    eps: 1e-6,
                }),
                &[&inputs[0], &inputs[1]],
            )?
            .transpose(1, 2)?;
            let (cos, sin) = (&inputs[2], &inputs[3]);
            let first = normed.narrow(3, 0, half)?;
            let second = normed.narrow(3, half, half)?;
            Ok(vec![Tensor::concat(
                &[
                    first.mul(cos)?.sub(&second.mul(sin)?)?,
                    second.mul(cos)?.add(&first.mul(sin)?)?,
                ],
                3,
            )?])
        },
    )
    .unwrap();
    let compiled = compiled(
        &graph,
        &eager,
        &[("x", &x), ("weight", &weight), ("cos", &cos), ("sin", &sin)],
    );
    let plan = groups(&compiled);
    assert_eq!(plan.len(), 1, "{}", compiled.report().fusion_plan());
    assert_eq!(plan[0].0, "row", "{}", compiled.report().fusion_plan());
    assert_eq!(plan[0].1[0], "rms_norm");
    assert_eq!(compiled.stats().kernel_child_graph_nodes(), 1);
}

#[test]
fn test_row_epilogues_take_the_side_chains_they_read() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let Some(x) = skip_if_stub(Tensor::from_vec(
        device,
        Shape::create([4, 8]).unwrap(),
        (0..32).map(|index| index as f32 / 16.0 - 1.0).collect(),
    )) else {
        return;
    };
    let (residual, gate, weight) = (
        patterned_tensor(device, &[4, 8], 1),
        patterned_tensor(device, &[4, 8], 2),
        patterned_tensor(device, &[8], 3),
    );
    let named = [
        ("x", x.clone()),
        ("residual", residual.clone()),
        ("gate", gate.clone()),
        ("weight", weight.clone()),
    ];
    // `silu(gate)` reads neither the norm nor its input, so it joins the epilogue that multiplies by it.
    let (graph, eager) = Graph::capture(named, |inputs| {
        let hidden = inputs[0].add(&inputs[1])?;
        let normed = call_one(
            Operation::RmsNorm,
            OperationAttributes::LayerNormalization(LayerNormalizationAttributes {
                normalized_shape: vec![8],
                has_weight: true,
                has_bias: false,
                eps: 1e-6,
            }),
            &[&hidden, &inputs[3]],
        )?;
        Ok(vec![hidden, normed.mul(&inputs[2].silu()?)?])
    })
    .unwrap();
    let compiled = compiled(
        &graph,
        &eager,
        &[
            ("x", &x),
            ("residual", &residual),
            ("gate", &gate),
            ("weight", &weight),
        ],
    );
    assert_eq!(
        groups(&compiled),
        [group(
            "row storing its input",
            &["add", "rms_norm", "silu", "mul"]
        )],
        "{}",
        compiled.report().fusion_plan()
    );
}

#[test]
fn test_softmax_rows_take_epilogues() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let Some(x) = skip_if_stub(Tensor::from_vec(
        device,
        Shape::create([4, 8]).unwrap(),
        (0..32).map(|index| index as f32 / 8.0 - 2.0).collect(),
    )) else {
        return;
    };
    let (mask, bias) = (
        patterned_tensor(device, &[4, 8], 1),
        patterned_tensor(device, &[8], 2),
    );
    let named = [
        ("x", x.clone()),
        ("mask", mask.clone()),
        ("bias", bias.clone()),
    ];
    // Each softmax stores its epilogue's value in place of its own, and the second also computes its input as a prologue.
    let (graph, eager) = Graph::capture(named, |inputs| {
        let weights = inputs[0].softmax(-1)?.mul(&inputs[1])?;
        let logs = inputs[0].mul_scalar(0.5)?.log_softmax(1)?;
        Ok(vec![weights, logs.mul_scalar(2.0)?.add(&inputs[2])?])
    })
    .unwrap();
    let compiled = compiled(
        &graph,
        &eager,
        &[("x", &x), ("mask", &mask), ("bias", &bias)],
    );
    let groups = groups(&compiled);
    assert!(
        groups.contains(&group("row", &["softmax", "mul"])),
        "{}",
        compiled.report().fusion_plan()
    );
    assert!(
        groups.iter().any(
            |(_, members)| members.iter().any(|member| member == "log_softmax")
                && members.iter().any(|member| member == "add")
        ),
        "{}",
        compiled.report().fusion_plan()
    );
}

#[test]
fn test_reduction_rows_take_epilogues() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let Some(x) = skip_if_stub(Tensor::from_vec(
        device,
        Shape::create([64, 32]).unwrap(),
        (0..64 * 32)
            .map(|index| (index % 17) as f32 / 8.0 - 1.0)
            .collect(),
    )) else {
        return;
    };
    let bias = patterned_tensor(device, &[1, 32], 1);
    let named = [("x", x.clone()), ("bias", bias.clone())];
    // A mean over the rows' last axis and a sum over their first, each followed by element-wise work on the reduced values, one of which reads a broadcast input.
    let (graph, eager) = Graph::capture(named, |inputs| {
        let squashed = inputs[0].mean(&[1], false)?.tanh()?.mul_scalar(2.0)?;
        let shifted = inputs[0].sum(&[0], true)?.add(&inputs[1])?.relu()?;
        Ok(vec![squashed, shifted])
    })
    .unwrap();
    let compiled = compiled(&graph, &eager, &[("x", &x), ("bias", &bias)]);
    let groups = groups(&compiled);
    assert!(
        groups.contains(&group("row", &["mean", "tanh", "scale"])),
        "{}",
        compiled.report().fusion_plan()
    );
    assert!(
        groups.contains(&group("row", &["sum", "add", "relu"])),
        "{}",
        compiled.report().fusion_plan()
    );
}

#[test]
fn test_reductions_that_keep_their_axes_store_epilogues_of_their_input_shape() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let Some(x) = skip_if_stub(Tensor::from_vec(
        device,
        Shape::create([2, 64, 4, 8]).unwrap(),
        (0..2 * 64 * 4 * 8)
            .map(|index| (index % 23) as f32 / 8.0 - 1.5)
            .collect(),
    )) else {
        return;
    };
    let scale = patterned_tensor(device, &[1, 1, 1, 8], 1);
    let named = [("x", x.clone()), ("scale", scale.clone())];
    // RMS norm over the channels, which divides each point by its column's root; centering along the last axis, which reads the reduced input itself; and a broadcast input scaling the result.
    let (graph, eager) = Graph::capture(named, |inputs| {
        let x = &inputs[0];
        let root = x.mul(x)?.mean(&[1], true)?.add_scalar(1e-5)?.sqrt()?;
        let normalized = x.div(&root)?;
        let centered = x.sub(&x.max_reduce(&[3], true)?)?.mul(&inputs[1])?;
        Ok(vec![normalized, centered])
    })
    .unwrap();
    let compiled = compiled(&graph, &eager, &[("x", &x), ("scale", &scale)]);
    let groups = groups(&compiled);
    let fused = |reduction: &str, last: &str| {
        groups.iter().any(|(kind, members)| {
            kind == "row"
                && members.iter().any(|member| member == reduction)
                && members.last().is_some_and(|member| member == last)
        })
    };
    assert!(fused("mean", "div"), "{}", compiled.report().fusion_plan());
    assert!(fused("max", "mul"), "{}", compiled.report().fusion_plan());
}

#[test]
fn test_argument_reductions_compute_their_input_as_a_prologue() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let Some(x) = skip_if_stub(Tensor::from_vec(
        device,
        Shape::create([16, 40]).unwrap(),
        (0..16 * 40)
            .map(|index| (index % 13) as f32 / 4.0 - 1.5)
            .collect(),
    )) else {
        return;
    };
    let bias = patterned_tensor(device, &[40], 1);
    let named = [("x", x.clone()), ("bias", bias.clone())];
    // Greedy sampling of scaled and shifted logits, along the rows and down the columns.
    let (graph, eager) = Graph::capture(named, |inputs| {
        let logits = inputs[0].mul_scalar(0.5)?.add(&inputs[1])?;
        let distances = inputs[0].sub_scalar(1.0)?.abs()?;
        Ok(vec![logits.argmax(-1)?, distances.argmin(0)?])
    })
    .unwrap();
    let mut compiled = cuda::compile_graph(&graph, &GraphCudaOptions::new()).unwrap();
    let outputs = compiled
        .run(&HashMap::from([
            ("x".to_owned(), x),
            ("bias".to_owned(), bias),
        ]))
        .unwrap();
    for (actual, expected) in outputs.iter().zip(&eager) {
        assert_eq!(
            actual.to_vec::<i64>().unwrap(),
            expected.to_vec::<i64>().unwrap()
        );
    }
    let groups = groups(&compiled);
    for reduction in ["argmax", "argmin"] {
        assert!(
            groups.iter().any(|(kind, members)| kind.starts_with("row")
                && members.last().is_some_and(|member| member == reduction)
                && members.len() > 1),
            "{}",
            compiled.report().fusion_plan()
        );
    }
}

#[test]
fn test_point_wise_groups_take_comparisons_and_casts_of_other_types() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let Some(x) = skip_if_stub(Tensor::from_vec(
        device,
        Shape::create([4, 8]).unwrap(),
        (0..32).map(|index| index as f32 / 16.0 - 1.0).collect(),
    )) else {
        return;
    };
    let threshold = patterned_tensor(device, &[8], 4);
    // The comparison is `Bool` and the cast brings it back to `F32`, all in one kernel.
    let (graph, eager) = Graph::capture(
        [("x", x.clone()), ("threshold", threshold.clone())],
        |inputs| {
            let above = inputs[0].greater(&inputs[1])?.cast(DataType::F32)?;
            Ok(vec![above.mul(&inputs[0])?])
        },
    )
    .unwrap();
    let compiled = compiled(&graph, &eager, &[("x", &x), ("threshold", &threshold)]);
    assert_eq!(
        groups(&compiled),
        [group("pointwise", &["greater", "cast", "mul"])],
        "{}",
        compiled.report().fusion_plan()
    );
}

#[test]
fn test_reductions_over_dimensions_of_one_are_identities() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let Some(x) = skip_if_stub(Tensor::from_vec(
        device,
        Shape::create([4, 1]).unwrap(),
        vec![0.5_f32, -1.0, 2.0, 3.5],
    )) else {
        return;
    };
    let (graph, eager) = Graph::capture([("x", x.clone())], |inputs| {
        Ok(vec![inputs[0].max_reduce(&[1], true)?.add(&inputs[0])?])
    })
    .unwrap();
    let compiled = compiled(&graph, &eager, &[("x", &x)]);
    let operations = compiled
        .ir()
        .operations()
        .iter()
        .filter_map(OperationNode::opcode)
        .collect::<Vec<_>>();
    assert_eq!(operations, [Operation::Add]);
}

#[test]
fn test_scans_compute_their_input_as_a_prologue() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let Some(x) = skip_if_stub(Tensor::from_vec(
        device,
        Shape::create([8, 300]).unwrap(),
        (0..8 * 300)
            .map(|index| (index % 11) as f32 / 8.0 - 0.5)
            .collect(),
    )) else {
        return;
    };
    let mask = patterned_tensor(device, &[8, 300], 1);
    let named = [("x", x.clone()), ("mask", mask.clone())];
    // A masked cumulative sum, as Mamba2's segment sums write it, scans the product it computes as it loads.
    let (graph, eager) = Graph::capture(named, |inputs| {
        Ok(vec![inputs[0].mul(&inputs[1])?.cumsum(1)?])
    })
    .unwrap();
    let compiled = compiled(&graph, &eager, &[("x", &x), ("mask", &mask)]);
    let groups = groups(&compiled);
    assert!(
        groups.contains(&group("row", &["mul", "cumsum"])),
        "{}",
        compiled.report().fusion_plan()
    );
}

// A value several readers share stays in one kernel that stores each output, and a trailing reduction joins the row group.
#[test]
fn test_fusion_groups_branch_and_end_in_reductions() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let Some(x) = skip_if_stub(Tensor::from_vec(
        device,
        Shape::create([4, 8]).unwrap(),
        (0..32).map(|index| index as f32 / 16.0 - 1.0).collect(),
    )) else {
        return;
    };
    let run = |function: &dyn Fn(&Tensor) -> Result<Vec<Tensor>>, kernels: usize| {
        let (graph, eager) =
            Graph::capture([("x", x.clone())], |inputs| function(&inputs[0])).unwrap();
        let mut compiled = cuda::compile_graph(&graph, &GraphCudaOptions::new()).unwrap();
        assert_eq!(compiled.stats().kernel_child_graph_nodes(), kernels);
        let outputs = compiled
            .run(&HashMap::from([("x".to_owned(), x.clone())]))
            .unwrap();
        for (actual, expected) in outputs.iter().zip(&eager) {
            assert_eq!(
                actual.to_vec::<f32>().unwrap(),
                expected.to_vec::<f32>().unwrap()
            );
        }
    };
    // `shared` has two readers, and both results leave the kernel.
    run(
        &|x| {
            let shared = x.add_scalar(1.0)?.relu()?;
            Ok(vec![shared.mul_scalar(2.0)?, shared.tanh()?])
        },
        1,
    );
    // The element-wise chain computes the reduction's input as the kernel loads it.
    run(
        &|x| Ok(vec![x.add_scalar(1.0)?.relu()?.sum(&[1], false)?]),
        1,
    );
}
