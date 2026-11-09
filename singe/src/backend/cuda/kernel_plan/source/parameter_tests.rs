// Every parameter tuning may choose must compute what the default does, so a choice among them only ever changes speed.

use std::sync::Arc;

use crate::{
    backend::{
        cuda::{
            codegen::PlanningContext,
            kernel_plan::{
                self, KernelInput, KernelParameters, KernelPlan, KernelPlanner,
                source::{attention::PAD, parameters::MAX_STATIC_SHARED},
            },
            runtime::runtime_for_device,
            tuning::{TunedChoice, TunedRecord, TuningRecords},
        },
        registry::{Operation, OperationAttributes, ScaledDotProductAttentionAttributes},
    },
    data_type::DataType,
    device::Device,
    shape::Shape,
    tensor::Tensor,
    testing::{maybe_lock_device, skip_if_stub},
};

fn planner_with(key: &str, parameters: Option<KernelParameters>) -> KernelPlanner {
    let mut records = TuningRecords::new(None);
    if let Some(parameters) = parameters {
        records.insert(
            key.to_owned(),
            TunedRecord {
                choice: TunedChoice::Parameters(parameters),
                default_ns: None,
                best_ns: 0,
                bytes: 0,
            },
        );
    }
    let target = runtime_for_device(Device::Cuda(0)).unwrap().architecture();
    KernelPlanner::new(PlanningContext {
        records: Some(Arc::new(records)),
        target: Some(target),
        ..PlanningContext::default()
    })
}

fn tensor(dimensions: &[i64], data_type: DataType, seed: u32) -> Tensor {
    let count = usize::try_from(dimensions.iter().product::<i64>()).unwrap();
    let mut state = seed;
    let values = (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            f32::from(u16::try_from(state >> 16).unwrap()) / 32_768.0 - 1.0
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(
        Device::Cuda(0),
        Shape::create(dimensions.to_vec()).unwrap(),
        values,
    )
    .unwrap()
    .cast(data_type)
    .unwrap()
}

fn indices(values: &[i32]) -> Tensor {
    Tensor::from_vec(
        Device::Cuda(0),
        Shape::create([i64::try_from(values.len()).unwrap()]).unwrap(),
        values.to_vec(),
    )
    .unwrap()
}

fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f64, label: &str) {
    let values = |tensor: &Tensor| tensor.cast(DataType::F64).unwrap().to_vec::<f64>().unwrap();
    for (index, (actual, expected)) in values(actual).iter().zip(&values(expected)).enumerate() {
        assert!(
            actual == expected || (actual - expected).abs() <= tolerance,
            "{label} [{index}]: {actual} is not {expected}"
        );
    }
}

// Runs every candidate through `plan` and compares its outputs with the default's.
fn check_candidates(
    operation: Operation,
    attributes: &OperationAttributes,
    inputs: &[Tensor],
    tolerance: f64,
    plan: impl Fn(&KernelPlanner, &[KernelInput]) -> KernelPlan,
) {
    let metadata = inputs
        .iter()
        .map(KernelInput::from_source)
        .collect::<Vec<_>>();
    let references = inputs.iter().collect::<Vec<_>>();
    let (key, candidates) = planner_with("", None)
        .parameter_candidates(operation, attributes, &metadata)
        .expect("this node has parameters");
    assert!(candidates.len() > 1, "{key}");
    let expected =
        kernel_plan::run_outputs(&plan(&planner_with("", None), &metadata), &references).unwrap();
    for candidate in candidates {
        let planned = plan(&planner_with(&key, Some(candidate)), &metadata);
        let actual = kernel_plan::run_outputs(&planned, &references)
            .unwrap_or_else(|error| panic!("{candidate:?}: {error}"));
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_close(actual, expected, tolerance, &format!("{candidate:?}"));
        }
    }
}

#[test]
fn test_every_matmul_parameter_matches_the_default() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some(lhs) = skip_if_stub(Ok(tensor(&[2, 64], DataType::F16, 1))) else {
        return;
    };
    let few_rows_weight = tensor(&[32, 64], DataType::F16, 2).transpose(0, 1).unwrap();
    check_candidates(
        Operation::MatMul,
        &OperationAttributes::None,
        &[lhs, few_rows_weight],
        4e-2,
        |planner, inputs| {
            planner
                .create(Operation::MatMul, &OperationAttributes::None, inputs)
                .unwrap()
        },
    );
    let many_rows = tensor(&[64, 64], DataType::Bf16, 3);
    let weight = tensor(&[64, 64], DataType::Bf16, 4)
        .transpose(0, 1)
        .unwrap();
    check_candidates(
        Operation::MatMul,
        &OperationAttributes::None,
        &[many_rows, weight],
        0.2,
        |planner, inputs| {
            planner
                .create(Operation::MatMul, &OperationAttributes::None, inputs)
                .unwrap()
        },
    );
}

#[test]
fn test_every_attention_parameter_matches_the_default() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let attention = ScaledDotProductAttentionAttributes {
        scale: None,
        causal: true,
        window: None,
        softcap: None,
        masked: false,
        sinks: false,
        key_lengths: false,
    };
    let attributes = OperationAttributes::ScaledDotProductAttention(attention);
    let Some(query) = skip_if_stub(Ok(tensor(&[1, 4, 8, 16], DataType::F16, 5))) else {
        return;
    };
    check_candidates(
        Operation::ScaledDotProductAttention,
        &attributes,
        &[
            query,
            tensor(&[1, 2, 96, 16], DataType::F16, 6),
            tensor(&[1, 2, 96, 16], DataType::F16, 7),
        ],
        2e-2,
        |planner, inputs| {
            planner
                .create(Operation::ScaledDotProductAttention, &attributes, inputs)
                .unwrap()
        },
    );
}

// A tile past what a kernel may declare statically lives in what its launch asks for, in both attention kernels.
#[test]
fn test_attention_past_static_shared_memory_matches_the_default() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let dim = 128;
    assert!(
        planner_with("", None)
            .attention_candidates(dim)
            .iter()
            .any(|candidate| candidate.shared_bytes(dim + PAD) > MAX_STATIC_SHARED)
    );
    let attention = ScaledDotProductAttentionAttributes {
        scale: None,
        causal: true,
        window: None,
        softcap: None,
        masked: false,
        sinks: false,
        key_lengths: false,
    };
    let attributes = OperationAttributes::ScaledDotProductAttention(attention);
    let Some(query) = skip_if_stub(Ok(tensor(&[1, 1, 4, 128], DataType::F16, 11))) else {
        return;
    };
    check_candidates(
        Operation::ScaledDotProductAttention,
        &attributes,
        &[
            query,
            tensor(&[1, 1, 96, 128], DataType::F16, 12),
            tensor(&[1, 1, 96, 128], DataType::F16, 13),
        ],
        2e-2,
        |planner, inputs| {
            planner
                .create(Operation::ScaledDotProductAttention, &attributes, inputs)
                .unwrap()
        },
    );
    check_candidates(
        Operation::PagedAttention,
        &attributes,
        &[
            tensor(&[3, 1, 128], DataType::Bf16, 14),
            tensor(&[4, 32, 1, 128], DataType::Bf16, 15),
            tensor(&[4, 32, 1, 128], DataType::Bf16, 16),
            indices(&[0, 3]),
            indices(&[0, 2]),
            indices(&[0, 1]),
            indices(&[8]),
        ],
        0.1,
        |planner, inputs| planner.paged_attention(inputs, attention).unwrap(),
    );
}

#[test]
fn test_every_paged_attention_parameter_matches_the_default() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let attention = ScaledDotProductAttentionAttributes {
        scale: None,
        causal: true,
        window: None,
        softcap: None,
        masked: false,
        sinks: false,
        key_lengths: false,
    };
    let attributes = OperationAttributes::ScaledDotProductAttention(attention);
    let Some(query) = skip_if_stub(Ok(tensor(&[5, 4, 16], DataType::Bf16, 8))) else {
        return;
    };
    check_candidates(
        Operation::PagedAttention,
        &attributes,
        &[
            query,
            tensor(&[4, 32, 2, 16], DataType::Bf16, 9),
            tensor(&[4, 32, 2, 16], DataType::Bf16, 10),
            indices(&[0, 3, 5]),
            indices(&[0, 2, 4]),
            indices(&[0, 1, 3, 2]),
            indices(&[8, 4]),
        ],
        0.1,
        |planner, inputs| planner.paged_attention(inputs, attention).unwrap(),
    );
}

#[test]
fn test_every_latent_attention_parameter_matches_the_default() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let attention = ScaledDotProductAttentionAttributes {
        scale: Some(0.25),
        causal: true,
        window: None,
        softcap: None,
        masked: false,
        sinks: false,
        key_lengths: false,
    };
    let attributes = OperationAttributes::ScaledDotProductAttention(attention);
    let Some(query) = skip_if_stub(Ok(tensor(&[3, 2, 64], DataType::Bf16, 11))) else {
        return;
    };
    check_candidates(
        Operation::PagedMlaAttention,
        &attributes,
        &[
            query,
            tensor(&[3, 2, 32], DataType::Bf16, 12),
            tensor(&[4, 64, 64], DataType::Bf16, 13),
            tensor(&[4, 64, 32], DataType::Bf16, 14),
            indices(&[0, 2, 3]),
            indices(&[0, 2, 4]),
            indices(&[0, 1, 3, 2]),
            indices(&[16, 8]),
        ],
        0.1,
        |planner, inputs| planner.paged_mla_attention(inputs, attention).unwrap(),
    );
}

#[test]
fn test_every_grouped_matmul_parameter_matches_the_default() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some(input) = skip_if_stub(Ok(tensor(&[4, 64], DataType::F16, 15))) else {
        return;
    };
    check_candidates(
        Operation::GroupedMatMul,
        &OperationAttributes::None,
        &[
            input,
            tensor(&[2, 64, 32], DataType::F16, 16),
            indices(&[0, 2, 4]),
        ],
        4e-2,
        |planner, inputs| {
            planner
                .create(Operation::GroupedMatMul, &OperationAttributes::None, inputs)
                .unwrap()
        },
    );
}

// A kernel's input pointer starts at the operand's first element, so its addressing must not add the storage offset again.
#[test]
fn test_kernels_read_operands_that_start_inside_their_storage() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some(rows) = skip_if_stub(Ok(tensor(&[6, 64], DataType::F32, 20))) else {
        return;
    };
    let same = |planned: &KernelPlan, sliced: &[Tensor], copied: &[Tensor], label: &str| {
        let run = |inputs: &[Tensor]| {
            kernel_plan::run_outputs(planned, &inputs.iter().collect::<Vec<_>>()).unwrap()
        };
        for (actual, expected) in run(sliced).iter().zip(&run(copied)) {
            assert_close(actual, expected, 1e-6, label);
        }
    };
    let copy = |tensor: &Tensor| tensor.add_scalar(0.0).unwrap();
    let planner = planner_with("", None);

    let sliced = [
        rows.narrow(0, 2, 4).unwrap(),
        tensor(&[4, 64, 32], DataType::F32, 21)
            .narrow(0, 2, 2)
            .unwrap(),
        indices(&[0, 2, 4]),
    ];
    assert!(sliced[0].storage_offset() > 0 && sliced[1].storage_offset() > 0);
    let copied = [copy(&sliced[0]), copy(&sliced[1]), sliced[2].clone()];
    let metadata = sliced
        .iter()
        .map(KernelInput::from_source)
        .collect::<Vec<_>>();
    let kernel = planner
        .create(
            Operation::GroupedMatMul,
            &OperationAttributes::None,
            &metadata,
        )
        .unwrap();
    same(&kernel, &sliced, &copied, "grouped matmul");

    let attention = ScaledDotProductAttentionAttributes {
        scale: None,
        causal: true,
        window: None,
        softcap: None,
        masked: false,
        sinks: false,
        key_lengths: false,
    };
    let sliced = [
        tensor(&[7, 4, 16], DataType::F32, 22)
            .narrow(0, 2, 5)
            .unwrap(),
        tensor(&[6, 32, 2, 16], DataType::F32, 23)
            .narrow(0, 2, 4)
            .unwrap(),
        tensor(&[6, 32, 2, 16], DataType::F32, 24)
            .narrow(0, 2, 4)
            .unwrap(),
        indices(&[0, 3, 5]),
        indices(&[0, 2, 4]),
        indices(&[0, 1, 3, 2]),
        indices(&[8, 4]),
    ];
    assert!(sliced[1].storage_offset() > 0);
    let mut copied = sliced.clone();
    for tensor in &mut copied[..3] {
        *tensor = copy(tensor);
    }
    let metadata = sliced
        .iter()
        .map(KernelInput::from_source)
        .collect::<Vec<_>>();
    let kernel = planner.paged_attention(&metadata, attention).unwrap();
    same(&kernel, &sliced, &copied, "general paged attention");
}
