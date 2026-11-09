use singe_cudnn::backend::behavior::BackendNumericalNote;

use crate::{
    backend::cuda::cudnn_backend::templates::tensors::accepts_numerical_notes,
    data_type::DataType,
    graph::{ComputePrecision, Determinism, NumericalPolicy},
};

#[test]
fn test_native_engine_policy_rejects_tf32_and_allows_half_tensor_cores_with_fp32_accumulation() {
    assert!(accepts_numerical_notes(
        NumericalPolicy::default(),
        DataType::F32,
        &[]
    ));
    assert!(!accepts_numerical_notes(
        NumericalPolicy::default(),
        DataType::F32,
        &[BackendNumericalNote::TensorCore]
    ));
    for dtype in [DataType::F16, DataType::Bf16, DataType::F32, DataType::F64] {
        assert!(!accepts_numerical_notes(
            NumericalPolicy::default(),
            dtype,
            &[BackendNumericalNote::DownConvertInputs]
        ));
        assert!(!accepts_numerical_notes(
            NumericalPolicy::default(),
            dtype,
            &[BackendNumericalNote::ReducedPrecisionReduction]
        ));
    }
    for dtype in [DataType::F16, DataType::Bf16] {
        assert!(accepts_numerical_notes(
            NumericalPolicy::default(),
            dtype,
            &[BackendNumericalNote::TensorCore]
        ));
        assert!(!accepts_numerical_notes(
            NumericalPolicy::default(),
            dtype,
            &[
                BackendNumericalNote::TensorCore,
                BackendNumericalNote::DownConvertInputs
            ]
        ));
    }
}

#[test]
fn test_numerical_notes_distinguish_tf32_permission_determinism_and_accumulator_precision() {
    let policy = NumericalPolicy {
        precision: ComputePrecision::AllowTf32,
        determinism: Determinism::Require,
    };
    assert!(accepts_numerical_notes(
        policy,
        DataType::F32,
        &[BackendNumericalNote::TensorCore]
    ));
    assert!(!accepts_numerical_notes(
        policy,
        DataType::F32,
        &[BackendNumericalNote::Nondeterministic]
    ));
    assert!(!accepts_numerical_notes(
        policy,
        DataType::F16,
        &[BackendNumericalNote::ReducedPrecisionReduction]
    ));
    assert!(!accepts_numerical_notes(
        policy,
        DataType::F32,
        &[BackendNumericalNote::DownConvertInputs]
    ));
    assert!(accepts_numerical_notes(
        NumericalPolicy::default(),
        DataType::F32,
        &[BackendNumericalNote::Nondeterministic]
    ));
}
