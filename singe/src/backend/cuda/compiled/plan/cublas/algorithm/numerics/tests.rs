use super::*;

#[test]
fn test_split_k_and_output_reductions_obey_independent_precision_and_determinism_contracts() {
    let default = NumericalPolicy::default();
    let deterministic = NumericalPolicy {
        determinism: Determinism::Require,
        ..default
    };
    for dtype in [DataType::F16, DataType::Bf16] {
        assert!(!accepts_config(
            default,
            dtype,
            4,
            ReductionScheme::OutputType as u32
        ));
        assert!(accepts_config(
            default,
            dtype,
            4,
            ReductionScheme::ComputeType as u32
        ));
        assert!(accepts_config(
            deterministic,
            dtype,
            0,
            ReductionScheme::None as u32
        ));
        assert!(!accepts_config(
            deterministic,
            dtype,
            4,
            ReductionScheme::ComputeType as u32
        ));
    }
    assert!(accepts_config(
        default,
        DataType::F32,
        4,
        ReductionScheme::OutputType as u32
    ));
    assert!(!accepts_config(
        deterministic,
        DataType::F32,
        4,
        ReductionScheme::OutputType as u32
    ));
}
