#[cfg(all(test, feature = "testing"))]
mod tests {
    use crate::{testing::setup_context, types::MathMode};

    #[test]
    fn test_math_mode_bit_combinations_roundtrip_through_the_integer_abi() {
        let context = setup_context().unwrap();
        let original = context.math_mode().unwrap();
        for mode in [
            MathMode::PEDANTIC | MathMode::DISALLOW_REDUCED_PRECISION_REDUCTION,
            MathMode::TF32_TENSOR_OP | MathMode::DISALLOW_REDUCED_PRECISION_REDUCTION,
            MathMode::DEFAULT,
        ] {
            context.set_math_mode(mode).unwrap();
            assert_eq!(context.math_mode().unwrap(), mode);
        }
        context.set_math_mode(original).unwrap();
    }
}
