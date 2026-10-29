/// Asserts that two numeric values or numeric slices are equal within a tolerance.
///
/// Slice forms compare lengths first and then compare each element using an
/// absolute difference. Scalar forms accept a label that is included in the panic
/// message. The default tolerance is `1.0e-5`.
#[macro_export]
macro_rules! assert_close {
    ($actual:expr, $expected:expr $(,)?) => {
        $crate::assert_close!($actual, $expected, 1.0e-5);
    };

    ($actual:expr, $expected:expr, $tolerance:expr $(,)?) => {{
        let actual = $actual;
        let expected = $expected;
        let tolerance = $tolerance as f64;

        assert_eq!(
            actual.len(),
            expected.len(),
            "assert_close length mismatch: actual len {}, expected len {}",
            actual.len(),
            expected.len(),
        );

        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            let actual = *actual as f64;
            let expected = *expected as f64;
            let difference = (actual - expected).abs();

            assert!(
                difference <= tolerance,
                "assert_close failed at index {index}: actual {actual}, expected {expected}, difference {difference}, tolerance {tolerance}",
            );
        }
    }};

    ($actual:expr, $expected:expr, $tolerance:expr, $name:expr $(,)?) => {{
        let actual = $actual as f64;
        let expected = $expected as f64;
        let tolerance = $tolerance as f64;
        let difference = (actual - expected).abs();

        assert!(
            difference <= tolerance,
            "assert_close failed for {}: actual {actual}, expected {expected}, difference {difference}, tolerance {tolerance}",
            $name,
        );
    }};
}

/// Asserts that two complex numeric values are equal within a tolerance.
///
/// The compared values must expose `re` and `im` numeric fields. This keeps the
/// helper independent of any particular complex-number crate while covering the
/// CUDA complex wrapper types used in examples and tests. The default tolerance
/// is `1.0e-5`.
#[macro_export]
macro_rules! assert_complex_close {
    ($actual:expr, $expected:expr $(,)?) => {
        $crate::assert_complex_close!($actual, $expected, 1.0e-5);
    };

    ($actual:expr, $expected:expr, $tolerance:expr $(,)?) => {{
        let actual = $actual;
        let expected = $expected;
        let tolerance = $tolerance as f64;
        let real_difference = (actual.re as f64 - expected.re as f64).abs();
        let imaginary_difference = (actual.im as f64 - expected.im as f64).abs();

        assert!(
            real_difference <= tolerance && imaginary_difference <= tolerance,
            "assert_complex_close failed: actual ({}, {}i), expected ({}, {}i), real difference {}, imaginary difference {}, tolerance {}",
            actual.re,
            actual.im,
            expected.re,
            expected.im,
            real_difference,
            imaginary_difference,
            tolerance,
        );
    }};
}
