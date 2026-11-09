//! Procedural macros for Singe tensor APIs.

mod tensor;

use proc_macro::TokenStream;

/// Builds a `singe::Tensor` from literal or expression-backed tensor data.
///
/// The macro accepts nested array syntax, a flat list, or a scalar expression.
/// Nested arrays must be rectangular; jagged inputs are rejected at compile time.
/// Scalars are created when the input is parsed as a scalar form, while array and flat-list inputs produce tensor data with inferred shape.
///
/// An optional `device = ...` argument selects the target device expression:
///
/// ```ignore
/// let cpu = singe::Device::Cpu;
/// let matrix = tensor!([[1.0f32, 2.0], [3.0, 4.0]], device = cpu);
/// let vector = tensor![1.0f32, 2.0, 3.0];
/// ```
///
/// Parse errors, unknown arguments, duplicate arguments, and inconsistent nested dimensions are reported as compile-time errors.
#[proc_macro]
pub fn tensor(input: TokenStream) -> TokenStream {
    match tensor::expand(input.into()) {
        Ok(output) => output.into(),
        Err(error) => error.to_compile_error().into(),
    }
}
