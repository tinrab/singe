//! Procedural macros for compiling CUDA source and generating typed Singe kernel launch modules.

mod cuda_module;

use proc_macro::TokenStream;

/// Compiles CUDA source with `nvcc` and generates a typed kernel-launch module.
///
/// The macro expects a module declaration containing a CUDA source file, exported
/// kernel names, optional header files, and optional compiler arguments:
///
/// ```ignore
/// cuda_module! {
///     pub mod kernels {
///         source: "kernels/add.cu",
///         exports: [add_kernel],
///         headers: ["kernels/common.cuh"],
///         compile: {
///             nvcc_args: ["--std=c++17"],
///             nvrtc_args: ["--std=c++17"],
///         },
///     }
/// }
/// ```
///
/// Expansion runs `nvcc`, using the surrounding CUDA environment including
/// `CUDA_CCBIN` when set, parses the resulting PTX with `singe-ptx`, and emits a
/// Rust module with a `create` constructor plus typed wrappers for exported
/// kernels. Syntax errors, missing files, `nvcc` failures, unsupported PTX, and
/// exports mismatches are reported as compile-time errors.
#[proc_macro]
pub fn cuda_module(input: TokenStream) -> TokenStream {
    match cuda_module::expand_module(input.into()) {
        Ok(output) => output.into(),
        Err(err) => err.to_compile_error().into(),
    }
}
