//! Procedural macros for compiling CUDA source and generating typed Singe kernel launch modules.

mod cuda_module;

use proc_macro::TokenStream;

/// Generates a typed kernel-launch module that compiles CUDA source with NVRTC at runtime.
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
///             nvrtc_args: ["--std=c++17"],
///         },
///     }
/// }
/// ```
///
/// Expansion parses the CUDA source with tree-sitter and emits a Rust module
/// with a `create` constructor plus typed wrappers for exported kernels.
/// CUDA compilation, PTX/cubin generation, and lowered kernel-name resolution happen through NVRTC when the generated module is created.
#[proc_macro]
pub fn cuda_module(input: TokenStream) -> TokenStream {
    match cuda_module::expand_module(input.into()) {
        Ok(output) => output.into(),
        Err(err) => err.to_compile_error().into(),
    }
}
