//! Compiles and caches generated kernels, whatever produced their code.
//!
//! A [`KernelArtifact`] is a kernel's name, its [`KernelCode`], its argument layout, its launch shape, and a [`KernelDescription`] of what its arguments hold, which [`KernelOutput`] can render as documentation.
//! Producers of kernels build one and hand it to [`CudaRuntime::kernel`](crate::backend::cuda::runtime::CudaRuntime::kernel), which compiles its code to cubin for the device, through the disk cache when [`enable_kernel_cache`] names a directory.
//! Code generation stays with the producers, so a new code language is a new [`KernelCode`] variant and its compiler here, not a change to the kernels' callers.

mod compile;
mod cuda_c;
mod description;
pub mod disk_cache;
mod emission;
mod kernel;
mod planning;
mod render;
#[cfg(test)]
mod test_support;
mod tile;

pub use self::{
    compile::kernel_compile_options,
    description::{BufferDescription, KernelDescription},
    disk_cache::{disable_kernel_cache, enable_kernel_cache},
    emission::{KernelLanguage, emit_kernel, plan_kernel},
    kernel::{
        CompileRequirements, KernelAbi, KernelArgument, KernelArtifact, KernelCode, KernelLaunch,
    },
    planning::{KernelDestination, KernelRecorder, PlanningContext},
    render::{KernelOutput, ProgramLaunch},
    tile::version as tile_compiler_version,
};
