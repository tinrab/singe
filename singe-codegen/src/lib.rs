//! Kernel IR, schedules, and code emitters for Singe's generated kernels.
//!
//! [`ir::Kernel`] describes what one kernel computes: operands with element types, sizes that are either written into the code or passed at launch, an iteration domain of parallel and reduction axes, an expression graph of loads, arithmetic, and reductions, and the stores that write its outputs.
//! It says nothing about threads, blocks, or a code language.
//! A [`schedule::Schedule`] says how GPU threads cover a kernel, and [`emit`] turns a kernel and a schedule into code, with the parameters [`abi`] lists.
//! [`interpret`] runs a kernel on the CPU, as the reference every emitter is checked against.
//!
//! The crate depends on no GPU library, so code generation can be tested without a GPU; Singe's runtime compiles and launches what the emitters produce.

pub mod abi;
pub mod emit;
pub mod interpret;
pub mod ir;
pub mod schedule;
