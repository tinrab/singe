//! Low-level cuDNN backend descriptor, engine, and execution APIs.

/// Typed backend attribute names, native representations, and attribute-value conversion.
pub mod attribute;
/// Native engine behavior notes, execution requirements, and numerical properties.
pub mod behavior;
/// Convolution mode and geometry types shared by backend operations.
pub mod convolution;
/// Owned backend descriptors with typed attribute access and explicit finalization.
pub mod descriptor;
/// Device properties for native and deviceless descriptor and engine planning.
pub mod device;
/// Native engine identities, properties, and tuning-knob declarations.
pub mod engine;
pub mod errata;
/// Finalized operation graphs, execution plans, tensor bindings, and native operation builders.
pub mod execution;
/// Native heuristic modes and policies for combining candidate queries.
pub mod heuristic;
/// Native engine tuning-knob types and their integer value ranges.
pub mod knob;
/// Native layout types and engine layout requirements.
pub mod layout;
/// Native normalization modes and forward-phase selectors.
pub mod normalization;
/// Pointwise operation modes and operand-role declarations.
pub mod pointwise;
/// Reduction operators, index-generation policies, and index types.
pub mod reduction;
/// Backend tensor identities, layouts, encodings, and finalized tensor descriptors.
pub mod tensor;
