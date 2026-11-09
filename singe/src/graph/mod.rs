mod builders;

/// Eager operation recording, explicit tensor bindings, and host observation policies.
pub mod capture;

mod compile;
mod constants;
mod dimensions;
mod fingerprint;
mod generators;
mod interpreter;
mod run;

/// Graph input metadata, storage, and symbolic dimension validation.
pub mod guards;

mod inputs;

/// Explicit graph operations, values, bindings, and control-flow blocks.
pub mod ir;

/// Construction graph nodes and declared output metadata.
pub mod node;

mod normalization;
mod numerics;
mod operation;
mod outputs;
mod passes;
mod prepare;
mod program;
mod remap;
mod shape_program;
mod signature;
mod specialization;
mod symbolic;
#[cfg(test)]
pub(crate) mod test_support;

#[cfg(feature = "onnx")]
pub(crate) use self::passes::decompose_all_attention;
#[cfg(feature = "onnx")]
pub(crate) use self::shape_program::{
    DimensionConstraint, ReshapeExpression, ShapeValue, ValueShapeConstraint,
};

#[cfg(feature = "cuda")]
pub(crate) use self::passes::merged_projection_bytes;
pub use self::{
    capture::{CaptureBindings, CaptureInfo, CaptureOptions},
    compile::{CompiledGraph, GraphCompileOptions},
};

#[cfg(feature = "onnx")]
pub(crate) use self::dimensions::{Arithmetic, DimensionExpression};
#[cfg(feature = "cuda")]
pub use self::prepare::GraphCudaOptions;
pub(crate) use self::prepare::SharedBuffers;
use self::{ir::ValueId, outputs::GraphOutput};

#[cfg(feature = "cuda")]
pub(crate) use self::run::narrow_counted;
pub use self::{
    dimensions::{DimensionExpr, GraphInputSpec},
    fingerprint::{GraphFingerprint, GraphFingerprintOptions},
    guards::GraphGuardFailure,
    inputs::{GraphInputClass, InputDimensionGuard},
    ir::Graph,
    numerics::{ComputePrecision, Determinism},
    operation::{GraphOperationInputs, GraphOperationOutput, GraphOperationRequest},
    prepare::{
        GraphExecutionCacheStats, GraphExecutionMemoryStats, GraphExecutionMode,
        GraphExecutionPlan, GraphExecutor, GraphPrepareOptions,
    },
    program::{
        GraphProgram, ProgramFingerprint, ProgramInput, ProgramInputBinding, ProgramSignature,
    },
    run::GraphRunStats,
    signature::{GraphSignature, GraphSignatureOutput},
    specialization::GraphSpecializationKey,
};
pub(crate) use self::{
    numerics::{NumericalPolicy, NumericalScope},
    operation::view_output_layout,
};
pub use crate::state::{
    StateBindingKind, StateBindingSpec, StateFreezePolicy, StateRevision, StateSnapshot,
};
