use singe_cuda::types::{Complex32, Complex64};

use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{SignalAllocator, Workspace},
    signal::{
        arithmetic, conversion,
        view::{SignalView, SignalViewMut},
    },
    types::{ComparisonOperation, ComplexI16},
};

#[path = "operation_traits.rs"]
mod operation_traits;

use self::operation_traits::{
    ComplexFixedThresholdSignal, ComplexThresholdSignal, ComplexValueThresholdSignal,
    ConstantSignal, DivideIntoConstantSignal, FixedThresholdSignal, SubtractFromConstantSignal,
    ThresholdSignal, UnarySignal, ValueThresholdSignal,
};
use super::{SignalBacking, SignalPipeline};

#[macro_use]
#[path = "operation_macros.rs"]
mod operation_macros;

#[path = "operation_registrations.rs"]
mod operation_registrations;

#[path = "operation_constant_methods.rs"]
mod operation_constant_methods;

#[path = "operation_unary_methods.rs"]
mod operation_unary_methods;

#[path = "operation_threshold_methods.rs"]
mod operation_threshold_methods;
