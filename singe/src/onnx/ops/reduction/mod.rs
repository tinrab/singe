mod arg;
mod basic;
mod composed;
mod shape;

pub use self::{
    arg::lower_arg_reduction,
    basic::lower_reduction,
    composed::{lower_composed_sum_reduction, lower_norm_reduction},
    shape::{reduction_axis_to_usize, reduction_shape},
};

#[derive(Debug, Clone, Copy)]
pub enum ReductionKind {
    Mean,
    Sum,
    Max,
    Min,
    Product,
}

#[derive(Debug, Clone, Copy)]
pub enum NormReductionKind {
    L1,
    L2,
}

#[derive(Debug, Clone, Copy)]
pub enum ComposedSumReductionKind {
    SumSquare,
    LogSum,
    LogSumExp,
}
