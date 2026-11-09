use singe_cublas::{
    lt::{
        matmul::MatmulAlgorithm,
        types::{MatmulAlgorithmConfigAttribute, ReductionScheme},
    },
    types::ComputeType,
};

use crate::{
    data_type::DataType,
    error::{Error, Result},
    graph::{Determinism, NumericalPolicy},
};

fn accepts_config(policy: NumericalPolicy, dtype: DataType, split: i32, reduction: u32) -> bool {
    if matches!(dtype, DataType::F16 | DataType::Bf16)
        && reduction != ReductionScheme::None as u32
        && reduction != ReductionScheme::ComputeType as u32
    {
        return false;
    }
    policy.determinism != Determinism::Require
        || ((0..=1).contains(&split) && reduction == ReductionScheme::None as u32)
}

pub fn algorithm_satisfies_numerics(algorithm: &MatmulAlgorithm, dtype: DataType) -> Result<bool> {
    let split = algorithm.config_attribute(MatmulAlgorithmConfigAttribute::SplitKCount)?;
    let reduction = algorithm.config_attribute(MatmulAlgorithmConfigAttribute::ReductionScheme)?;
    Ok(accepts_config(
        NumericalPolicy::current(),
        dtype,
        split,
        reduction,
    ))
}

pub fn validate_algorithm_numerics(
    algorithm: &MatmulAlgorithm,
    dtype: DataType,
    compute: ComputeType,
) -> Result<()> {
    let precision = NumericalPolicy::current().precision;
    let compute_allowed = compute == ComputeType::F32
        || (dtype == DataType::F32
            && precision.allows_tf32()
            && compute == ComputeType::F32FastTf32);
    if !compute_allowed || !algorithm_satisfies_numerics(algorithm, dtype)? {
        return Err(Error::unsupported_operation(
            "cuBLASLt algorithm violates precision or determinism policy",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
