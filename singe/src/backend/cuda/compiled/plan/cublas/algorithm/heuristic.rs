use singe_cublas::{
    error::{Error, Status as CublasStatus},
    lt::{
        context::Context as CublasLtContext,
        matmul::{
            MatmulAlgorithm, MatmulPreference, check_matmul_algorithm, matmul_algorithm,
            matmul_algorithm_heuristics,
        },
        types::Epilogue,
    },
    types::ComputeType,
};

use crate::{
    backend::cuda::{
        compiled::plan::cublas::{
            algorithm::{
                CublasLtMatmulDTypePlan, algorithm_satisfies_numerics,
                cublaslt_matmul_heuristic_workspace_fits_limit,
            },
            descriptor::CublasLtMatmulDescriptors,
            layout::{CublasLtMatmulOutputLayout, CublasLtMatrixLayout},
        },
        graph_value::CudaGraphValue,
        runtime::{CublasLtMatmulAlgorithmEntry, CublasLtMatmulAlgorithmFingerprint},
    },
    error::Result,
};

pub fn cublaslt_heuristic_matmul_algorithm(
    cublaslt: &CublasLtContext,
    dtype_plan: CublasLtMatmulDTypePlan,
    compute_type: ComputeType,
    a_layout: &CublasLtMatrixLayout,
    b_layout: &CublasLtMatrixLayout,
    epilogue: Epilogue,
    bias: Option<&CudaGraphValue>,
    output_layout: &CublasLtMatmulOutputLayout,
    rows: u64,
    cols: u64,
    workspace_limit_bytes: usize,
) -> Result<Option<CublasLtMatmulAlgorithmEntry>> {
    let descriptors = CublasLtMatmulDescriptors::create(
        dtype_plan,
        compute_type,
        a_layout,
        b_layout,
        epilogue,
        bias,
        output_layout,
        rows,
        cols,
    )?;
    let mut preference = MatmulPreference::create()?;
    preference.set_max_workspace_bytes(workspace_limit_bytes)?;
    let heuristic = match matmul_algorithm_heuristics(
        cublaslt,
        &descriptors.descriptor,
        &descriptors.a_layout,
        &descriptors.b_layout,
        &descriptors.output_layout,
        &descriptors.output_layout,
        &preference,
        8,
    ) {
        Ok(heuristics) => {
            let mut selected = None;
            for candidate in heuristics {
                if !cublaslt_matmul_heuristic_workspace_fits_limit(
                    candidate.status,
                    candidate.workspace_size,
                    workspace_limit_bytes,
                ) {
                    continue;
                }

                let valid = check_matmul_algorithm(
                    cublaslt,
                    &descriptors.descriptor,
                    &descriptors.a_layout,
                    &descriptors.b_layout,
                    &descriptors.output_layout,
                    &descriptors.output_layout,
                    &candidate.algorithm,
                )
                .is_ok_and(|checked| {
                    cublaslt_matmul_heuristic_workspace_fits_limit(
                        checked.status,
                        checked.workspace_size,
                        workspace_limit_bytes,
                    )
                }) && candidate.algorithm.supports_strided_batch().is_ok()
                    && candidate.algorithm.supports_pointer_array_batch().is_ok();
                if valid
                    && algorithm_satisfies_numerics(&candidate.algorithm, dtype_plan.data_type)?
                {
                    selected = Some(candidate);
                    break;
                }
            }
            selected
        }
        Err(error) => {
            if matches!(
                error,
                Error::Cublas {
                    code: CublasStatus::NotSupported,
                    ..
                }
            ) {
                None
            } else {
                return Err(error.into());
            }
        }
    };
    heuristic
        .map(|heuristic| {
            let supports_strided_batch = heuristic.algorithm.supports_strided_batch()?;
            let supports_pointer_array_batch =
                heuristic.algorithm.supports_pointer_array_batch()?;
            let fingerprint =
                CublasLtMatmulAlgorithmFingerprint::from_algorithm(&heuristic.algorithm)?;
            Ok(CublasLtMatmulAlgorithmEntry::new_with_fingerprint(
                heuristic.algorithm,
                compute_type,
                heuristic.workspace_size,
                heuristic.waves_count,
                supports_strided_batch,
                supports_pointer_array_batch,
                fingerprint,
            ))
        })
        .transpose()
}

// What a matmul's descriptors say a heuristic may be judged against: whether cuBLASLt accepts it here, and whether it keeps the numerical policy.
fn usable_algorithm(
    cublaslt: &CublasLtContext,
    descriptors: &CublasLtMatmulDescriptors<'_>,
    dtype_plan: CublasLtMatmulDTypePlan,
    algorithm: &MatmulAlgorithm,
) -> Result<Option<CublasLtMatmulAlgorithmEntryParts>> {
    let Ok(checked) = check_matmul_algorithm(
        cublaslt,
        &descriptors.descriptor,
        &descriptors.a_layout,
        &descriptors.b_layout,
        &descriptors.output_layout,
        &descriptors.output_layout,
        algorithm,
    ) else {
        return Ok(None);
    };
    let usable =
        cublaslt_matmul_heuristic_workspace_fits_limit(checked.status, checked.workspace_size, 0)
            && algorithm.supports_strided_batch().is_ok()
            && algorithm.supports_pointer_array_batch().is_ok()
            && algorithm_satisfies_numerics(algorithm, dtype_plan.data_type)?;
    Ok(usable.then_some(CublasLtMatmulAlgorithmEntryParts {
        workspace_size: checked.workspace_size,
        waves_count: checked.waves_count,
    }))
}

struct CublasLtMatmulAlgorithmEntryParts {
    workspace_size: usize,
    waves_count: f32,
}

// The algorithm `config` states, if cuBLASLt accepts it for this matmul and it keeps the numerical policy.
pub fn cublaslt_configured_matmul_algorithm(
    cublaslt: &CublasLtContext,
    dtype_plan: CublasLtMatmulDTypePlan,
    compute_type: ComputeType,
    a_layout: &CublasLtMatrixLayout,
    b_layout: &CublasLtMatrixLayout,
    epilogue: Epilogue,
    bias: Option<&CudaGraphValue>,
    output_layout: &CublasLtMatmulOutputLayout,
    rows: u64,
    cols: u64,
    config: &CublasLtMatmulAlgorithmFingerprint,
) -> Result<Option<CublasLtMatmulAlgorithmEntry>> {
    let descriptors = CublasLtMatmulDescriptors::create(
        dtype_plan,
        compute_type,
        a_layout,
        b_layout,
        epilogue,
        bias,
        output_layout,
        rows,
        cols,
    )?;
    let Ok(mut algorithm) = configured_algorithm(cublaslt, dtype_plan, compute_type, config) else {
        return Ok(None);
    };
    if config.apply_to(&mut algorithm).is_err() {
        return Ok(None);
    }

    let Some(parts) = usable_algorithm(cublaslt, &descriptors, dtype_plan, &algorithm)? else {
        return Ok(None);
    };
    let supports_strided_batch = algorithm.supports_strided_batch()?;
    let supports_pointer_array_batch = algorithm.supports_pointer_array_batch()?;
    Ok(Some(CublasLtMatmulAlgorithmEntry::new_with_fingerprint(
        algorithm,
        compute_type,
        parts.workspace_size,
        parts.waves_count,
        supports_strided_batch,
        supports_pointer_array_batch,
        *config,
    )))
}

fn configured_algorithm(
    cublaslt: &CublasLtContext,
    dtype_plan: CublasLtMatmulDTypePlan,
    compute_type: ComputeType,
    config: &CublasLtMatmulAlgorithmFingerprint,
) -> Result<MatmulAlgorithm> {
    Ok(matmul_algorithm(
        cublaslt,
        compute_type,
        dtype_plan.scale_type,
        dtype_plan.cuda_data_type,
        dtype_plan.cuda_data_type,
        dtype_plan.cuda_data_type,
        dtype_plan.cuda_data_type,
        config.id,
    )?)
}

// How many configurations of one matmul tuning times, a bound the budget also limits.
const LISTED_CONFIGS: usize = 32;

// What cuBLASLt could run a matmul with, as TensorRT's `fcPlugin` times: what its heuristic lists, past the one it would choose first, and those again with their depth split.
pub fn cublaslt_listed_matmul_configs(
    cublaslt: &CublasLtContext,
    dtype_plan: CublasLtMatmulDTypePlan,
    compute_type: ComputeType,
    a_layout: &CublasLtMatrixLayout,
    b_layout: &CublasLtMatrixLayout,
    epilogue: Epilogue,
    bias: Option<&CudaGraphValue>,
    output_layout: &CublasLtMatmulOutputLayout,
    rows: u64,
    cols: u64,
) -> Result<Vec<CublasLtMatmulAlgorithmFingerprint>> {
    let descriptors = CublasLtMatmulDescriptors::create(
        dtype_plan,
        compute_type,
        a_layout,
        b_layout,
        epilogue,
        bias,
        output_layout,
        rows,
        cols,
    )?;
    let mut preference = MatmulPreference::create()?;
    preference.set_max_workspace_bytes(0)?;
    let Ok(heuristics) = matmul_algorithm_heuristics(
        cublaslt,
        &descriptors.descriptor,
        &descriptors.a_layout,
        &descriptors.b_layout,
        &descriptors.output_layout,
        &descriptors.output_layout,
        &preference,
        LISTED_CONFIGS,
    ) else {
        return Ok(Vec::new());
    };
    let mut configs = Vec::new();
    for heuristic in &heuristics {
        if usable_algorithm(cublaslt, &descriptors, dtype_plan, &heuristic.algorithm)?.is_some() {
            let config = CublasLtMatmulAlgorithmFingerprint::from_algorithm(&heuristic.algorithm)?;
            if !configs.contains(&config) {
                configs.push(config);
            }
        }
    }

    for heuristic in &heuristics {
        if !heuristic.algorithm.supports_split_k().unwrap_or(false) {
            continue;
        }

        let config = CublasLtMatmulAlgorithmFingerprint::from_algorithm(&heuristic.algorithm)?;
        for split in [2, 4, 8] {
            let split_config = config.with_split(split);
            if configs.contains(&split_config) || configs.len() >= LISTED_CONFIGS {
                continue;
            }

            let mut algorithm =
                configured_algorithm(cublaslt, dtype_plan, compute_type, &split_config)?;
            if split_config.apply_to(&mut algorithm).is_ok()
                && usable_algorithm(cublaslt, &descriptors, dtype_plan, &algorithm)?.is_some()
            {
                configs.push(split_config);
            }
        }
    }
    Ok(configs.into_iter().skip(1).collect())
}
