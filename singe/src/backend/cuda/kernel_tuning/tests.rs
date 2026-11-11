use std::time::Duration;

use crate::{
    backend::{
        cuda::kernel_tuning::{Outcome, TuneOptions, tune},
        kernel_lowering::{ElementGroup, LoweredKernel, normalization, softmax},
        registry::{LayerNormalizationAttributes, Operation},
    },
    data_type::DataType,
    device::Device,
    layout::Layout,
    testing::{maybe_lock_device, skip_if_stub},
};

/// Lowers RMS norm of BF16 `rows` × `width` with a weight.
fn rms_norm(rows: i64, width: i64) -> LoweredKernel {
    let input = Layout::contiguous([rows, width]).unwrap();
    let weight = Layout::contiguous([width]).unwrap();
    normalization(
        Operation::RmsNorm,
        &LayerNormalizationAttributes {
            normalized_shape: vec![width],
            has_weight: true,
            has_bias: false,
            eps: 1e-6,
        },
        &ElementGroup::input(DataType::Bf16, &input),
        &[(DataType::Bf16, &weight)],
    )
    .unwrap()
}

#[test]
fn test_every_listed_schedule_of_the_milestone_rows_matches_the_default() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let scores = Layout::contiguous([16, 256, 256]).unwrap();
    let kernels = [
        rms_norm(1, 1024),
        rms_norm(16, 1024),
        rms_norm(4096, 128),
        softmax(
            Operation::Softmax,
            &ElementGroup::input(DataType::Bf16, &scores),
            -1,
            None,
        )
        .unwrap(),
    ];
    let options = TuneOptions {
        tile: false,
        rounds: 1,
        budget: Duration::from_secs(120),
    };
    for lowered in kernels {
        let Some(result) = skip_if_stub(tune(
            device,
            &lowered.kernel,
            lowered.schedule,
            &lowered.bindings,
            options,
        )) else {
            return;
        };
        assert!(result.candidates.len() > 10, "{result:?}");
        assert!(
            result
                .candidates
                .iter()
                .all(|(_, outcome)| matches!(outcome, Outcome::Timed(_))),
            "{} {result:?}",
            lowered.kernel.name
        );
    }
}
