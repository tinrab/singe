use std::sync::Arc;

use crate::{
    backend::{
        descriptor::BackendDescriptorType,
        execution::{
            EngineChoice, EngineConfig, EngineHeuristics, EnginePlanningOptions, ExecutionPlan,
            KernelCache, OperationGraph, OperationGraphOptions, VariantPack,
        },
        heuristic::BackendHeuristicMode,
        operation::{
            attention::{
                AttentionSequenceInputs, DiagonalBandMaskBound, DiagonalBandMaskOperation,
                DiagonalBandMaskSpec, SdpaBackwardOperation, SdpaBackwardOptions, SdpaBackwardSink,
                SdpaBackwardTensors, SdpaForwardDropout, SdpaForwardOperation, SdpaForwardOptions,
                SdpaForwardSoftmax, SdpaForwardTensors, SdpaSubgraph, SoftmaxOperation,
                SoftmaxOutputs, SoftmaxSpec, SoftmaxStatsBase,
            },
            convolution::{
                ConvolutionBackwardDataOperation, ConvolutionBackwardFilterOperation,
                ConvolutionDescriptor, ConvolutionDescriptorSpec, ConvolutionGeometry,
                ConvolutionMode,
            },
            matmul::{MatmulDescriptor, MatmulDescriptorSpec, MatmulOperation, MatmulSpec},
            moe::{
                MoeGroupedMatmulInputs, MoeGroupedMatmulOperation, MoeGroupedMatmulRouting,
                MoeGroupedMatmulTensors,
            },
            normalization::{
                BackendNormalizationForwardPhase, BackendNormalizationMode,
                BatchNormalizationBackwardWeightsInputs,
                BatchNormalizationBackwardWeightsOperation,
                BatchNormalizationBackwardWeightsOutputs, GenStatsMode, GenStatsOperation,
                GenStatsOutputs, NormalizationBackwardOperation, NormalizationBackwardSpec,
                NormalizationBackwardTensors, NormalizationForwardOperation,
                NormalizationForwardSpec, NormalizationForwardTensors,
                NormalizationSavedStatistics,
            },
            pointwise::{
                PointwiseDescriptor, PointwiseDescriptorSpec, PointwiseMode, PointwiseOperation,
                PointwiseSpec,
            },
            reduction::{ReduceTensorOperator, ReductionDescriptor, ReductionDescriptorSpec},
            resample::{
                Fraction, PaddingMode, ResampleBackwardInputs, ResampleBackwardOperation,
                ResampleDescriptor, ResampleDescriptorSpec, ResampleForwardOperation,
                ResampleGeometry, ResampleMode, ResampleOutputs,
            },
            rng::{RandomNumberDistribution, RngDescriptor, RngDistribution},
            tensor_ops::{
                ConcatInPlaceMode, ConcatOperation, ConcatSpec, ReshapeMode, ReshapeOperation,
                ReshapeSpec,
            },
        },
        tensor::{BackendTensorReordering, BackendTensorUid, Tensor},
    },
    data_type::DataType,
    error::{Error, Result},
    math::NanPropagation,
    testing::setup_context,
};

#[test]
fn test_backend_tensor_descriptor_smoke_test() -> Result<()> {
    let _test = setup_context()?;

    let tensor = Tensor::create_with_id(
        BackendTensorUid::new(7),
        DataType::F32,
        &[2, 3],
        &[3, 1],
        16,
    )?;

    assert_eq!(tensor.id(), 7.into());
    assert_eq!(tensor.shape().dimensions(), &[2, 3]);
    assert_eq!(tensor.shape().strides(), &[3, 1]);
    assert!(tensor.descriptor().is_finalized());
    assert_eq!(
        tensor.descriptor().descriptor_type(),
        BackendDescriptorType::Tensor,
    );

    Ok(())
}

#[test]
fn test_backend_vectorized_tensor_descriptor_smoke_test() -> Result<()> {
    let _test = setup_context()?;

    let tensor = Tensor::create_vectorized(
        BackendTensorUid::new(9),
        DataType::I8,
        &[2, 32, 4, 4],
        &[512, 16, 4, 1],
        16,
        1,
        32,
        Some(BackendTensorReordering::Int8x32),
    )?;

    assert_eq!(tensor.id(), 9.into());
    assert_eq!(tensor.vectorized_dimension(), Some(1));
    assert_eq!(tensor.vector_count(), Some(32));
    assert!(tensor.descriptor().is_finalized());

    Ok(())
}

#[test]
fn test_variant_pack_accepts_empty_bindings() -> Result<()> {
    let test = setup_context()?;
    let context = test.context();
    // SAFETY: an empty variant pack has no bindings to keep alive.
    let pack = unsafe { VariantPack::create(context.cuda_context(), &[], None) }?;
    assert!(pack.descriptor().is_finalized());
    Ok(())
}

#[test]
fn test_operation_graph_requires_operations() -> Result<()> {
    let test = setup_context()?;
    let context = test.context();

    let error = OperationGraph::create(context, &[], OperationGraphOptions::default()).unwrap_err();
    assert!(matches!(error, Error::Empty { name } if name == "operations"));

    Ok(())
}

#[test]
fn test_engine_heuristics_returns_typed_engine_configs() -> Result<()> {
    let test = setup_context()?;
    let context = test.context();
    let x = Tensor::create_with_id(
        BackendTensorUid::new(1),
        DataType::F32,
        &[2, 3],
        &[3, 1],
        16,
    )?;
    let y = Tensor::create_with_id(
        BackendTensorUid::new(2),
        DataType::F32,
        &[2, 3],
        &[3, 1],
        16,
    )?;
    let pointwise = PointwiseDescriptor::create(
        PointwiseDescriptorSpec::new(PointwiseMode::ReluFwd, DataType::F32)
            .with_nan_propagation(NanPropagation::NotPropagate),
    )?;
    let op = PointwiseOperation::create(
        &pointwise,
        PointwiseSpec::Unary {
            x: &x,
            y: &y,
            alpha1: 1.0,
        },
    )?;
    let graph = OperationGraph::create(
        context,
        &[op.descriptor()],
        OperationGraphOptions::default(),
    )?;
    let heuristics = EngineHeuristics::create(
        context,
        &graph,
        BackendHeuristicMode::Instant,
        None,
        EnginePlanningOptions::default(),
    )?;

    let configs = heuristics.engine_configs(None)?;
    for config in &configs {
        let _ = config.workspace_size()?;
    }

    Ok(())
}

#[test]
fn test_engine_config_exposes_typed_inspection_helpers() -> Result<()> {
    let test = setup_context()?;
    let context = test.context();
    let x = Tensor::create_with_id(
        BackendTensorUid::new(1),
        DataType::F32,
        &[2, 3],
        &[3, 1],
        16,
    )?;
    let y = Tensor::create_with_id(
        BackendTensorUid::new(2),
        DataType::F32,
        &[2, 3],
        &[3, 1],
        16,
    )?;
    let pointwise = PointwiseDescriptor::create(
        PointwiseDescriptorSpec::new(PointwiseMode::ReluFwd, DataType::F32)
            .with_nan_propagation(NanPropagation::NotPropagate),
    )?;
    let op = PointwiseOperation::create(
        &pointwise,
        PointwiseSpec::Unary {
            x: &x,
            y: &y,
            alpha1: 1.0,
        },
    )?;
    let graph = OperationGraph::create(
        context,
        &[op.descriptor()],
        OperationGraphOptions::default(),
    )?;
    let heuristics = EngineHeuristics::create(
        context,
        &graph,
        BackendHeuristicMode::Instant,
        None,
        EnginePlanningOptions::default(),
    )?;

    for config in heuristics.engine_configs(None)? {
        let _ = config.workspace_size()?;
        let _ = config.knob_choices()?;
        let _ = config.knob_infos()?;
        let _ = config.intermediate_infos()?;
    }

    Ok(())
}

#[test]
fn test_engine_config_can_be_created_from_engine_index_and_knobs() -> Result<()> {
    let test = setup_context()?;
    let context = test.context();
    let x = Tensor::create_with_id(
        BackendTensorUid::new(1),
        DataType::F32,
        &[2, 3],
        &[3, 1],
        16,
    )?;
    let y = Tensor::create_with_id(
        BackendTensorUid::new(2),
        DataType::F32,
        &[2, 3],
        &[3, 1],
        16,
    )?;
    let pointwise = PointwiseDescriptor::create(
        PointwiseDescriptorSpec::new(PointwiseMode::ReluFwd, DataType::F32)
            .with_nan_propagation(NanPropagation::NotPropagate),
    )?;
    let op = PointwiseOperation::create(
        &pointwise,
        PointwiseSpec::Unary {
            x: &x,
            y: &y,
            alpha1: 1.0,
        },
    )?;
    let graph = OperationGraph::create(
        context,
        &[op.descriptor()],
        OperationGraphOptions::default(),
    )?;
    let heuristics = EngineHeuristics::create(
        context,
        &graph,
        BackendHeuristicMode::Instant,
        None,
        EnginePlanningOptions::default(),
    )?;

    for config in heuristics.engine_configs(None)? {
        let engine = config.engine()?;
        let knob_choices = config.knob_choices()?;
        let recreated = EngineConfig::create(
            context,
            &graph,
            &EngineChoice::new(engine.index(), knob_choices),
            None,
            EnginePlanningOptions::default(),
        )?;

        let _ = recreated.workspace_size()?;
        let recreated_engine = recreated.engine()?;
        assert_eq!(recreated_engine.index(), engine.index());
    }

    Ok(())
}

#[test]
fn test_execution_plan_create_smoke_test() -> Result<()> {
    let test = setup_context()?;
    let context = test.context();
    let x = Tensor::create_with_id(
        BackendTensorUid::new(1),
        DataType::F32,
        &[2, 3],
        &[3, 1],
        16,
    )?;
    let y = Tensor::create_with_id(
        BackendTensorUid::new(2),
        DataType::F32,
        &[2, 3],
        &[3, 1],
        16,
    )?;
    let pointwise = PointwiseDescriptor::create(
        PointwiseDescriptorSpec::new(PointwiseMode::ReluFwd, DataType::F32)
            .with_nan_propagation(NanPropagation::NotPropagate),
    )?;
    let op = PointwiseOperation::create(
        &pointwise,
        PointwiseSpec::Unary {
            x: &x,
            y: &y,
            alpha1: 1.0,
        },
    )?;
    let graph = OperationGraph::create(
        context,
        &[op.descriptor()],
        OperationGraphOptions::default(),
    )?;
    let heuristics = EngineHeuristics::create(
        context,
        &graph,
        BackendHeuristicMode::Instant,
        None,
        EnginePlanningOptions::default(),
    )?;

    if let Some(config) = heuristics.engine_configs(None)?.into_iter().next() {
        let plan = ExecutionPlan::create(context, &config, None, None)?;
        let _ = plan.workspace_size()?;
    }

    Ok(())
}

#[test]
fn test_pointwise_op_descriptor_smoke_test() -> Result<()> {
    let descriptor = PointwiseDescriptor::create(
        PointwiseDescriptorSpec::new(PointwiseMode::ReluFwd, DataType::F32)
            .with_nan_propagation(NanPropagation::NotPropagate),
    )?;

    assert!(descriptor.descriptor().is_finalized());
    assert_eq!(
        descriptor.descriptor().descriptor_type(),
        BackendDescriptorType::Pointwise,
    );

    Ok(())
}

#[test]
fn test_reduction_op_descriptor_smoke_test() -> Result<()> {
    let descriptor = ReductionDescriptor::create(ReductionDescriptorSpec::new(
        ReduceTensorOperator::Add,
        DataType::F32,
    ))?;

    assert!(descriptor.descriptor().is_finalized());
    assert_eq!(
        descriptor.descriptor().descriptor_type(),
        BackendDescriptorType::Reduction,
    );

    Ok(())
}

#[test]
fn test_gen_stats_op_smoke_test() -> Result<()> {
    let _test = setup_context()?;

    let x = Tensor::create_with_id(
        BackendTensorUid::new(60),
        DataType::F32,
        &[2, 4, 3, 3],
        &[36, 9, 3, 1],
        16,
    )?;
    let sum = Tensor::create_with_id(
        BackendTensorUid::new(61),
        DataType::F32,
        &[1, 4, 1, 1],
        &[4, 1, 1, 1],
        16,
    )?;
    let sqsum = Tensor::create_with_id(
        BackendTensorUid::new(62),
        DataType::F32,
        &[1, 4, 1, 1],
        &[4, 1, 1, 1],
        16,
    )?;

    let op = GenStatsOperation::create(
        GenStatsMode::SumSqsum,
        x.data_type(),
        &x,
        GenStatsOutputs::new(&sum, &sqsum),
    )?;

    assert!(op.descriptor().is_finalized());
    assert_eq!(
        op.descriptor().descriptor_type(),
        BackendDescriptorType::OperationGenStats,
    );

    Ok(())
}

#[test]
fn test_bn_backward_weights_op_smoke_test() -> Result<()> {
    let _test = setup_context()?;

    let mean = Tensor::create_with_id(
        BackendTensorUid::new(78),
        DataType::F32,
        &[1, 4, 1, 1],
        &[4, 1, 1, 1],
        16,
    )?;
    let inv_std = Tensor::create_with_id(
        BackendTensorUid::new(79),
        DataType::F32,
        &[1, 4, 1, 1],
        &[4, 1, 1, 1],
        16,
    )?;
    let bn_scale = Tensor::create_with_id(
        BackendTensorUid::new(80),
        DataType::F32,
        &[1, 4, 1, 1],
        &[4, 1, 1, 1],
        16,
    )?;
    let x = Tensor::create_with_id(
        BackendTensorUid::new(81),
        DataType::F32,
        &[2, 4, 3, 3],
        &[36, 9, 3, 1],
        16,
    )?;
    let dy = Tensor::create_with_id(
        BackendTensorUid::new(82),
        DataType::F32,
        &[2, 4, 3, 3],
        &[36, 9, 3, 1],
        16,
    )?;
    let dbn_scale = Tensor::create_with_id(
        BackendTensorUid::new(83),
        DataType::F32,
        &[1, 4, 1, 1],
        &[4, 1, 1, 1],
        16,
    )?;
    let dbn_bias = Tensor::create_with_id(
        BackendTensorUid::new(84),
        DataType::F32,
        &[1, 4, 1, 1],
        &[4, 1, 1, 1],
        16,
    )?;
    let eq_dy_scale = Tensor::create_with_id(
        BackendTensorUid::new(85),
        DataType::F32,
        &[1, 4, 1, 1],
        &[4, 1, 1, 1],
        16,
    )?;
    let eq_x_scale = Tensor::create_with_id(
        BackendTensorUid::new(86),
        DataType::F32,
        &[1, 4, 1, 1],
        &[4, 1, 1, 1],
        16,
    )?;
    let eq_bias = Tensor::create_with_id(
        BackendTensorUid::new(87),
        DataType::F32,
        &[1, 4, 1, 1],
        &[4, 1, 1, 1],
        16,
    )?;

    let op = BatchNormalizationBackwardWeightsOperation::create(
        DataType::F32,
        BatchNormalizationBackwardWeightsInputs {
            input: &x,
            statistics: NormalizationSavedStatistics {
                mean: &mean,
                inv_variance: &inv_std,
            },
            output_gradient: &dy,
            scale: &bn_scale,
        },
        BatchNormalizationBackwardWeightsOutputs {
            scale_gradient: &dbn_scale,
            bias_gradient: &dbn_bias,
            equivalent_bias: &eq_bias,
            equivalent_output_gradient_scale: &eq_dy_scale,
            equivalent_input_scale: &eq_x_scale,
        },
    )?;

    assert!(op.descriptor().is_finalized());
    assert_eq!(
        op.descriptor().descriptor_type(),
        BackendDescriptorType::OperationBnBwdWeights,
    );

    Ok(())
}

#[test]
fn test_resample_descriptor_smoke_test() -> Result<()> {
    let descriptor = ResampleDescriptor::create(
        ResampleDescriptorSpec::new(
            ResampleMode::Bilinear,
            DataType::F32,
            ResampleGeometry::create(
                [Fraction::whole(0), Fraction::whole(0)],
                [Fraction::whole(0), Fraction::whole(0)],
                [Fraction::whole(2), Fraction::whole(2)],
                [Fraction::whole(2), Fraction::whole(2)],
            )?,
        )
        .with_nan_propagation(NanPropagation::NotPropagate)
        .with_padding_mode(PaddingMode::Zero),
    )?;

    assert!(descriptor.descriptor().is_finalized());
    assert_eq!(
        descriptor.descriptor().descriptor_type(),
        BackendDescriptorType::Resample,
    );

    Ok(())
}

#[test]
fn test_resample_ops_use_scalar_alpha_beta_attributes() -> Result<()> {
    let _test = setup_context()?;

    let x = Tensor::create_with_id(
        BackendTensorUid::new(50),
        DataType::F32,
        &[2, 3, 8, 8],
        &[192, 64, 8, 1],
        16,
    )?;
    let y = Tensor::create_with_id(
        BackendTensorUid::new(51),
        DataType::F32,
        &[2, 3, 4, 4],
        &[48, 16, 4, 1],
        16,
    )?;
    let dy = Tensor::create_with_id(
        BackendTensorUid::new(52),
        DataType::F32,
        &[2, 3, 4, 4],
        &[48, 16, 4, 1],
        16,
    )?;
    let dx = Tensor::create_with_id(
        BackendTensorUid::new(53),
        DataType::F32,
        &[2, 3, 8, 8],
        &[192, 64, 8, 1],
        16,
    )?;
    let descriptor = ResampleDescriptor::create(
        ResampleDescriptorSpec::new(
            ResampleMode::MaxPool,
            DataType::F32,
            ResampleGeometry::create(
                [Fraction::whole(0), Fraction::whole(0)],
                [Fraction::whole(0), Fraction::whole(0)],
                [Fraction::whole(2), Fraction::whole(2)],
                [Fraction::whole(2), Fraction::whole(2)],
            )?,
        )
        .with_nan_propagation(NanPropagation::Propagate)
        .with_padding_mode(PaddingMode::Zero),
    )?;

    let forward = ResampleForwardOperation::create(
        &descriptor,
        &x,
        ResampleOutputs::new(&y, None),
        1.0,
        0.0,
    )?;
    assert!(forward.descriptor().is_finalized());

    let backward = ResampleBackwardOperation::create(
        &descriptor,
        ResampleBackwardInputs {
            output_gradient: &dy,
            input: Some(&x),
            output: Some(&y),
            indices: None,
        },
        &dx,
        1.0,
        0.0,
    )?;
    assert!(backward.descriptor().is_finalized());

    Ok(())
}

#[test]
fn test_reshape_op_smoke_test() -> Result<()> {
    let _test = setup_context()?;

    let x = Tensor::create_with_id(
        BackendTensorUid::new(40),
        DataType::F32,
        &[2, 3, 4],
        &[12, 4, 1],
        16,
    )?;
    let y = Tensor::create_with_id(
        BackendTensorUid::new(41),
        DataType::F32,
        &[2, 12],
        &[12, 1],
        16,
    )?;

    let op = ReshapeOperation::create(&x, &y, ReshapeSpec::new(ReshapeMode::ViewOnly))?;

    assert!(op.descriptor().is_finalized());
    assert_eq!(
        op.descriptor().descriptor_type(),
        BackendDescriptorType::OperationReshape,
    );

    Ok(())
}

#[test]
fn test_concat_op_smoke_test() -> Result<()> {
    let _test = setup_context()?;

    let x0 = Tensor::create_with_id(
        BackendTensorUid::new(42),
        DataType::F32,
        &[2, 3],
        &[3, 1],
        16,
    )?;
    let x1 = Tensor::create_with_id(
        BackendTensorUid::new(43),
        DataType::F32,
        &[2, 5],
        &[5, 1],
        16,
    )?;
    let y = Tensor::create_with_id(
        BackendTensorUid::new(44),
        DataType::F32,
        &[2, 8],
        &[8, 1],
        16,
    )?;

    let op = ConcatOperation::create(&[&x0, &x1], &y, ConcatSpec::new(1))?;

    assert!(op.descriptor().is_finalized());
    assert_eq!(
        op.descriptor().descriptor_type(),
        BackendDescriptorType::OperationConcat,
    );

    Ok(())
}

#[test]
fn test_concat_op_tracks_absent_inplace_index() -> Result<()> {
    let _test = setup_context()?;

    let x0 = Tensor::create_with_id(
        BackendTensorUid::new(42),
        DataType::F32,
        &[2, 3],
        &[3, 1],
        16,
    )?;
    let x1 = Tensor::create_with_id(
        BackendTensorUid::new(43),
        DataType::F32,
        &[2, 5],
        &[5, 1],
        16,
    )?;
    let y = Tensor::create_with_id(
        BackendTensorUid::new(44),
        DataType::F32,
        &[2, 8],
        &[8, 1],
        16,
    )?;

    let op = ConcatOperation::create(&[&x0, &x1], &y, ConcatSpec::new(1))?;

    assert_eq!(op.spec().in_place, ConcatInPlaceMode::None);

    Ok(())
}

#[test]
fn test_concat_op_tracks_explicit_inplace_index() -> Result<()> {
    let _test = setup_context()?;

    let x0 = Tensor::create_with_id(
        BackendTensorUid::new(42),
        DataType::F32,
        &[2, 3],
        &[3, 1],
        16,
    )?;
    let x1 = Tensor::create_with_id(
        BackendTensorUid::new(43),
        DataType::F32,
        &[2, 5],
        &[5, 1],
        16,
    )?;
    let y = Tensor::create_with_id(
        BackendTensorUid::new(44),
        DataType::F32,
        &[2, 8],
        &[8, 1],
        16,
    )?;

    let op = ConcatOperation::create(
        &[&x0, &x1],
        &y,
        ConcatSpec::new(1).with_in_place(ConcatInPlaceMode::Input(0)),
    )?;
    assert_eq!(op.spec().in_place, ConcatInPlaceMode::Input(0));

    let op = ConcatOperation::create(
        &[&x0, &x1],
        &y,
        ConcatSpec::new(1).with_in_place(ConcatInPlaceMode::Input(1)),
    )?;
    assert_eq!(op.spec().in_place, ConcatInPlaceMode::Input(1));

    Ok(())
}

#[test]
fn test_rng_descriptor_smoke_test() -> Result<()> {
    let descriptor = RngDescriptor::create(RandomNumberDistribution::Uniform {
        minimum: -1.0,
        maximum: 1.0,
    })?;

    assert!(descriptor.descriptor().is_finalized());
    assert_eq!(descriptor.distribution().kind(), RngDistribution::Uniform);
    assert_eq!(
        descriptor.descriptor().descriptor_type(),
        BackendDescriptorType::Rng,
    );

    Ok(())
}

#[test]
fn test_rng_op_smoke_test() {
    let error = RngDescriptor::create(RandomNumberDistribution::Normal {
        mean: 0.0,
        standard_deviation: -1.0,
    })
    .unwrap_err();
    assert!(matches!(
        error,
        Error::OutOfRange { name } if name == "standard_deviation"
    ));
}

#[test]
fn test_kernel_cache_smoke_test() -> Result<()> {
    let descriptor = KernelCache::create()?;

    assert_eq!(
        descriptor.descriptor().descriptor_type(),
        BackendDescriptorType::KernelCache,
    );
    assert!(!descriptor.descriptor().is_finalized());

    Ok(())
}

#[test]
fn test_kernel_cache_build_smoke_test() -> Result<()> {
    let test = setup_context()?;
    let context_ref = test.context();
    let x = Tensor::create_with_id(
        BackendTensorUid::new(1),
        DataType::F32,
        &[2, 3],
        &[3, 1],
        16,
    )?;
    let y = Tensor::create_with_id(
        BackendTensorUid::new(2),
        DataType::F32,
        &[2, 3],
        &[3, 1],
        16,
    )?;
    let pointwise = PointwiseDescriptor::create(
        PointwiseDescriptorSpec::new(PointwiseMode::ReluFwd, DataType::F32)
            .with_nan_propagation(NanPropagation::NotPropagate),
    )?;
    let op = PointwiseOperation::create(
        &pointwise,
        PointwiseSpec::Unary {
            x: &x,
            y: &y,
            alpha1: 1.0,
        },
    )?;
    let graph = OperationGraph::create(
        context_ref,
        &[op.descriptor()],
        OperationGraphOptions::default(),
    )?;

    let mut descriptor = KernelCache::create()?;
    descriptor.build(Some(&graph))?;

    assert!(descriptor.is_finalized());

    Ok(())
}

#[test]
fn test_matmul_descriptor_smoke_test() -> Result<()> {
    let descriptor = MatmulDescriptor::create(MatmulDescriptorSpec::new(DataType::F32))?;

    assert!(descriptor.descriptor().is_finalized());
    assert_eq!(
        descriptor.descriptor().descriptor_type(),
        BackendDescriptorType::Matmul,
    );

    Ok(())
}

#[test]
fn test_matmul_op_smoke_test() -> Result<()> {
    let _test = setup_context()?;

    let matmul = MatmulDescriptor::create(MatmulDescriptorSpec::new(DataType::F32))?;
    let a = Tensor::create_with_id(
        BackendTensorUid::new(20),
        DataType::F32,
        &[4, 8],
        &[8, 1],
        16,
    )?;
    let b = Tensor::create_with_id(
        BackendTensorUid::new(21),
        DataType::F32,
        &[8, 16],
        &[16, 1],
        16,
    )?;
    let c = Tensor::create_with_id(
        BackendTensorUid::new(22),
        DataType::F32,
        &[4, 16],
        &[16, 1],
        16,
    )?;

    let op = MatmulOperation::create(&matmul, &a, &b, &c, MatmulSpec::default())?;

    assert!(op.descriptor().is_finalized());
    assert_eq!(
        op.descriptor().descriptor_type(),
        BackendDescriptorType::OperationMatmul,
    );

    Ok(())
}

#[test]
fn test_sdpa_forward_op_smoke_test() -> Result<()> {
    let _test = setup_context()?;

    let q = Tensor::create_with_id(
        BackendTensorUid::new(30),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;
    let k = Tensor::create_with_id(
        BackendTensorUid::new(31),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;
    let v = Tensor::create_with_id(
        BackendTensorUid::new(32),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;
    let o = Tensor::create_with_id(
        BackendTensorUid::new(33),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;
    let scale = Tensor::create_by_value(BackendTensorUid::new(34), DataType::F32, &[1], &[1], 4)?;
    let stats = Tensor::create_virtual(
        BackendTensorUid::new(35),
        DataType::F32,
        &[2, 4, 8, 1],
        &[32, 8, 1, 1],
        16,
    )?;

    let op = SdpaForwardOperation::create(
        SdpaForwardTensors {
            query: &q,
            key: &k,
            value: &v,
            output: &o,
            scale: &scale,
        },
        SdpaForwardOptions {
            softmax: SdpaForwardSoftmax::Default {
                stats: Some(&stats),
            },
            ..Default::default()
        },
    )?;

    assert!(op.descriptor().is_finalized());
    assert_eq!(
        op.descriptor().descriptor_type(),
        BackendDescriptorType::OperationSdpaForward,
    );

    Ok(())
}

#[test]
fn test_sdpa_forward_op_accepts_softmax_descriptor() -> Result<()> {
    let _test = setup_context()?;

    let q = Tensor::create_with_id(
        BackendTensorUid::new(230),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;
    let k = Tensor::create_with_id(
        BackendTensorUid::new(231),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;
    let v = Tensor::create_with_id(
        BackendTensorUid::new(232),
        DataType::F16,
        &[2, 4, 8, 12],
        &[384, 96, 12, 1],
        16,
    )?;
    let o = Tensor::create_with_id(
        BackendTensorUid::new(233),
        DataType::F16,
        &[2, 4, 8, 12],
        &[384, 96, 12, 1],
        16,
    )?;
    let scale = Tensor::create_by_value(BackendTensorUid::new(234), DataType::F32, &[1], &[1], 4)?;
    let stats = Tensor::create_virtual(
        BackendTensorUid::new(235),
        DataType::F32,
        &[2, 4, 8, 1],
        &[32, 8, 1, 1],
        16,
    )?;
    let max = Tensor::create_virtual(
        BackendTensorUid::new(236),
        DataType::F32,
        &[2, 4, 8, 1],
        &[32, 8, 1, 1],
        16,
    )?;
    let sum_exp = Tensor::create_virtual(
        BackendTensorUid::new(237),
        DataType::F32,
        &[2, 4, 8, 1],
        &[32, 8, 1, 1],
        16,
    )?;
    let softmax_p = Tensor::create_virtual(
        BackendTensorUid::new(238),
        DataType::F16,
        &[2, 4, 8, 8],
        &[256, 64, 8, 1],
        16,
    )?;
    let softmax_s = Tensor::create_virtual(
        BackendTensorUid::new(239),
        DataType::F16,
        &[2, 4, 8, 8],
        &[256, 64, 8, 1],
        16,
    )?;
    let softmax = SoftmaxOperation::create(
        &softmax_p,
        SoftmaxOutputs {
            output: &softmax_s,
            stats: Some(&stats),
            max: Some(&max),
            sum_exp: Some(&sum_exp),
        },
        SoftmaxSpec {
            stats_base: SoftmaxStatsBase::NaturalLog,
            sink: None,
        },
    )?;

    let op = SdpaForwardOperation::create(
        SdpaForwardTensors {
            query: &q,
            key: &k,
            value: &v,
            output: &o,
            scale: &scale,
        },
        SdpaForwardOptions {
            softmax: SdpaForwardSoftmax::Descriptor(Arc::new(softmax)),
            ..Default::default()
        },
    )?;

    assert!(op.descriptor().is_finalized());
    assert_eq!(
        op.descriptor().descriptor_type(),
        BackendDescriptorType::OperationSdpaForward,
    );

    Ok(())
}

#[test]
fn test_sdpa_forward_op_accepts_optional_direct_attributes() -> Result<()> {
    let _test = setup_context()?;

    let q = Tensor::create_with_id(
        BackendTensorUid::new(130),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;
    let k = Tensor::create_with_id(
        BackendTensorUid::new(131),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;
    let v = Tensor::create_with_id(
        BackendTensorUid::new(132),
        DataType::F16,
        &[2, 4, 8, 12],
        &[384, 96, 12, 1],
        16,
    )?;
    let o = Tensor::create_with_id(
        BackendTensorUid::new(133),
        DataType::F16,
        &[2, 4, 8, 12],
        &[384, 96, 12, 1],
        16,
    )?;
    let scale = Tensor::create_by_value(BackendTensorUid::new(134), DataType::F32, &[1], &[1], 4)?;
    let stats = Tensor::create_virtual(
        BackendTensorUid::new(135),
        DataType::F32,
        &[2, 4, 8, 1],
        &[32, 8, 1, 1],
        16,
    )?;
    let sequence_length_query = Tensor::create_with_id(
        BackendTensorUid::new(136),
        DataType::I32,
        &[2, 1, 1, 1],
        &[1, 1, 1, 1],
        4,
    )?;
    let sequence_length_key_value = Tensor::create_with_id(
        BackendTensorUid::new(137),
        DataType::I32,
        &[2, 1, 1, 1],
        &[1, 1, 1, 1],
        4,
    )?;
    let seed = Tensor::create_by_value(
        BackendTensorUid::new(138),
        DataType::I64,
        &[1, 1, 1, 1],
        &[1, 1, 1, 1],
        8,
    )?;
    let offset = Tensor::create_by_value(
        BackendTensorUid::new(139),
        DataType::I64,
        &[1, 1, 1, 1],
        &[1, 1, 1, 1],
        8,
    )?;
    let rng_dump = Tensor::create_virtual(
        BackendTensorUid::new(140),
        DataType::F16,
        &[2, 4, 8, 8],
        &[256, 64, 8, 1],
        16,
    )?;

    let op = SdpaForwardOperation::create(
        SdpaForwardTensors {
            query: &q,
            key: &k,
            value: &v,
            output: &o,
            scale: &scale,
        },
        SdpaForwardOptions {
            sequence: AttentionSequenceInputs::per_batch(
                &sequence_length_query,
                &sequence_length_key_value,
            ),
            page_table_k: None,
            page_table_v: None,
            block_mask: None,
            softmax: SdpaForwardSoftmax::Default {
                stats: Some(&stats),
            },
            subgraph: None,
            dropout: Some(SdpaForwardDropout {
                probability: 0.25_f32,
                seed: &seed,
                offset: &offset,
                random_number_generator_dump: Some(&rng_dump),
            }),
            unfuse_fma: true,
            quantization: None,
        },
    )?;

    assert!(op.descriptor().is_finalized());
    assert_eq!(
        op.descriptor().descriptor_type(),
        BackendDescriptorType::OperationSdpaForward,
    );

    Ok(())
}

#[test]
fn test_sdpa_forward_op_accepts_block_mask() -> Result<()> {
    let _test = setup_context()?;

    let q = Tensor::create_with_id(
        BackendTensorUid::new(330),
        DataType::F16,
        &[1, 2, 4, 16],
        &[128, 64, 16, 1],
        16,
    )?;
    let k = Tensor::create_with_id(
        BackendTensorUid::new(331),
        DataType::F16,
        &[1, 2, 4, 16],
        &[128, 64, 16, 1],
        16,
    )?;
    let v = Tensor::create_with_id(
        BackendTensorUid::new(332),
        DataType::F16,
        &[1, 2, 4, 12],
        &[96, 48, 12, 1],
        16,
    )?;
    let o = Tensor::create_with_id(
        BackendTensorUid::new(333),
        DataType::F16,
        &[1, 2, 4, 12],
        &[96, 48, 12, 1],
        16,
    )?;
    let scale = Tensor::create_by_value(BackendTensorUid::new(334), DataType::F32, &[1], &[1], 4)?;
    let block_mask = Tensor::create_with_id(
        BackendTensorUid::new(335),
        DataType::U8,
        &[1, 2, 1, 1],
        &[2, 1, 1, 1],
        1,
    )?;

    let op = SdpaForwardOperation::create(
        SdpaForwardTensors {
            query: &q,
            key: &k,
            value: &v,
            output: &o,
            scale: &scale,
        },
        SdpaForwardOptions {
            block_mask: Some(&block_mask),
            ..Default::default()
        },
    )?;

    assert!(op.descriptor().is_finalized());
    assert_eq!(
        op.descriptor().descriptor_type(),
        BackendDescriptorType::OperationSdpaForward,
    );

    Ok(())
}

#[test]
fn test_sdpa_forward_op_accepts_subgraph_descriptor() -> Result<()> {
    let test = setup_context()?;
    let context_ref = test.context();

    let q = Tensor::create_with_id(
        BackendTensorUid::new(430),
        DataType::F16,
        &[1, 2, 4, 16],
        &[128, 64, 16, 1],
        16,
    )?;
    let k = Tensor::create_with_id(
        BackendTensorUid::new(431),
        DataType::F16,
        &[1, 2, 4, 16],
        &[128, 64, 16, 1],
        16,
    )?;
    let v = Tensor::create_with_id(
        BackendTensorUid::new(432),
        DataType::F16,
        &[1, 2, 4, 12],
        &[96, 48, 12, 1],
        16,
    )?;
    let o = Tensor::create_with_id(
        BackendTensorUid::new(433),
        DataType::F16,
        &[1, 2, 4, 12],
        &[96, 48, 12, 1],
        16,
    )?;
    let scale = Tensor::create_by_value(BackendTensorUid::new(434), DataType::F32, &[1], &[1], 4)?;
    let stats = Tensor::create_virtual(
        BackendTensorUid::new(435),
        DataType::F32,
        &[1, 2, 4, 1],
        &[8, 4, 1, 1],
        16,
    )?;

    let subgraph_input = Tensor::create_virtual(
        BackendTensorUid::new(530),
        DataType::F16,
        &[1, 2, 4, 4],
        &[32, 16, 4, 1],
        16,
    )?;
    let subgraph_output = Tensor::create_virtual(
        BackendTensorUid::new(531),
        DataType::F16,
        &[1, 2, 4, 4],
        &[32, 16, 4, 1],
        16,
    )?;
    let pointwise = PointwiseDescriptor::create(
        PointwiseDescriptorSpec::new(PointwiseMode::ReluFwd, DataType::F16)
            .with_nan_propagation(NanPropagation::NotPropagate),
    )?;
    let subgraph_op = PointwiseOperation::create(
        &pointwise,
        PointwiseSpec::Unary {
            x: &subgraph_input,
            y: &subgraph_output,
            alpha1: 1.0,
        },
    )?;
    let subgraph = OperationGraph::create(
        context_ref,
        &[subgraph_op.descriptor()],
        OperationGraphOptions::default(),
    )?;

    let op = SdpaForwardOperation::create(
        SdpaForwardTensors {
            query: &q,
            key: &k,
            value: &v,
            output: &o,
            scale: &scale,
        },
        SdpaForwardOptions {
            softmax: SdpaForwardSoftmax::Default {
                stats: Some(&stats),
            },
            subgraph: Some(SdpaSubgraph {
                operation_graph: &subgraph,
                input_id: subgraph_input.id(),
                output_id: subgraph_output.id(),
            }),
            ..Default::default()
        },
    )?;

    assert!(op.descriptor().is_finalized());
    assert_eq!(
        op.descriptor().descriptor_type(),
        BackendDescriptorType::OperationSdpaForward,
    );

    Ok(())
}

#[test]
fn test_sdpa_backward_op_smoke_test() -> Result<()> {
    let _test = setup_context()?;

    let q = Tensor::create_with_id(
        BackendTensorUid::new(40),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;
    let k = Tensor::create_with_id(
        BackendTensorUid::new(41),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;
    let v = Tensor::create_with_id(
        BackendTensorUid::new(42),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;
    let o = Tensor::create_with_id(
        BackendTensorUid::new(43),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;
    let stats = Tensor::create_virtual(
        BackendTensorUid::new(44),
        DataType::F32,
        &[2, 4, 8, 1],
        &[32, 8, 1, 1],
        16,
    )?;
    let scale = Tensor::create_by_value(BackendTensorUid::new(45), DataType::F32, &[1], &[1], 4)?;
    let d_o = Tensor::create_with_id(
        BackendTensorUid::new(46),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;
    let d_q = Tensor::create_with_id(
        BackendTensorUid::new(47),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;
    let d_k = Tensor::create_with_id(
        BackendTensorUid::new(48),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;
    let d_v = Tensor::create_with_id(
        BackendTensorUid::new(49),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;

    let op = SdpaBackwardOperation::create(
        SdpaBackwardTensors {
            query: &q,
            key: &k,
            value: &v,
            output: &o,
            stats: &stats,
            scale: &scale,
            output_gradient: &d_o,
            query_gradient: &d_q,
            key_gradient: &d_k,
            value_gradient: &d_v,
        },
        SdpaBackwardOptions::default(),
    )?;

    assert!(op.descriptor().is_finalized());
    assert_eq!(
        op.descriptor().descriptor_type(),
        BackendDescriptorType::OperationSdpaBackward,
    );

    Ok(())
}

#[test]
fn test_sdpa_backward_op_accepts_optional_direct_attributes() -> Result<()> {
    let _test = setup_context()?;

    let q = Tensor::create_with_id(
        BackendTensorUid::new(150),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;
    let k = Tensor::create_with_id(
        BackendTensorUid::new(151),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;
    let v = Tensor::create_with_id(
        BackendTensorUid::new(152),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;
    let o = Tensor::create_with_id(
        BackendTensorUid::new(153),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;
    let stats = Tensor::create_virtual(
        BackendTensorUid::new(154),
        DataType::F32,
        &[2, 4, 8, 1],
        &[32, 8, 1, 1],
        16,
    )?;
    let scale = Tensor::create_by_value(BackendTensorUid::new(155), DataType::F32, &[1], &[1], 4)?;
    let d_o = Tensor::create_with_id(
        BackendTensorUid::new(156),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;
    let d_q = Tensor::create_with_id(
        BackendTensorUid::new(157),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;
    let d_k = Tensor::create_with_id(
        BackendTensorUid::new(158),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;
    let d_v = Tensor::create_with_id(
        BackendTensorUid::new(159),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;
    let sequence_length_query = Tensor::create_with_id(
        BackendTensorUid::new(160),
        DataType::I32,
        &[2, 1, 1, 1],
        &[1, 1, 1, 1],
        4,
    )?;
    let sequence_length_key_value = Tensor::create_with_id(
        BackendTensorUid::new(161),
        DataType::I32,
        &[2, 1, 1, 1],
        &[1, 1, 1, 1],
        4,
    )?;
    let sink = Tensor::create_with_id(
        BackendTensorUid::new(162),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;
    let sink_gradient = Tensor::create_with_id(
        BackendTensorUid::new(163),
        DataType::F16,
        &[2, 4, 8, 16],
        &[512, 128, 16, 1],
        16,
    )?;

    let op = SdpaBackwardOperation::create(
        SdpaBackwardTensors {
            query: &q,
            key: &k,
            value: &v,
            output: &o,
            stats: &stats,
            scale: &scale,
            output_gradient: &d_o,
            query_gradient: &d_q,
            key_gradient: &d_k,
            value_gradient: &d_v,
        },
        SdpaBackwardOptions {
            sequence: AttentionSequenceInputs::per_batch(
                &sequence_length_query,
                &sequence_length_key_value,
            ),
            subgraph: None,
            sink: Some(SdpaBackwardSink::new(&sink).with_gradient(&sink_gradient)),
            max_total_sequence_length_query: Some(128),
            max_total_sequence_length_key_value: Some(256),
        },
    )?;

    assert!(op.descriptor().is_finalized());
    assert_eq!(
        op.descriptor().descriptor_type(),
        BackendDescriptorType::OperationSdpaBackward,
    );

    Ok(())
}

#[test]
fn test_diagonal_band_mask_op_smoke_test() -> Result<()> {
    let _test = setup_context()?;

    let x = Tensor::create_with_id(
        BackendTensorUid::new(60),
        DataType::F32,
        &[2, 4, 8, 8],
        &[256, 64, 8, 1],
        16,
    )?;
    let b = Tensor::create_by_value(BackendTensorUid::new(61), DataType::F32, &[1], &[1], 4)?;
    let sequence_length_query = Tensor::create_with_id(
        BackendTensorUid::new(62),
        DataType::I32,
        &[2, 1, 1, 1],
        &[1, 1, 1, 1],
        16,
    )?;
    let sequence_length_key_value = Tensor::create_with_id(
        BackendTensorUid::new(63),
        DataType::I32,
        &[2, 1, 1, 1],
        &[1, 1, 1, 1],
        16,
    )?;
    let shift_right_bound =
        Tensor::create_by_value(BackendTensorUid::new(64), DataType::I32, &[1], &[1], 4)?;
    let y = Tensor::create_with_id(
        BackendTensorUid::new(65),
        DataType::F32,
        &[2, 4, 8, 8],
        &[256, 64, 8, 1],
        16,
    )?;

    let op = DiagonalBandMaskOperation::create(
        &x,
        &b,
        &y,
        DiagonalBandMaskSpec::new(PointwiseMode::CmpGe)
            .with_sequence_inputs(AttentionSequenceInputs::per_batch(
                &sequence_length_query,
                &sequence_length_key_value,
            ))
            .with_bound(DiagonalBandMaskBound::Right {
                shift: Some(&shift_right_bound),
            }),
    )?;

    assert!(op.descriptor().is_finalized());
    assert_eq!(
        op.descriptor().descriptor_type(),
        BackendDescriptorType::OperationDiagonalBandMask,
    );

    Ok(())
}

#[test]
fn test_moe_grouped_matmul_op_smoke_test() -> Result<()> {
    let _test = setup_context()?;

    let token = Tensor::create_with_id(
        BackendTensorUid::new(200),
        DataType::F16,
        &[1, 2048, 512],
        &[2048 * 512, 512, 1],
        16,
    )?;
    let weight = Tensor::create_with_id(
        BackendTensorUid::new(201),
        DataType::F16,
        &[3, 512, 256],
        &[512 * 256, 1, 512],
        16,
    )?;
    let first_token_offset = Tensor::create_with_id(
        BackendTensorUid::new(202),
        DataType::I32,
        &[6, 1, 1],
        &[1, 1, 1],
        4,
    )?;
    let output = Tensor::create_with_id(
        BackendTensorUid::new(203),
        DataType::F16,
        &[1, 2048, 256],
        &[2048 * 256, 256, 1],
        16,
    )?;

    let _op = MoeGroupedMatmulOperation::create(
        MoeGroupedMatmulRouting::None,
        DataType::F16,
        MoeGroupedMatmulTensors::new(
            MoeGroupedMatmulInputs::new(&token, &weight, &first_token_offset),
            &output,
        ),
    )?;

    Ok(())
}

#[test]
fn test_convolution_backward_data_op_smoke_test() -> Result<()> {
    let _test = setup_context()?;

    let convolution = ConvolutionDescriptor::create(ConvolutionDescriptorSpec::new(
        DataType::F32,
        ConvolutionMode::CrossCorrelation,
        ConvolutionGeometry::create([0, 0], [0, 0], [1, 1], [1, 1])?,
    ))?;
    let w = Tensor::create_with_id(
        BackendTensorUid::new(1),
        DataType::F32,
        &[4, 3, 3, 3],
        &[27, 9, 3, 1],
        16,
    )?;
    let output_gradient = Tensor::create_with_id(
        BackendTensorUid::new(2),
        DataType::F32,
        &[1, 4, 8, 8],
        &[256, 64, 8, 1],
        16,
    )?;
    let input_gradient = Tensor::create_with_id(
        BackendTensorUid::new(3),
        DataType::F32,
        &[1, 3, 10, 10],
        &[300, 100, 10, 1],
        16,
    )?;

    let op = ConvolutionBackwardDataOperation::create(
        &convolution,
        &w,
        &output_gradient,
        &input_gradient,
        1.0,
        0.0,
    )?;

    assert!(op.descriptor().is_finalized());
    assert_eq!(
        op.descriptor().descriptor_type(),
        BackendDescriptorType::OperationConvolutionBackwardData,
    );

    Ok(())
}

#[test]
fn test_convolution_backward_filter_op_smoke_test() -> Result<()> {
    let _test = setup_context()?;

    let convolution = ConvolutionDescriptor::create(ConvolutionDescriptorSpec::new(
        DataType::F32,
        ConvolutionMode::CrossCorrelation,
        ConvolutionGeometry::create([0, 0], [0, 0], [1, 1], [1, 1])?,
    ))?;
    let x = Tensor::create_with_id(
        BackendTensorUid::new(4),
        DataType::F32,
        &[1, 3, 10, 10],
        &[300, 100, 10, 1],
        16,
    )?;
    let output_gradient = Tensor::create_with_id(
        BackendTensorUid::new(5),
        DataType::F32,
        &[1, 4, 8, 8],
        &[256, 64, 8, 1],
        16,
    )?;
    let filter_gradient = Tensor::create_with_id(
        BackendTensorUid::new(6),
        DataType::F32,
        &[4, 3, 3, 3],
        &[27, 9, 3, 1],
        16,
    )?;

    let op = ConvolutionBackwardFilterOperation::create(
        &convolution,
        &x,
        &output_gradient,
        &filter_gradient,
        1.0,
        0.0,
    )?;

    assert!(op.descriptor().is_finalized());
    assert_eq!(
        op.descriptor().descriptor_type(),
        BackendDescriptorType::OperationConvolutionBackwardFilter,
    );

    Ok(())
}

#[test]
fn test_norm_forward_op_smoke_test() -> Result<()> {
    let _test = setup_context()?;

    let norm = NormalizationForwardSpec::new(
        BackendNormalizationMode::Layer,
        BackendNormalizationForwardPhase::Inference,
    );
    let x = Tensor::create_with_id(
        BackendTensorUid::new(10),
        DataType::F32,
        &[2, 8],
        &[8, 1],
        16,
    )?;
    let mean = Tensor::create_with_id(
        BackendTensorUid::new(11),
        DataType::F32,
        &[2, 1],
        &[1, 1],
        16,
    )?;
    let inv_variance = Tensor::create_with_id(
        BackendTensorUid::new(12),
        DataType::F32,
        &[2, 1],
        &[1, 1],
        16,
    )?;
    let scale = Tensor::create_with_id(
        BackendTensorUid::new(13),
        DataType::F32,
        &[1, 8],
        &[8, 1],
        16,
    )?;
    let bias = Tensor::create_with_id(
        BackendTensorUid::new(14),
        DataType::F32,
        &[1, 8],
        &[8, 1],
        16,
    )?;
    let epsilon = Tensor::create_by_value(BackendTensorUid::new(15), DataType::F32, &[1], &[1], 4)?;
    let y = Tensor::create_with_id(
        BackendTensorUid::new(16),
        DataType::F32,
        &[2, 8],
        &[8, 1],
        16,
    )?;

    let op = NormalizationForwardOperation::create(
        &norm,
        NormalizationForwardTensors {
            mean: Some(&mean),
            inv_variance: Some(&inv_variance),
            scale: Some(&scale),
            bias: Some(&bias),
            epsilon: Some(&epsilon),
            ..NormalizationForwardTensors::new(&x, &y)
        },
    )?;

    assert!(op.descriptor().is_finalized());
    assert_eq!(
        op.descriptor().descriptor_type(),
        BackendDescriptorType::OperationNormForward,
    );

    Ok(())
}

#[test]
fn test_norm_backward_op_smoke_test() -> Result<()> {
    let _test = setup_context()?;

    let norm = NormalizationBackwardSpec::new(BackendNormalizationMode::Layer);
    let x = Tensor::create_with_id(
        BackendTensorUid::new(50),
        DataType::F32,
        &[2, 8],
        &[8, 1],
        16,
    )?;
    let mean = Tensor::create_with_id(
        BackendTensorUid::new(51),
        DataType::F32,
        &[2, 1],
        &[1, 1],
        16,
    )?;
    let inv_variance = Tensor::create_with_id(
        BackendTensorUid::new(52),
        DataType::F32,
        &[2, 1],
        &[1, 1],
        16,
    )?;
    let dy = Tensor::create_with_id(
        BackendTensorUid::new(53),
        DataType::F32,
        &[2, 8],
        &[8, 1],
        16,
    )?;
    let scale = Tensor::create_with_id(
        BackendTensorUid::new(54),
        DataType::F32,
        &[1, 8],
        &[8, 1],
        16,
    )?;
    let epsilon = Tensor::create_by_value(BackendTensorUid::new(55), DataType::F32, &[1], &[1], 4)?;
    let dscale = Tensor::create_with_id(
        BackendTensorUid::new(56),
        DataType::F32,
        &[1, 8],
        &[8, 1],
        16,
    )?;
    let bias_gradient = Tensor::create_with_id(
        BackendTensorUid::new(57),
        DataType::F32,
        &[1, 8],
        &[8, 1],
        16,
    )?;
    let dx = Tensor::create_with_id(
        BackendTensorUid::new(58),
        DataType::F32,
        &[2, 8],
        &[8, 1],
        16,
    )?;

    let op = NormalizationBackwardOperation::create(
        &norm,
        NormalizationBackwardTensors {
            mean: Some(&mean),
            scale: Some(&scale),
            epsilon: Some(&epsilon),
            scale_gradient: Some(&dscale),
            bias_gradient: Some(&bias_gradient),
            ..NormalizationBackwardTensors::new(&x, &dy, &dx, &inv_variance)
        },
    )?;

    assert!(op.descriptor().is_finalized());
    assert_eq!(
        op.descriptor().descriptor_type(),
        BackendDescriptorType::OperationNormBackward,
    );

    Ok(())
}
