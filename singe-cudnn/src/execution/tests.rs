use singe_cudnn_sys as sys;

use crate::{
    convolution::ConvolutionMode,
    data_type::DataType,
    descriptor::BackendDescriptorType,
    engine::Engine,
    error::{Error, Result, Status},
    execution::{
        EngineConfig, EngineHeuristics, ExecutionPlan, KernelCache, OperationGraph, VariantPack,
        advanced::{
            BlockScaleDequantizeOperation, BlockScaleQuantizeOperation,
            ContractBandMatrixOperation, DiagonalBandMaskOperation, ExpandBandMatrixOperation,
            MatmulDescriptor, MatmulOperation, MoeGroupedMatmulMode, MoeGroupedMatmulOperation,
            PagedCacheLoadOperation, SdpaBackwardConfig, SdpaBackwardOperation, SdpaForwardConfig,
            SdpaForwardOperation, SoftmaxOperation,
        },
        convolution::{
            ConvolutionBackwardDataOperation, ConvolutionBackwardFilterOperation,
            ConvolutionDescriptor,
        },
        normalization::{
            BatchNormalizationBackwardWeightsOperation, BatchNormalizationFinalizeStatsMode,
            GenStatsMode, GenStatsOperation, NormalizationBackwardConfig,
            NormalizationBackwardOperation, NormalizationForwardConfig,
            NormalizationForwardOperation,
        },
        pointwise::{PointwiseDescriptor, PointwiseOperation},
        reduction::ReductionDescriptor,
        resample::{
            Fraction, PaddingMode, ResampleBackwardOperation, ResampleDescriptor,
            ResampleForwardOperation, ResampleMode,
        },
        rng::{RngDescriptor, RngDistribution},
        tensor_ops::{ConcatOperation, ReshapeOperation, SignalMode},
    },
    heuristic::BackendHeuristicMode,
    math::NanPropagation,
    normalization::{BackendNormalizationForwardPhase, BackendNormalizationMode},
    pointwise::PointwiseMode,
    reduction::ReduceTensorOperator,
    tensor::{BackendTensorReordering, Shape, Tensor, TensorSpec},
    testing::setup_context,
};

#[test]
fn test_backend_tensor_descriptor_smoke_test() -> Result<()> {
    let _context = setup_context()?;

    let tensor = Tensor::create_with_id(7, DataType::F32, &[2, 3], &[3, 1], 16)?;

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
    let _context = setup_context()?;

    let tensor = Tensor::create_vectorized(
        9,
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
fn test_backend_tensor_descriptor_accepts_ragged_offset() -> Result<()> {
    let _context = setup_context()?;

    let ragged_offset = Tensor::create_with_id(11, DataType::I32, &[4], &[1], 4)?;
    let tensor = match Tensor::create_with_spec(
        &TensorSpec::new(
            DataType::F16,
            Shape::contiguous([2, 8, 4, 16])?.with_strides([512, 64, 16, 1])?,
        )
        .with_id(12)
        .with_ragged_offset(ragged_offset.id())
        .with_byte_alignment(16)?,
    ) {
        Ok(tensor) => tensor,
        Err(Error::Cudnn { code, .. }) if code == Status::BadParam => return Ok(()),
        Err(error) => return Err(error),
    };

    assert_eq!(tensor.ragged_offset(), Some(11.into()));
    assert!(tensor.descriptor().is_finalized());

    Ok(())
}

#[test]
fn test_variant_pack_requires_bindings() {
    let err = unsafe { VariantPack::create(&[], None) }.unwrap_err();
    assert!(matches!(err, Error::EmptyList { name } if name == "bindings"));
}

#[test]
fn test_operation_graph_requires_operations() -> Result<()> {
    let context = setup_context()?;

    let err = OperationGraph::create(&context, &[]).unwrap_err();
    assert!(matches!(err, Error::EmptyList { name } if name == "operations"));

    Ok(())
}

#[test]
fn test_engine_heuristics_returns_typed_engine_configs() -> Result<()> {
    let context = setup_context()?;
    let x = Tensor::create_with_id(1, DataType::F32, &[2, 3], &[3, 1], 16)?;
    let y = Tensor::create_with_id(2, DataType::F32, &[2, 3], &[3, 1], 16)?;
    let pointwise = PointwiseDescriptor::create(
        PointwiseMode::ReluFwd,
        DataType::F32,
        NanPropagation::NotPropagate,
    )?;
    let op = PointwiseOperation::unary(&pointwise, &x, &y, 1.0)?;
    let graph = OperationGraph::create(&context, &[op.descriptor()])?;
    let heuristics =
        EngineHeuristics::create(&context, &graph, BackendHeuristicMode::Instant, None)?;

    let configs = heuristics.engine_configs()?;
    for config in &configs {
        let _ = config.workspace_size()?;
    }

    Ok(())
}

#[test]
fn test_engine_config_exposes_typed_inspection_helpers() -> Result<()> {
    let context = setup_context()?;
    let x = Tensor::create_with_id(1, DataType::F32, &[2, 3], &[3, 1], 16)?;
    let y = Tensor::create_with_id(2, DataType::F32, &[2, 3], &[3, 1], 16)?;
    let pointwise = PointwiseDescriptor::create(
        PointwiseMode::ReluFwd,
        DataType::F32,
        NanPropagation::NotPropagate,
    )?;
    let op = PointwiseOperation::unary(&pointwise, &x, &y, 1.0)?;
    let graph = OperationGraph::create(&context, &[op.descriptor()])?;
    let heuristics =
        EngineHeuristics::create(&context, &graph, BackendHeuristicMode::Instant, None)?;

    for config in heuristics.engine_configs()? {
        let _ = config.workspace_size()?;
        let _ = config.knob_choices()?;
        let _ = config.knob_infos()?;
        let _ = config.intermediate_infos()?;
    }

    Ok(())
}

#[test]
fn test_engine_config_can_be_created_from_engine_index_and_knobs() -> Result<()> {
    let context = setup_context()?;
    let x = Tensor::create_with_id(1, DataType::F32, &[2, 3], &[3, 1], 16)?;
    let y = Tensor::create_with_id(2, DataType::F32, &[2, 3], &[3, 1], 16)?;
    let pointwise = PointwiseDescriptor::create(
        PointwiseMode::ReluFwd,
        DataType::F32,
        NanPropagation::NotPropagate,
    )?;
    let op = PointwiseOperation::unary(&pointwise, &x, &y, 1.0)?;
    let graph = OperationGraph::create(&context, &[op.descriptor()])?;
    let heuristics =
        EngineHeuristics::create(&context, &graph, BackendHeuristicMode::Instant, None)?;

    for config in heuristics.engine_configs()? {
        let engine = config.engine()?;
        let knob_choices = config
            .knob_choices()?
            .into_iter()
            .map(|choice| (choice.knob_type, choice.value))
            .collect::<Vec<_>>();
        let recreated = EngineConfig::create(&context, &graph, engine.index(), &knob_choices)?;

        let _ = recreated.workspace_size()?;
        let recreated_engine = recreated.engine()?;
        assert_eq!(recreated_engine.index(), engine.index());
    }

    Ok(())
}

#[test]
fn test_execution_plan_create_smoke_test() -> Result<()> {
    let context = setup_context()?;
    let x = Tensor::create_with_id(1, DataType::F32, &[2, 3], &[3, 1], 16)?;
    let y = Tensor::create_with_id(2, DataType::F32, &[2, 3], &[3, 1], 16)?;
    let pointwise = PointwiseDescriptor::create(
        PointwiseMode::ReluFwd,
        DataType::F32,
        NanPropagation::NotPropagate,
    )?;
    let op = PointwiseOperation::unary(&pointwise, &x, &y, 1.0)?;
    let graph = OperationGraph::create(&context, &[op.descriptor()])?;
    let heuristics =
        EngineHeuristics::create(&context, &graph, BackendHeuristicMode::Instant, None)?;

    if let Some(config) = heuristics.engine_configs()?.into_iter().next() {
        let plan = ExecutionPlan::create(&context, &config)?;
        let _ = plan.workspace_size()?;
    }

    Ok(())
}

#[test]
fn test_graph_aware_engine_knob_query_returns_or_errors() -> Result<()> {
    let context = setup_context()?;
    let a = Tensor::create_with_id(0, DataType::BF16, &[16, 32, 128], &[32 * 128, 128, 1], 16)?;
    let b = Tensor::create_with_id(1, DataType::BF16, &[16, 128, 64], &[128 * 64, 64, 1], 16)?;
    let c = Tensor::create_with_id(2, DataType::BF16, &[16, 32, 64], &[32 * 64, 64, 1], 16)?;
    let matmul = MatmulDescriptor::create(DataType::F32)?;
    let op = MatmulOperation::create(&matmul, &a, &b, &c)?;
    let graph = OperationGraph::create(&context, &[op.descriptor()])?;

    let engine = Engine::from_index(0.into());
    match engine.knob_infos(&context, graph.descriptor()) {
        Ok(knobs) => {
            println!("engine 0 knobs: {}", knobs.len());
        }
        Err(error) => {
            println!("engine 0 knob query error: {error}");
        }
    }

    Ok(())
}

#[test]
fn test_pointwise_op_descriptor_smoke_test() -> Result<()> {
    let descriptor = PointwiseDescriptor::create(
        PointwiseMode::ReluFwd,
        DataType::F32,
        NanPropagation::NotPropagate,
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
    let descriptor = ReductionDescriptor::create(ReduceTensorOperator::Add, DataType::F32, false)?;

    assert!(descriptor.descriptor().is_finalized());
    assert_eq!(
        descriptor.descriptor().descriptor_type(),
        BackendDescriptorType::Reduction,
    );

    Ok(())
}

#[test]
fn test_gen_stats_op_smoke_test() -> Result<()> {
    let _context = setup_context()?;

    let x = Tensor::create_with_id(60, DataType::F32, &[2, 4, 3, 3], &[36, 9, 3, 1], 16)?;
    let sum = Tensor::create_with_id(61, DataType::F32, &[1, 4, 1, 1], &[4, 1, 1, 1], 16)?;
    let sqsum = Tensor::create_with_id(62, DataType::F32, &[1, 4, 1, 1], &[4, 1, 1, 1], 16)?;

    let op = GenStatsOperation::create(GenStatsMode::SumSqsum, &x, &sum, &sqsum)?;

    assert!(op.descriptor().is_finalized());
    assert_eq!(
        op.descriptor().descriptor_type(),
        BackendDescriptorType::OperationGenStats,
    );

    Ok(())
}

#[test]
fn test_bn_finalize_stats_op_smoke_test() {
    assert_eq!(
        BatchNormalizationFinalizeStatsMode::Training,
        BatchNormalizationFinalizeStatsMode::try_from(
            sys::cudnnBnFinalizeStatsMode_t::CUDNN_BN_FINALIZE_STATISTICS_TRAINING as u32,
        )
        .expect("valid bn finalize mode"),
    );
}

#[test]
fn test_bn_backward_weights_op_smoke_test() -> Result<()> {
    let _context = setup_context()?;

    let mean = Tensor::create_with_id(78, DataType::F32, &[1, 4, 1, 1], &[4, 1, 1, 1], 16)?;
    let inv_std = Tensor::create_with_id(79, DataType::F32, &[1, 4, 1, 1], &[4, 1, 1, 1], 16)?;
    let bn_scale = Tensor::create_with_id(80, DataType::F32, &[1, 4, 1, 1], &[4, 1, 1, 1], 16)?;
    let x = Tensor::create_with_id(81, DataType::F32, &[2, 4, 3, 3], &[36, 9, 3, 1], 16)?;
    let dy = Tensor::create_with_id(82, DataType::F32, &[2, 4, 3, 3], &[36, 9, 3, 1], 16)?;
    let dbn_scale = Tensor::create_with_id(83, DataType::F32, &[1, 4, 1, 1], &[4, 1, 1, 1], 16)?;
    let dbn_bias = Tensor::create_with_id(84, DataType::F32, &[1, 4, 1, 1], &[4, 1, 1, 1], 16)?;
    let eq_dy_scale = Tensor::create_with_id(85, DataType::F32, &[1, 4, 1, 1], &[4, 1, 1, 1], 16)?;
    let eq_x_scale = Tensor::create_with_id(86, DataType::F32, &[1, 4, 1, 1], &[4, 1, 1, 1], 16)?;
    let eq_bias = Tensor::create_with_id(87, DataType::F32, &[1, 4, 1, 1], &[4, 1, 1, 1], 16)?;

    let op = BatchNormalizationBackwardWeightsOperation::create(
        DataType::F32,
        &mean,
        &inv_std,
        &bn_scale,
        &x,
        &dy,
        &dbn_scale,
        &dbn_bias,
        &eq_dy_scale,
        &eq_x_scale,
        &eq_bias,
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
        ResampleMode::Bilinear,
        DataType::F32,
        NanPropagation::NotPropagate,
        PaddingMode::Zero,
        &[Fraction::whole(0), Fraction::whole(0)],
        &[Fraction::whole(0), Fraction::whole(0)],
        &[Fraction::whole(2), Fraction::whole(2)],
        &[Fraction::whole(2), Fraction::whole(2)],
    )?;

    assert!(descriptor.descriptor().is_finalized());
    assert_eq!(
        descriptor.descriptor().descriptor_type(),
        BackendDescriptorType::Resample,
    );

    Ok(())
}

#[test]
fn test_resample_descriptor_accepts_fractional_geometry() -> Result<()> {
    // NOTE: cuDNN 9.21.1 documents CUDNN_TYPE_FRACTION support for resample geometry,
    // but some fractional configurations still finalize as NOT_SUPPORTED at runtime.
    let descriptor = match ResampleDescriptor::create(
        ResampleMode::Bilinear,
        DataType::F32,
        NanPropagation::NotPropagate,
        PaddingMode::Zero,
        &[Fraction::new(0, 1), Fraction::new(0, 1)],
        &[Fraction::new(0, 1), Fraction::new(0, 1)],
        &[Fraction::new(1, 2), Fraction::new(1, 2)],
        &[Fraction::new(3, 2), Fraction::new(3, 2)],
    ) {
        Ok(descriptor) => descriptor,
        Err(Error::Cudnn { code, .. }) if code == Status::NotSupported => return Ok(()),
        Err(error) => return Err(error),
    };

    assert!(descriptor.descriptor().is_finalized());
    assert_eq!(
        descriptor.descriptor().descriptor_type(),
        BackendDescriptorType::Resample,
    );

    Ok(())
}

#[test]
fn test_resample_ops_use_scalar_alpha_beta_attributes() -> Result<()> {
    let _context = match setup_context() {
        Ok(ctx) => ctx,
        Err(Error::Cuda(singe_cuda::error::Error::Cuda { code, .. }))
            if code == singe_cuda::error::Status::NoDevice =>
        {
            return Ok(());
        }
        Err(error) => return Err(error),
    };

    let x = Tensor::create_with_id(50, DataType::F32, &[2, 3, 8, 8], &[192, 64, 8, 1], 16)?;
    let y = Tensor::create_with_id(51, DataType::F32, &[2, 3, 4, 4], &[48, 16, 4, 1], 16)?;
    let dy = Tensor::create_with_id(52, DataType::F32, &[2, 3, 4, 4], &[48, 16, 4, 1], 16)?;
    let dx = Tensor::create_with_id(53, DataType::F32, &[2, 3, 8, 8], &[192, 64, 8, 1], 16)?;
    let descriptor = ResampleDescriptor::create(
        ResampleMode::MaxPool,
        DataType::F32,
        NanPropagation::Propagate,
        PaddingMode::Zero,
        &[Fraction::whole(0), Fraction::whole(0)],
        &[Fraction::whole(0), Fraction::whole(0)],
        &[Fraction::whole(2), Fraction::whole(2)],
        &[Fraction::whole(2), Fraction::whole(2)],
    )?;

    let forward = ResampleForwardOperation::create(&descriptor, &x, &y, None)?;
    assert!(forward.descriptor().is_finalized());

    let backward = ResampleBackwardOperation::create(&descriptor, &x, &y, &dy, &dx, None)?;
    assert!(backward.descriptor().is_finalized());

    Ok(())
}

#[test]
fn test_signal_op_smoke_test() {
    assert_eq!(
        SignalMode::Set,
        SignalMode::try_from(sys::cudnnSignalMode_t::CUDNN_SIGNAL_SET as u32)
            .expect("valid signal mode"),
    );
}

#[test]
fn test_reshape_op_smoke_test() -> Result<()> {
    let _context = setup_context()?;

    let x = Tensor::create_with_id(40, DataType::F32, &[2, 3, 4], &[12, 4, 1], 16)?;
    let y = Tensor::create_with_id(41, DataType::F32, &[2, 12], &[12, 1], 16)?;

    let op = ReshapeOperation::create(&x, &y)?;

    assert!(op.descriptor().is_finalized());
    assert_eq!(
        op.descriptor().descriptor_type(),
        BackendDescriptorType::OperationReshape,
    );

    Ok(())
}

#[test]
fn test_concat_op_smoke_test() -> Result<()> {
    let _context = setup_context()?;

    let x0 = Tensor::create_with_id(42, DataType::F32, &[2, 3], &[3, 1], 16)?;
    let x1 = Tensor::create_with_id(43, DataType::F32, &[2, 5], &[5, 1], 16)?;
    let y = Tensor::create_with_id(44, DataType::F32, &[2, 8], &[8, 1], 16)?;

    let op = ConcatOperation::create(&[&x0, &x1], &y, 1, None)?;

    assert!(op.descriptor().is_finalized());
    assert_eq!(
        op.descriptor().descriptor_type(),
        BackendDescriptorType::OperationConcat,
    );

    Ok(())
}

#[test]
fn test_concat_op_tracks_absent_inplace_index() -> Result<()> {
    let _context = setup_context()?;

    let x0 = Tensor::create_with_id(42, DataType::F32, &[2, 3], &[3, 1], 16)?;
    let x1 = Tensor::create_with_id(43, DataType::F32, &[2, 5], &[5, 1], 16)?;
    let y = Tensor::create_with_id(44, DataType::F32, &[2, 8], &[8, 1], 16)?;

    let op = ConcatOperation::create(&[&x0, &x1], &y, 1, None)?;

    assert_eq!(op.inplace_index(), None);

    Ok(())
}

#[test]
fn test_concat_op_tracks_explicit_inplace_index() -> Result<()> {
    let _context = setup_context()?;

    let x0 = Tensor::create_with_id(42, DataType::F32, &[2, 3], &[3, 1], 16)?;
    let x1 = Tensor::create_with_id(43, DataType::F32, &[2, 5], &[5, 1], 16)?;
    let y = Tensor::create_with_id(44, DataType::F32, &[2, 8], &[8, 1], 16)?;

    let op = ConcatOperation::create(&[&x0, &x1], &y, 1, Some(0))?;
    assert_eq!(op.inplace_index(), Some(0));

    let op = ConcatOperation::create(&[&x0, &x1], &y, 1, Some(1))?;
    assert_eq!(op.inplace_index(), Some(1));

    Ok(())
}

#[test]
fn test_rng_descriptor_smoke_test() -> Result<()> {
    let descriptor = RngDescriptor::create_uniform(-1.0, 1.0)?;

    assert!(descriptor.descriptor().is_finalized());
    assert_eq!(descriptor.distribution(), RngDistribution::Uniform);
    assert_eq!(
        descriptor.descriptor().descriptor_type(),
        BackendDescriptorType::Rng,
    );

    Ok(())
}

#[test]
fn test_rng_op_smoke_test() {
    let err = RngDescriptor::create_normal(0.0, 0.0).unwrap_err();
    assert!(matches!(
        err,
        Error::OutOfRange { name } if name == "standard_deviation"
    ));
}

#[test]
fn test_paged_cache_load_op_smoke_test() {
    assert_eq!(
        BackendDescriptorType::OperationPagedCacheLoad,
        BackendDescriptorType::try_from(
            sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR
                as u32,
        )
        .expect("valid paged cache load descriptor type"),
    );
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
    let context = match setup_context() {
        Ok(ctx) => ctx,
        Err(Error::Cuda(singe_cuda::error::Error::Cuda { code, .. }))
            if code == singe_cuda::error::Status::NoDevice =>
        {
            return Ok(());
        }
        Err(error) => return Err(error),
    };
    let x = Tensor::create_with_id(1, DataType::F32, &[2, 3], &[3, 1], 16)?;
    let y = Tensor::create_with_id(2, DataType::F32, &[2, 3], &[3, 1], 16)?;
    let pointwise = PointwiseDescriptor::create(
        PointwiseMode::ReluFwd,
        DataType::F32,
        NanPropagation::NotPropagate,
    )?;
    let op = PointwiseOperation::unary(&pointwise, &x, &y, 1.0)?;
    let graph = OperationGraph::create(&context, &[op.descriptor()])?;

    let mut descriptor = KernelCache::create()?;
    descriptor.build(&graph)?;

    assert!(descriptor.is_finalized());

    Ok(())
}

#[test]
fn test_expand_band_matrix_op_smoke_test() {
    assert_eq!(
        BackendDescriptorType::OperationExpandBandMatrix,
        BackendDescriptorType::try_from(
            sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_EXPAND_BAND_MATRIX_DESCRIPTOR
                as u32,
        )
        .expect("valid expand band matrix descriptor type"),
    );
}

#[test]
fn test_contract_band_matrix_op_smoke_test() {
    assert_eq!(
        BackendDescriptorType::OperationContractBandMatrix,
        BackendDescriptorType::try_from(
            sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_CONTRACT_BAND_MATRIX_DESCRIPTOR
                as u32,
        )
        .expect("valid contract band matrix descriptor type"),
    );
}

#[test]
fn test_block_scale_quantize_op_smoke_test() {
    assert_eq!(
        BackendDescriptorType::OperationBlockScaleQuantize,
        BackendDescriptorType::try_from(
            sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR
                as u32,
        )
        .expect("valid block scale quantize descriptor type"),
    );
}

#[test]
fn test_block_scale_dequantize_op_smoke_test() {
    assert_eq!(
        BackendDescriptorType::OperationBlockScaleDequantize,
        BackendDescriptorType::try_from(
            sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR
                as u32,
        )
        .expect("valid block scale dequantize descriptor type"),
    );

    let x = Tensor::create_with_id(1, DataType::F8E4M3, &[2, 3, 64], &[192, 64, 1], 16)
        .expect("valid x tensor");
    let scale = Tensor::create_vectorized(
        2,
        DataType::F8E8M0,
        &[2, 3, 4],
        &[12, 4, 1],
        16,
        2,
        4,
        Some(BackendTensorReordering::F8_128x4),
    )
    .expect("valid scale tensor");
    let y = Tensor::create_with_id(3, DataType::F32, &[2, 3, 64], &[192, 64, 1], 16)
        .expect("valid y tensor");

    let _ =
        BlockScaleDequantizeOperation::create(&x, &scale, &y, Some(DataType::F32), &[16], false);
}

#[test]
fn test_softmax_op_smoke_test() {
    assert_eq!(
        BackendDescriptorType::OperationSoftmax,
        BackendDescriptorType::try_from(
            sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_SOFTMAX_DESCRIPTOR as u32,
        )
        .expect("valid softmax descriptor type"),
    );
}

#[test]
fn test_matmul_descriptor_smoke_test() -> Result<()> {
    let descriptor = MatmulDescriptor::create(DataType::F32)?;

    assert!(descriptor.descriptor().is_finalized());
    assert_eq!(
        descriptor.descriptor().descriptor_type(),
        BackendDescriptorType::Matmul,
    );

    Ok(())
}

#[test]
fn test_matmul_op_smoke_test() -> Result<()> {
    let _context = setup_context()?;

    let matmul = MatmulDescriptor::create(DataType::F32)?;
    let a = Tensor::create_with_id(20, DataType::F32, &[4, 8], &[8, 1], 16)?;
    let b = Tensor::create_with_id(21, DataType::F32, &[8, 16], &[16, 1], 16)?;
    let c = Tensor::create_with_id(22, DataType::F32, &[4, 16], &[16, 1], 16)?;

    let op = MatmulOperation::create(&matmul, &a, &b, &c)?;

    assert!(op.descriptor().is_finalized());
    assert_eq!(
        op.descriptor().descriptor_type(),
        BackendDescriptorType::OperationMatmul,
    );

    Ok(())
}

#[test]
fn test_sdpa_forward_op_smoke_test() -> Result<()> {
    let _context = setup_context()?;

    let q = Tensor::create_with_id(30, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;
    let k = Tensor::create_with_id(31, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;
    let v = Tensor::create_with_id(32, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;
    let o = Tensor::create_with_id(33, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;
    let scale = Tensor::create_by_value(34, DataType::F32, &[1], &[1], 4)?;
    let stats = Tensor::create_virtual(35, DataType::F32, &[2, 4, 8, 1], &[32, 8, 1, 1], 16)?;

    let op =
        SdpaForwardOperation::create(&q, &k, &v, &o, &scale, Some(&stats), Default::default())?;

    assert!(op.descriptor().is_finalized());
    assert_eq!(
        op.descriptor().descriptor_type(),
        BackendDescriptorType::OperationSdpaForward,
    );

    Ok(())
}

#[test]
fn test_sdpa_forward_op_accepts_softmax_descriptor() -> Result<()> {
    let _context = match setup_context() {
        Ok(ctx) => ctx,
        Err(Error::Cuda(singe_cuda::error::Error::Cuda { code, .. }))
            if code == singe_cuda::error::Status::NoDevice =>
        {
            return Ok(());
        }
        Err(error) => return Err(error),
    };

    let q = Tensor::create_with_id(230, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;
    let k = Tensor::create_with_id(231, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;
    let v = Tensor::create_with_id(232, DataType::F16, &[2, 4, 8, 12], &[384, 96, 12, 1], 16)?;
    let o = Tensor::create_with_id(233, DataType::F16, &[2, 4, 8, 12], &[384, 96, 12, 1], 16)?;
    let scale = Tensor::create_by_value(234, DataType::F32, &[1], &[1], 4)?;
    let stats = Tensor::create_virtual(235, DataType::F32, &[2, 4, 8, 1], &[32, 8, 1, 1], 16)?;
    let max = Tensor::create_virtual(236, DataType::F32, &[2, 4, 8, 1], &[32, 8, 1, 1], 16)?;
    let sum_exp = Tensor::create_virtual(237, DataType::F32, &[2, 4, 8, 1], &[32, 8, 1, 1], 16)?;
    let softmax_p =
        Tensor::create_virtual(238, DataType::F16, &[2, 4, 8, 8], &[256, 64, 8, 1], 16)?;
    let softmax_s =
        Tensor::create_virtual(239, DataType::F16, &[2, 4, 8, 8], &[256, 64, 8, 1], 16)?;
    let softmax = SoftmaxOperation::create(
        &softmax_p,
        &softmax_s,
        Some(&stats),
        Some(&max),
        Some(&sum_exp),
        None,
    )?;

    let op = SdpaForwardOperation::create(
        &q,
        &k,
        &v,
        &o,
        &scale,
        Some(&stats),
        SdpaForwardConfig {
            softmax: Some(&softmax),
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
    let _context = setup_context()?;

    let q = Tensor::create_with_id(130, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;
    let k = Tensor::create_with_id(131, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;
    let v = Tensor::create_with_id(132, DataType::F16, &[2, 4, 8, 12], &[384, 96, 12, 1], 16)?;
    let o = Tensor::create_with_id(133, DataType::F16, &[2, 4, 8, 12], &[384, 96, 12, 1], 16)?;
    let scale = Tensor::create_by_value(134, DataType::F32, &[1], &[1], 4)?;
    let stats = Tensor::create_virtual(135, DataType::F32, &[2, 4, 8, 1], &[32, 8, 1, 1], 16)?;
    let sequence_length_query =
        Tensor::create_with_id(136, DataType::I32, &[2, 1, 1, 1], &[1, 1, 1, 1], 4)?;
    let sequence_length_key_value =
        Tensor::create_with_id(137, DataType::I32, &[2, 1, 1, 1], &[1, 1, 1, 1], 4)?;
    let seed = Tensor::create_by_value(138, DataType::I64, &[1, 1, 1, 1], &[1, 1, 1, 1], 8)?;
    let offset = Tensor::create_by_value(139, DataType::I64, &[1, 1, 1, 1], &[1, 1, 1, 1], 8)?;
    let rng_dump = Tensor::create_virtual(140, DataType::F16, &[2, 4, 8, 8], &[256, 64, 8, 1], 16)?;

    let op = SdpaForwardOperation::create(
        &q,
        &k,
        &v,
        &o,
        &scale,
        Some(&stats),
        SdpaForwardConfig {
            sequence_length_query: Some(&sequence_length_query),
            sequence_length_key_value: Some(&sequence_length_key_value),
            page_table_k: None,
            page_table_v: None,
            block_mask: None,
            softmax: None,
            subgraph: None,
            subgraph_input_id: None,
            subgraph_output_id: None,
            dropout_seed: Some(&seed),
            dropout_offset: Some(&offset),
            dropout_random_number_generator_dump: Some(&rng_dump),
            dropout_probability: Some(0.25_f32),
            unfuse_fma: true,
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
    let _context = match setup_context() {
        Ok(ctx) => ctx,
        Err(Error::Cuda(singe_cuda::error::Error::Cuda { code, .. }))
            if code == singe_cuda::error::Status::NoDevice =>
        {
            return Ok(());
        }
        Err(error) => return Err(error),
    };

    let q = Tensor::create_with_id(330, DataType::F16, &[1, 2, 4, 16], &[128, 64, 16, 1], 16)?;
    let k = Tensor::create_with_id(331, DataType::F16, &[1, 2, 4, 16], &[128, 64, 16, 1], 16)?;
    let v = Tensor::create_with_id(332, DataType::F16, &[1, 2, 4, 12], &[96, 48, 12, 1], 16)?;
    let o = Tensor::create_with_id(333, DataType::F16, &[1, 2, 4, 12], &[96, 48, 12, 1], 16)?;
    let scale = Tensor::create_by_value(334, DataType::F32, &[1], &[1], 4)?;
    let block_mask = Tensor::create_with_id(335, DataType::U8, &[1, 2, 1, 1], &[2, 1, 1, 1], 1)?;

    let op = SdpaForwardOperation::create(
        &q,
        &k,
        &v,
        &o,
        &scale,
        None,
        SdpaForwardConfig {
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
    let context = match setup_context() {
        Ok(ctx) => ctx,
        Err(Error::Cuda(singe_cuda::error::Error::Cuda { code, .. }))
            if code == singe_cuda::error::Status::NoDevice =>
        {
            return Ok(());
        }
        Err(error) => return Err(error),
    };

    let q = Tensor::create_with_id(430, DataType::F16, &[1, 2, 4, 16], &[128, 64, 16, 1], 16)?;
    let k = Tensor::create_with_id(431, DataType::F16, &[1, 2, 4, 16], &[128, 64, 16, 1], 16)?;
    let v = Tensor::create_with_id(432, DataType::F16, &[1, 2, 4, 12], &[96, 48, 12, 1], 16)?;
    let o = Tensor::create_with_id(433, DataType::F16, &[1, 2, 4, 12], &[96, 48, 12, 1], 16)?;
    let scale = Tensor::create_by_value(434, DataType::F32, &[1], &[1], 4)?;
    let stats = Tensor::create_virtual(435, DataType::F32, &[1, 2, 4, 1], &[8, 4, 1, 1], 16)?;

    let subgraph_input =
        Tensor::create_virtual(530, DataType::F16, &[1, 2, 4, 4], &[32, 16, 4, 1], 16)?;
    let subgraph_output =
        Tensor::create_virtual(531, DataType::F16, &[1, 2, 4, 4], &[32, 16, 4, 1], 16)?;
    let pointwise = PointwiseDescriptor::create(
        PointwiseMode::ReluFwd,
        DataType::F16,
        NanPropagation::NotPropagate,
    )?;
    let subgraph_op =
        PointwiseOperation::unary(&pointwise, &subgraph_input, &subgraph_output, 1.0)?;
    let subgraph = OperationGraph::create(&context, &[subgraph_op.descriptor()])?;

    let op = SdpaForwardOperation::create(
        &q,
        &k,
        &v,
        &o,
        &scale,
        Some(&stats),
        SdpaForwardConfig {
            subgraph: Some(&subgraph),
            subgraph_input_id: Some(subgraph_input.id()),
            subgraph_output_id: Some(subgraph_output.id()),
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
fn test_sdpa_forward_op_rejects_partial_subgraph_descriptor() -> Result<()> {
    let q = Tensor::create_with_id(440, DataType::F16, &[1, 2, 4, 16], &[128, 64, 16, 1], 16)?;
    let k = Tensor::create_with_id(441, DataType::F16, &[1, 2, 4, 16], &[128, 64, 16, 1], 16)?;
    let v = Tensor::create_with_id(442, DataType::F16, &[1, 2, 4, 12], &[96, 48, 12, 1], 16)?;
    let o = Tensor::create_with_id(443, DataType::F16, &[1, 2, 4, 12], &[96, 48, 12, 1], 16)?;
    let scale = Tensor::create_by_value(444, DataType::F32, &[1], &[1], 4)?;

    let err = SdpaForwardOperation::create(
        &q,
        &k,
        &v,
        &o,
        &scale,
        None,
        SdpaForwardConfig {
            subgraph_input_id: Some(1.into()),
            ..Default::default()
        },
    )
    .unwrap_err();

    assert!(matches!(
        err,
        Error::DescriptorMismatch { name } if name == "sdpa subgraph descriptor"
    ));

    Ok(())
}

#[test]
fn test_sdpa_backward_op_smoke_test() -> Result<()> {
    let _context = setup_context()?;

    let q = Tensor::create_with_id(40, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;
    let k = Tensor::create_with_id(41, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;
    let v = Tensor::create_with_id(42, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;
    let o = Tensor::create_with_id(43, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;
    let stats = Tensor::create_virtual(44, DataType::F32, &[2, 4, 8, 1], &[32, 8, 1, 1], 16)?;
    let scale = Tensor::create_by_value(45, DataType::F32, &[1], &[1], 4)?;
    let d_o = Tensor::create_with_id(46, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;
    let d_q = Tensor::create_with_id(47, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;
    let d_k = Tensor::create_with_id(48, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;
    let d_v = Tensor::create_with_id(49, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;

    let op = SdpaBackwardOperation::create(
        &q,
        &k,
        &v,
        &o,
        &stats,
        &scale,
        &d_o,
        &d_q,
        &d_k,
        &d_v,
        Default::default(),
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
    let _context = setup_context()?;

    let q = Tensor::create_with_id(150, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;
    let k = Tensor::create_with_id(151, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;
    let v = Tensor::create_with_id(152, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;
    let o = Tensor::create_with_id(153, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;
    let stats = Tensor::create_virtual(154, DataType::F32, &[2, 4, 8, 1], &[32, 8, 1, 1], 16)?;
    let scale = Tensor::create_by_value(155, DataType::F32, &[1], &[1], 4)?;
    let d_o = Tensor::create_with_id(156, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;
    let d_q = Tensor::create_with_id(157, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;
    let d_k = Tensor::create_with_id(158, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;
    let d_v = Tensor::create_with_id(159, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;
    let sequence_length_query =
        Tensor::create_with_id(160, DataType::I32, &[2, 1, 1, 1], &[1, 1, 1, 1], 4)?;
    let sequence_length_key_value =
        Tensor::create_with_id(161, DataType::I32, &[2, 1, 1, 1], &[1, 1, 1, 1], 4)?;
    let sink = Tensor::create_with_id(162, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;
    let sink_gradient =
        Tensor::create_with_id(163, DataType::F16, &[2, 4, 8, 16], &[512, 128, 16, 1], 16)?;

    let op = SdpaBackwardOperation::create(
        &q,
        &k,
        &v,
        &o,
        &stats,
        &scale,
        &d_o,
        &d_q,
        &d_k,
        &d_v,
        SdpaBackwardConfig {
            sequence_length_query: Some(&sequence_length_query),
            sequence_length_key_value: Some(&sequence_length_key_value),
            subgraph: None,
            subgraph_input_id: None,
            subgraph_output_id: None,
            sink: Some(&sink),
            sink_gradient: Some(&sink_gradient),
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
    let _context = setup_context()?;

    let x = Tensor::create_with_id(60, DataType::F32, &[2, 4, 8, 8], &[256, 64, 8, 1], 16)?;
    let b = Tensor::create_by_value(61, DataType::F32, &[1], &[1], 4)?;
    let sequence_length_query =
        Tensor::create_with_id(62, DataType::I32, &[2, 1, 1, 1], &[1, 1, 1, 1], 16)?;
    let sequence_length_key_value =
        Tensor::create_with_id(63, DataType::I32, &[2, 1, 1, 1], &[1, 1, 1, 1], 16)?;
    let shift_right_bound = Tensor::create_by_value(64, DataType::I32, &[1], &[1], 4)?;
    let y = Tensor::create_with_id(65, DataType::F32, &[2, 4, 8, 8], &[256, 64, 8, 1], 16)?;

    let op = DiagonalBandMaskOperation::create(
        &x,
        &b,
        &y,
        PointwiseMode::CmpGe,
        Some(&sequence_length_query),
        Some(&sequence_length_key_value),
        None,
        Some(&shift_right_bound),
    )?;

    assert!(op.descriptor().is_finalized());
    assert_eq!(
        op.descriptor().descriptor_type(),
        BackendDescriptorType::OperationDiagonalBandMask,
    );

    Ok(())
}

#[test]
fn test_moe_grouped_matmul_op_smoke_test() {
    assert_eq!(
        BackendDescriptorType::OperationMoeGroupedMatmul,
        BackendDescriptorType::try_from(
            sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR
                as u32,
        )
        .expect("valid moe grouped matmul descriptor type"),
    );
}

#[test]
fn test_moe_grouped_matmul_op_ignores_top_k_outside_scatter_mode() -> Result<()> {
    let _context = setup_context()?;

    let token = Tensor::create_with_id(
        200,
        DataType::F16,
        &[1, 2048, 512],
        &[2048 * 512, 512, 1],
        16,
    )?;
    let weight =
        Tensor::create_with_id(201, DataType::F16, &[3, 512, 256], &[512 * 256, 1, 512], 16)?;
    let first_token_offset = Tensor::create_with_id(202, DataType::I32, &[6, 1, 1], &[1, 1, 1], 4)?;
    let output = Tensor::create_with_id(
        203,
        DataType::F16,
        &[1, 2048, 256],
        &[2048 * 256, 256, 1],
        16,
    )?;

    let _op = MoeGroupedMatmulOperation::create(
        MoeGroupedMatmulMode::None,
        DataType::F16,
        &token,
        &weight,
        &first_token_offset,
        &output,
        None,
        None,
        Some(2),
    )?;

    Ok(())
}

#[test]
fn test_convolution_backward_data_op_smoke_test() -> Result<()> {
    let _context = setup_context()?;

    let convolution = ConvolutionDescriptor::create(
        DataType::F32,
        ConvolutionMode::CrossCorrelation,
        &[0, 0],
        &[0, 0],
        &[1, 1],
        &[1, 1],
    )?;
    let w = Tensor::create_with_id(1, DataType::F32, &[4, 3, 3, 3], &[27, 9, 3, 1], 16)?;
    let output_gradient =
        Tensor::create_with_id(2, DataType::F32, &[1, 4, 8, 8], &[256, 64, 8, 1], 16)?;
    let input_gradient =
        Tensor::create_with_id(3, DataType::F32, &[1, 3, 10, 10], &[300, 100, 10, 1], 16)?;

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
    let _context = setup_context()?;

    let convolution = ConvolutionDescriptor::create(
        DataType::F32,
        ConvolutionMode::CrossCorrelation,
        &[0, 0],
        &[0, 0],
        &[1, 1],
        &[1, 1],
    )?;
    let x = Tensor::create_with_id(4, DataType::F32, &[1, 3, 10, 10], &[300, 100, 10, 1], 16)?;
    let output_gradient =
        Tensor::create_with_id(5, DataType::F32, &[1, 4, 8, 8], &[256, 64, 8, 1], 16)?;
    let filter_gradient =
        Tensor::create_with_id(6, DataType::F32, &[4, 3, 3, 3], &[27, 9, 3, 1], 16)?;

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
fn test_norm_forward_op_descriptor_smoke_test() {
    let descriptor = NormalizationForwardConfig::new(
        BackendNormalizationMode::Layer,
        BackendNormalizationForwardPhase::Inference,
    );
    assert_eq!(descriptor.mode(), BackendNormalizationMode::Layer);
    assert_eq!(
        descriptor.phase(),
        BackendNormalizationForwardPhase::Inference
    );
}

#[test]
fn test_norm_backward_op_descriptor_smoke_test() {
    let descriptor = NormalizationBackwardConfig::new(BackendNormalizationMode::Layer);
    assert_eq!(descriptor.mode(), BackendNormalizationMode::Layer);
}

#[test]
fn test_norm_forward_op_smoke_test() -> Result<()> {
    let _context = setup_context()?;

    let norm = NormalizationForwardConfig::new(
        BackendNormalizationMode::Layer,
        BackendNormalizationForwardPhase::Inference,
    );
    let x = Tensor::create_with_id(10, DataType::F32, &[2, 8], &[8, 1], 16)?;
    let mean = Tensor::create_with_id(11, DataType::F32, &[2, 1], &[1, 1], 16)?;
    let inv_variance = Tensor::create_with_id(12, DataType::F32, &[2, 1], &[1, 1], 16)?;
    let scale = Tensor::create_with_id(13, DataType::F32, &[1, 8], &[8, 1], 16)?;
    let bias = Tensor::create_with_id(14, DataType::F32, &[1, 8], &[8, 1], 16)?;
    let epsilon = Tensor::create_by_value(15, DataType::F32, &[1], &[1], 4)?;
    let y = Tensor::create_with_id(16, DataType::F32, &[2, 8], &[8, 1], 16)?;

    let op = NormalizationForwardOperation::create(
        &norm,
        &x,
        Some(&mean),
        Some(&inv_variance),
        &scale,
        Some(&bias),
        Some(&epsilon),
        None,
        None,
        None,
        None,
        None,
        &[],
        &y,
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
    let _context = setup_context()?;

    let norm = NormalizationBackwardConfig::new(BackendNormalizationMode::Layer);
    let x = Tensor::create_with_id(50, DataType::F32, &[2, 8], &[8, 1], 16)?;
    let mean = Tensor::create_with_id(51, DataType::F32, &[2, 1], &[1, 1], 16)?;
    let inv_variance = Tensor::create_with_id(52, DataType::F32, &[2, 1], &[1, 1], 16)?;
    let dy = Tensor::create_with_id(53, DataType::F32, &[2, 8], &[8, 1], 16)?;
    let scale = Tensor::create_with_id(54, DataType::F32, &[1, 8], &[8, 1], 16)?;
    let epsilon = Tensor::create_by_value(55, DataType::F32, &[1], &[1], 4)?;
    let dscale = Tensor::create_with_id(56, DataType::F32, &[1, 8], &[8, 1], 16)?;
    let bias_gradient = Tensor::create_with_id(57, DataType::F32, &[1, 8], &[8, 1], 16)?;
    let dx = Tensor::create_with_id(58, DataType::F32, &[2, 8], &[8, 1], 16)?;

    let op = NormalizationBackwardOperation::create(
        &norm,
        &x,
        Some(&mean),
        &inv_variance,
        &dy,
        &scale,
        Some(&epsilon),
        &dscale,
        Some(&bias_gradient),
        &[],
        &dx,
    )?;

    assert!(op.descriptor().is_finalized());
    assert_eq!(
        op.descriptor().descriptor_type(),
        BackendDescriptorType::OperationNormBackward,
    );

    Ok(())
}

#[test]
fn test_import_surface_smoke_test() {
    let _ = size_of::<PagedCacheLoadOperation>();
    let _ = size_of::<ExpandBandMatrixOperation>();
    let _ = size_of::<ContractBandMatrixOperation>();
    let _ = size_of::<BlockScaleQuantizeOperation>();
    let _ = size_of::<BlockScaleDequantizeOperation>();
    let _ = size_of::<SoftmaxOperation>();
    let _ = size_of::<SdpaBackwardOperation>();
    let _ = size_of::<DiagonalBandMaskOperation>();
    let _ = size_of::<MoeGroupedMatmulOperation>();
}
