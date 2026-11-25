use singe_onnx::Node;

use crate::{
    backend::registry::{Operation, OperationAttributes, ResizeAttributes},
    data_type::DataType,
    graph::{DimensionExpr, SymbolicAttributes, node::NodeKind},
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{
            expect_input_range, expect_outputs, register_node_outputs, shape_input_expressions,
            shape_ops::onnx_dimension_to_usize,
        },
        utility::{node_label, node_name},
    },
    shape::{Shape, spatial::Spatial2d, transform::normalize_dim},
    tensor::{InterpolateMode, Interpolation},
};

// ONNX `Resize` of the last two axes, in the modes PyTorch exports `interpolate` as.
pub fn lower_resize(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_range(importer, node, 1, 4)?;
    expect_outputs(importer, node, 1)?;
    let refuse = |reason: &str| Error::UnsupportedShape {
        name: node_label(node),
        reason: reason.into(),
    };
    let version = importer.opset_version(node.domain.as_deref());
    if version < 11 {
        return Err(refuse("resize before opset 11"));
    }

    let interpolation = interpolation(node)?;
    let input = importer.required_input(node, 0)?;
    let rank = input.shape.rank();
    let attrs = Attributes::create(node)?;
    let axes = match attrs.ints("axes")? {
        Some(axes) => axes
            .iter()
            .map(|&axis| normalize_dim(axis, rank).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?,
        None => (0..rank).collect(),
    };
    if rank < 2 || !axes.ends_with(&[rank - 2, rank - 1]) {
        return Err(refuse("resize of axes other than the last two"));
    }

    let axis = |axis: i64| onnx_dimension_to_usize(node, "resize axis", axis);
    let (height_axis, width_axis) = (axis(rank - 2)?, axis(rank - 1)?);
    let leading = axes.len() - 2;
    let size = if node.inputs.get(3).is_some_and(|name| !name.is_empty()) {
        let sizes = shape_input_expressions(importer, node, 3, "sizes")?;
        if sizes.len() != axes.len() {
            return Err(refuse("resize sizes must name every resized axis"));
        }

        for (dim, &leading_axis) in sizes.iter().zip(&axes[..leading]) {
            let extent = input.shape.dimensions()[axis(leading_axis)?];
            if importer.evaluate_dimension(dim)? != extent {
                return Err(refuse("resize of axes other than the last two"));
            }
        }
        Spatial2d::new(sizes[leading].clone(), sizes[leading + 1].clone())
    } else {
        let (name, scales) = importer.required_static_input_tensor(node, 2, "scales")?;
        let scales = scales.cast(DataType::F32)?.to_vec::<f32>()?;
        if scales.len() != axes.len() {
            return Err(Error::invalid_initializer(
                &name,
                "resize scales must name every resized axis",
            ));
        }

        if scales[..leading].iter().any(|&scale| scale != 1.0) {
            return Err(refuse("resize of axes other than the last two"));
        }

        let [height, width] = [
            (height_axis, scales[leading]),
            (width_axis, scales[leading + 1]),
        ]
        .map(|(axis, scale)| {
            scaled(importer.metadata_dimension(input.node, axis)?, scale).ok_or_else(|| {
                refuse("resize by a scale that is neither whole nor one over a whole number")
            })
        });
        Spatial2d::new(height?, width?)
    };
    let evaluated = Spatial2d::new(
        importer.evaluate_dimension(&size.height)?,
        importer.evaluate_dimension(&size.width)?,
    );
    let mut dimensions = input.shape.dimensions().to_vec();
    dimensions[height_axis] = evaluated.height;
    dimensions[width_axis] = evaluated.width;
    let shape = Shape::create(dimensions)?;
    let device = importer.options().device;
    let id = importer.graph().add_named_operation(
        node_name(node),
        NodeKind::operation_invocation(
            Operation::Resize,
            OperationAttributes::Resize(ResizeAttributes {
                size: evaluated,
                interpolation,
            }),
            vec![input.node],
        ),
        input.data_type,
        shape.clone(),
        device,
    )?;
    if !(size.height.is_constant() && size.width.is_constant()) {
        importer.graph().shape_program.attributes.insert(
            id,
            SymbolicAttributes::Resize {
                size,
                interpolation,
            },
        );
    }
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, input.data_type, shape)],
    )
}

// The extent `scale` times `extent`, for a whole scale or one over a whole number.
fn scaled(extent: DimensionExpr, scale: f32) -> Option<DimensionExpr> {
    if scale >= 1.0 && scale.fract() == 0.0 {
        Some(extent * scale as i64)
    } else if scale > 0.0 && scale < 1.0 && scale.recip().fract() == 0.0 {
        Some(extent / scale.recip() as i64)
    } else {
        None
    }
}

// The interpolation a `Resize` node's attributes describe, among those `Tensor::interpolate` computes.
fn interpolation(node: &Node) -> Result<Interpolation> {
    let attrs = Attributes::create(node)?;
    let mode = attrs.string("mode", "nearest")?;
    let coordinates = attrs.string("coordinate_transformation_mode", "half_pixel")?;
    let antialias = attrs.int("antialias", 0)? != 0;
    let cubic = attrs.float("cubic_coeff_a", -0.75)?;
    let exclude_outside = attrs.int("exclude_outside", 0)? != 0;
    let nearest = attrs.string("nearest_mode", "round_prefer_floor")?;
    let policy = attrs.string("keep_aspect_ratio_policy", "stretch")?;
    let half_pixel = matches!(
        coordinates.as_str(),
        "half_pixel" | "pytorch_half_pixel" | "align_corners"
    );
    let interpolated = match mode.as_str() {
        "nearest" if coordinates == "asymmetric" && nearest == "floor" && !antialias => {
            Some(InterpolateMode::Nearest)
        }
        "linear" if half_pixel => Some(InterpolateMode::Bilinear),
        "cubic" if half_pixel && cubic == if antialias { -0.5 } else { -0.75 } => {
            Some(InterpolateMode::Bicubic)
        }
        _ => None,
    }
    .filter(|_| !exclude_outside && policy == "stretch");
    let Some(interpolated) = interpolated else {
        return Err(Error::UnsupportedShape {
            name: node_label(node),
            reason: format!("resize in {mode} mode with {coordinates} coordinates"),
        });
    };
    Ok(Interpolation::new(interpolated)
        .with_align_corners(coordinates == "align_corners")
        .with_antialias(antialias))
}
