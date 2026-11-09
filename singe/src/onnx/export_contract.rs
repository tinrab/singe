use std::collections::BTreeMap;

use singe_onnx::{
    DimensionValue, Model, TypeValue, ValueInfo,
    proto::onnx::{StringStringEntryProto, ValueInfoProto},
};

use crate::onnx::{
    error::{Error, Result},
    options::OnnxOptions,
};

pub const SHAPE_CONTRACT: &str = "singe.shape_contract";

pub fn annotate_dimension(
    info: &mut ValueInfoProto,
    axis: usize,
    representative: i64,
    min: Option<i64>,
    max: Option<i64>,
) {
    for (field, value) in [
        ("representative", Some(representative)),
        ("min", min),
        ("max", max),
    ] {
        if let Some(value) = value {
            info.metadata_props.push(StringStringEntryProto {
                key: format!("singe.axis.{axis}.{field}"),
                value: value.to_string(),
            });
        }
    }
}

/// Standard ONNX `dim_param` preserves equality, but cannot express inclusive ranges or an import representative.
/// Our versioned metadata adds those for Singe consumers.
/// External runtimes still require caller-side range guards.
pub fn apply_shape_contract(model: &Model, options: &mut OnnxOptions) -> Result<()> {
    let Some(version) = model.metadata_props.get(SHAPE_CONTRACT) else {
        return Ok(());
    };
    if version != "1" {
        return Err(Error::InvalidModel {
            reason: "unsupported singe shape contract version".into(),
        });
    }

    let mut representatives = BTreeMap::new();
    for info in model
        .graph
        .inputs
        .iter()
        .chain(&model.graph.value_info)
        .chain(&model.graph.outputs)
    {
        let dimensions = parse_dimensions(info)?;
        if dimensions.is_empty() {
            continue;
        }

        let Some(TypeValue::Tensor(tensor)) = info.value_type.as_ref().map(|value| &value.value)
        else {
            return Err(invalid_contract(info, "tensor type is required"));
        };
        let shape = tensor
            .shape
            .as_ref()
            .ok_or_else(|| invalid_contract(info, "tensor shape is required"))?;
        for (axis, metadata) in dimensions {
            let Some(DimensionValue::Parameter(symbol)) = shape
                .dimensions
                .get(axis)
                .and_then(|dim| dim.value.as_ref())
            else {
                return Err(invalid_contract(
                    info,
                    "axis must refer to a named symbolic dimension",
                ));
            };
            let representative = metadata
                .representative
                .ok_or_else(|| invalid_contract(info, "representative value is required"))?;
            let (min, max) = (metadata.min, metadata.max);
            if representatives
                .insert(symbol.clone(), representative)
                .is_some_and(|previous| previous != representative)
            {
                return Err(invalid_contract(
                    info,
                    "inconsistent representatives for a symbol",
                ));
            }

            // Validate the serialized contract independently of any caller override, then intersect caller bounds instead of weakening it.
            OnnxOptions::new()
                .with_symbolic_dimension_range(symbol, representative, min, max)
                .validate()?;
            options
                .symbolic_dimensions
                .entry(symbol.clone())
                .or_insert(representative);
            let bounds = options
                .symbolic_dimension_bounds
                .entry(symbol.clone())
                .or_default();
            if let Some(min) = min {
                bounds.0 = Some(bounds.0.map_or(min, |old| old.max(min)));
            }
            if let Some(max) = max {
                bounds.1 = Some(bounds.1.map_or(max, |old| old.min(max)));
            }
        }
    }
    Ok(())
}

#[derive(Default)]
struct DimensionMetadata {
    representative: Option<i64>,
    min: Option<i64>,
    max: Option<i64>,
}

fn invalid_contract(info: &ValueInfo, reason: &str) -> Error {
    Error::InvalidModel {
        reason: format!("shape contract for `{}`: {reason}", info.name),
    }
}

fn parse_dimensions(info: &ValueInfo) -> Result<BTreeMap<usize, DimensionMetadata>> {
    let mut dimensions: BTreeMap<usize, DimensionMetadata> = BTreeMap::new();
    for (key, value) in &info.metadata_props {
        let Some(key) = key.strip_prefix("singe.axis.") else {
            continue;
        };
        let (axis, field) = key
            .split_once('.')
            .ok_or_else(|| invalid_contract(info, "invalid axis metadata key"))?;
        let axis = axis
            .parse()
            .map_err(|_| invalid_contract(info, "invalid axis index"))?;
        let value = value
            .parse()
            .map_err(|_| invalid_contract(info, "invalid integer"))?;
        let metadata = dimensions.entry(axis).or_default();
        match field {
            "representative" => metadata.representative = Some(value),
            "min" => metadata.min = Some(value),
            "max" => metadata.max = Some(value),
            _ => return Err(invalid_contract(info, "unknown dimension field")),
        }
    }
    Ok(dimensions)
}
