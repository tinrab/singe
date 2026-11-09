mod schemas;

use std::collections::BTreeMap;

use singe_onnx::{AttributeType, Model, Node};

use self::schemas::SCHEMAS;
use crate::{
    data_type::DataType,
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::Importer,
    },
};

pub struct OperatorSchema {
    operator: &'static str,
    min_opset: i64,
    max_opset: i64,
    schema_version: i64,
    input_range: (usize, usize),
    output_range: (usize, usize),
    required_inputs: &'static [usize],
    required_outputs: &'static [usize],
    input_types: &'static [(&'static str, &'static [DataType], bool)],
    attributes: &'static [(&'static str, AttributeType, bool)],
}

pub fn validate_inputs(importer: &Importer, node: &Node) -> Result<()> {
    let version = importer.opset_version(node.domain.as_deref());
    let schema = SCHEMAS
        .iter()
        .find(|schema| {
            schema.operator == node.operator_type
                && (schema.min_opset..=schema.max_opset).contains(&version)
        })
        .ok_or_else(|| Error::InvalidModel {
            reason: "node has no validated schema".into(),
        })?;
    let mut variables = BTreeMap::new();
    for (index, name) in node
        .inputs
        .iter()
        .enumerate()
        .filter(|(_, name)| !name.is_empty())
    {
        let Some(&(parameter, allowed, homogeneous)) = schema
            .input_types
            .get(index)
            .or_else(|| schema.input_types.last())
        else {
            continue;
        };
        let value = importer.value(name)?;
        if !allowed.contains(&value.data_type)
            || homogeneous
                && variables
                    .insert(parameter, value.data_type)
                    .is_some_and(|previous| previous != value.data_type)
        {
            return Err(Error::unsupported_operator(
                node.domain.as_deref(),
                &node.operator_type,
                version,
                format!(
                    "input `{name}` dtype {:?} violates schema {} type {parameter}",
                    value.data_type, schema.schema_version
                ),
            ));
        }
    }
    Ok(())
}

fn validate_schema(node: &Node, version: i64) -> Result<()> {
    let schema = SCHEMAS
        .iter()
        .find(|schema| {
            schema.operator == node.operator_type
                && (schema.min_opset..=schema.max_opset).contains(&version)
        })
        .ok_or_else(|| {
            Error::unsupported_operator(
                node.domain.as_deref(),
                &node.operator_type,
                version,
                "operator schema version is not supported",
            )
        })?;
    for (label, values, (minimum, maximum), required) in [
        (
            "input",
            &node.inputs,
            schema.input_range,
            schema.required_inputs,
        ),
        (
            "output",
            &node.outputs,
            schema.output_range,
            schema.required_outputs,
        ),
    ] {
        if !(minimum..=maximum).contains(&values.len())
            || required
                .iter()
                .any(|&index| values.get(index).is_none_or(String::is_empty))
        {
            return Err(Error::unsupported_operator(
                node.domain.as_deref(),
                &node.operator_type,
                version,
                format!("invalid {label} slots for schema {}", schema.schema_version),
            ));
        }
    }

    for attribute in &node.attributes {
        if !attribute.ref_attribute_name.is_empty() {
            return Err(Error::invalid_attribute(
                &node.name,
                &attribute.name,
                "function attribute references are not supported",
            ));
        }

        let Some((_, expected, _)) = schema
            .attributes
            .iter()
            .find(|(name, _, _)| *name == attribute.name)
        else {
            return Err(Error::invalid_attribute(
                &node.name,
                &attribute.name,
                format!(
                    "attribute is not defined by {} schema {} at opset {version}",
                    node.operator_type, schema.schema_version
                ),
            ));
        };
        if attribute.attribute_type != *expected {
            return Err(Error::invalid_attribute(
                &node.name,
                &attribute.name,
                format!("expected {expected:?}"),
            ));
        }
    }

    for &(name, _, required) in schema.attributes {
        if required
            && !node
                .attributes
                .iter()
                .any(|attribute| attribute.name == name)
        {
            return Err(Error::invalid_attribute(
                &node.name,
                name,
                "required schema attribute is missing",
            ));
        }
    }
    Ok(())
}

#[expect(
    clippy::float_cmp,
    reason = "exact equality selects an identity, specialization, or deterministic tie; approximate equality changes semantics"
)]
pub fn validated_opsets(model: &Model) -> Result<BTreeMap<Option<String>, i64>> {
    if !(7..=10).contains(&model.ir_version) {
        return Err(Error::InvalidModel {
            reason: format!(
                "supported IR versions are 7 through 10, got {}",
                model.ir_version
            ),
        });
    }

    if !model.functions.is_empty()
        || !model.training_info.is_empty()
        || !model.graph.sparse_initializers.is_empty()
    {
        return Err(Error::InvalidModel {
            reason: "local functions, training graphs, and sparse initializers are not supported"
                .into(),
        });
    }

    let mut opsets = BTreeMap::new();
    for opset in &model.operator_set_imports {
        let domain = opset.domain.clone().filter(|domain| !domain.is_empty());
        if opset.version <= 0 {
            return Err(Error::InvalidModel {
                reason: "opset versions must be positive".into(),
            });
        }
        if opsets.insert(domain, opset.version).is_some() {
            return Err(Error::InvalidModel {
                reason: "duplicate opset domain".into(),
            });
        }
    }

    for node in &model.graph.nodes {
        if node.overload.is_some() {
            return Err(Error::InvalidModel {
                reason: "operator overloads are not supported".into(),
            });
        }

        let domain = node.domain.clone().filter(|domain| !domain.is_empty());
        let version = *opsets.get(&domain).ok_or_else(|| Error::InvalidModel {
            reason: format!(
                "missing opset import for domain `{}`",
                domain.as_deref().unwrap_or("")
            ),
        })?;
        if domain.is_some() {
            return Err(Error::unsupported_operator(
                node.domain.as_deref(),
                &node.operator_type,
                version,
                "operator domain is not supported",
            ));
        }

        validate_schema(node, version)?;
        let attributes = Attributes::create(node)?;
        let unsupported = match node.operator_type.as_str() {
            "Gelu"
                if !matches!(
                    attributes.string("approximate", "none")?.as_str(),
                    "tanh" | "none"
                ) =>
            {
                Some("GELU approximation must be none or tanh")
            }
            "Swish" if attributes.float("alpha", 1.0)? != 1.0 => {
                Some("only Swish alpha=1 is implemented")
            }
            _ => None,
        };
        if let Some(reason) = unsupported {
            return Err(Error::unsupported_operator(
                node.domain.as_deref(),
                &node.operator_type,
                version,
                reason,
            ));
        }
    }
    Ok(opsets)
}
