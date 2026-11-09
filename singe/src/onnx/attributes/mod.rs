use std::collections::HashMap;

use singe_onnx::{AttributeValue, Node};

use crate::onnx::{
    error::{Error, Result},
    utility::node_label,
};

pub struct Attributes<'a> {
    node: &'a Node,
    values: HashMap<&'a str, &'a AttributeValue>,
}

impl<'a> Attributes<'a> {
    pub(crate) fn create(node: &'a Node) -> Result<Self> {
        let mut values = HashMap::new();
        for attribute in &node.attributes {
            if values
                .insert(attribute.name.as_str(), &attribute.value)
                .is_some()
            {
                return Err(Error::invalid_attribute(
                    node_label(node),
                    &attribute.name,
                    "duplicate attribute",
                ));
            }
        }
        Ok(Self { node, values })
    }

    pub(crate) fn int(&self, name: &str, default: i64) -> Result<i64> {
        match self.values.get(name) {
            Some(AttributeValue::Int(value)) => Ok(*value),
            Some(_) => Err(self.invalid_type(name, "int")),
            None => Ok(default),
        }
    }

    pub(crate) fn float(&self, name: &str, default: f32) -> Result<f32> {
        match self.values.get(name) {
            Some(AttributeValue::Float(value)) => Ok(*value),
            Some(_) => Err(self.invalid_type(name, "float")),
            None => Ok(default),
        }
    }

    pub(crate) fn string(&self, name: &str, default: &str) -> Result<String> {
        match self.values.get(name) {
            Some(AttributeValue::String(value)) => Ok(value.clone()),
            Some(_) => Err(self.invalid_type(name, "string")),
            None => Ok(default.into()),
        }
    }

    pub(crate) fn ints(&self, name: &str) -> Result<Option<Vec<i64>>> {
        match self.values.get(name) {
            Some(AttributeValue::Ints(value)) => Ok(Some(value.clone())),
            Some(_) => Err(self.invalid_type(name, "ints")),
            None => Ok(None),
        }
    }

    pub(crate) fn ints_or(&self, name: &str, default: &[i64]) -> Result<Vec<i64>> {
        Ok(self.ints(name)?.unwrap_or_else(|| default.to_vec()))
    }

    pub(crate) fn value(&self, name: &str) -> Option<&'a AttributeValue> {
        self.values.get(name).copied()
    }

    fn invalid_type(&self, name: &str, expected: &str) -> Error {
        Error::invalid_attribute(node_label(self.node), name, format!("expected {expected}"))
    }
}

#[cfg(test)]
mod tests;
