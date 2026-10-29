mod tagged;

use std::{collections::HashMap, mem};

use serde_json::{Map, Value};

pub use crate::frontend::graph::key::tagged::to_value;
use crate::{error::Result, frontend::error::Error as FrontendError};

type TensorIds = HashMap<String, usize>;

/// Canonicalizes typed identities while keeping all scalar literals and ordinary strings intact.
/// Engine-choice cache hits rebuild descriptors and binding maps for the caller's actual IDs.
pub fn canonicalize(value: &mut Value) -> Result<()> {
    let mut ids = TensorIds::new();
    assign_references(value, &mut ids);
    loop {
        let previous = ids.len();
        visit_graphs(value, &mut |graph| assign_metadata(graph, &mut ids));
        if ids.len() == previous {
            break;
        }
    }
    prune_unused(value, &ids);
    rewrite(value, &ids)
}

fn register(value: &str, ids: &mut TensorIds) {
    if tagged::is_tensor_id(value) {
        let next = ids.len();
        ids.entry(value.to_owned()).or_insert(next);
    }
}

fn assign_references(value: &Value, ids: &mut TensorIds) {
    match value {
        Value::Object(object) if object.contains_key(tagged::GRAPH) => {
            // Tensor declaration order depends on original IDs. Operations establish their roles first.
            assign_references(&object["operations"], ids);
        }
        Value::Object(object) => {
            let mut fields = object.iter().collect::<Vec<_>>();
            fields.sort_unstable_by(|left, right| left.0.cmp(right.0));
            for (key, value) in fields {
                register(key, ids);
                assign_references(value, ids);
            }
        }
        Value::Array(values) => {
            for value in values {
                assign_references(value, ids);
            }
        }
        Value::String(value) => register(value, ids),
        _ => {}
    }
}

fn visit_graphs(value: &Value, visit: &mut impl FnMut(&Map<String, Value>)) {
    match value {
        Value::Object(object) => {
            if object.contains_key(tagged::GRAPH) {
                visit(object);
                visit_graphs(&object["operations"], visit);
            } else {
                let mut fields = object.iter().collect::<Vec<_>>();
                fields.sort_unstable_by(|left, right| left.0.cmp(right.0));
                for (_, value) in fields {
                    visit_graphs(value, visit);
                }
            }
        }
        Value::Array(values) => {
            for value in values {
                visit_graphs(value, visit);
            }
        }
        _ => {}
    }
}

fn assign_metadata(graph: &Map<String, Value>, ids: &mut TensorIds) {
    if let Some(Value::Object(tensors)) = graph.get("tensors") {
        let mut active = tensors
            .iter()
            .filter_map(|(id, spec)| ids.get(id).copied().map(|id| (id, spec)))
            .collect::<Vec<_>>();
        active.sort_unstable_by_key(|(id, _)| *id);
        for (_, spec) in active {
            assign_references(spec, ids);
        }
    }
    if let Some(Value::Array(aliases)) = graph.get("storage_views") {
        let mut active = aliases
            .iter()
            .filter_map(|alias| {
                let target = alias.get("target")?.as_str()?;
                ids.get(target).copied().map(|id| (id, alias))
            })
            .collect::<Vec<_>>();
        active.sort_unstable_by_key(|(id, _)| *id);
        for (_, alias) in active {
            assign_references(alias, ids);
        }
    }
}

fn referenced(value: &Value, ids: &TensorIds) -> bool {
    value.as_str().is_some_and(|id| ids.contains_key(id))
}

fn prune_unused(value: &mut Value, ids: &TensorIds) {
    match value {
        Value::Object(object) => {
            if object.contains_key(tagged::GRAPH) {
                for field in ["tensors", "runtime_shape_constraints"] {
                    if let Some(Value::Object(tensors)) = object.get_mut(field) {
                        tensors.retain(|id, _| ids.contains_key(id));
                    }
                }
                if let Some(Value::Array(aliases)) = object.get_mut("storage_views") {
                    aliases.retain(|alias| referenced(&alias["target"], ids));
                    aliases.sort_unstable_by_key(|alias| {
                        alias["target"].as_str().and_then(|id| ids.get(id)).copied()
                    });
                }
                if let Some(Value::Array(aligned)) = object.get_mut("vector_aligned_tensors") {
                    aligned.retain(|id| referenced(id, ids));
                    aligned
                        .sort_unstable_by_key(|id| id.as_str().and_then(|id| ids.get(id)).copied());
                }
            }
            for value in object.values_mut() {
                prune_unused(value, ids);
            }
        }
        Value::Array(values) => {
            for value in values {
                prune_unused(value, ids);
            }
        }
        _ => {}
    }
}

fn rewrite_text(value: String, ids: &TensorIds) -> Result<String> {
    if tagged::is_tensor_id(&value) {
        ids.get(&value).map(usize::to_string).ok_or(
            FrontendError::InternalStateMissing {
                name: "compilation-cache tensor identity",
            }
            .into(),
        )
    } else if let Some(literal) = tagged::unescape(&value) {
        Ok(literal.to_owned())
    } else {
        Ok(value)
    }
}

fn rewrite(value: &mut Value, ids: &TensorIds) -> Result<()> {
    match value {
        Value::Object(object) => {
            let mut fields = Vec::with_capacity(object.len());
            for (key, mut value) in mem::take(object) {
                if key == tagged::GRAPH {
                    continue;
                }
                rewrite(&mut value, ids)?;
                fields.push((rewrite_text(key, ids)?, value));
            }
            // Keep identity deterministic even if serde_json's preserve_order feature is enabled.
            fields.sort_unstable_by(|left, right| left.0.cmp(&right.0));
            object.extend(fields);
        }
        Value::Array(values) => {
            for value in values {
                rewrite(value, ids)?;
            }
        }
        Value::String(value) => {
            *value = rewrite_text(mem::take(value), ids)?;
        }
        _ => {}
    }
    Ok(())
}
