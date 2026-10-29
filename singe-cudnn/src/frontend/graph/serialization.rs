use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::{
    backend::device::{DeviceProperties, DevicePropertiesSource},
    context::Context,
    error::Result,
    frontend::{
        graph::{Graph, key},
        plan::CompileConfig,
    },
    version,
};

/// Whether a graph key includes tensor dimensions and strides.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GraphKeyShapePolicy {
    /// Omits shapes for dynamic-shape graphs and includes them otherwise, matching the C++ frontend.
    #[default]
    Auto,
    /// Distinguishes concrete tensor layouts.
    Include,
    /// Omits tensor dimensions and strides, including those in nested graphs.
    /// Operation attributes, tensor encodings and runtime shape constraints remain included.
    Exclude,
}

#[derive(Clone, Copy)]
enum KeyProjection {
    WithoutShapes,
    Compilation(GraphKeyShapePolicy),
}

impl Graph {
    /// Returns a SHA-256 graph key with the requested tensor-shape policy.
    /// The policy affects the key only; it does not enable dynamic compilation or runtime overrides.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph cannot be serialized for hashing.
    pub fn key_digest(&self, shape_policy: GraphKeyShapePolicy) -> Result<[u8; 32]> {
        let serialized = match self.resolve_key_shape_policy(shape_policy) {
            GraphKeyShapePolicy::Exclude => {
                let mut graph = serde_json::to_value(self)?;
                project_key(&mut graph, KeyProjection::WithoutShapes);
                serde_json::to_vec(&graph)?
            }
            _ => serde_json::to_vec(self)?,
        };
        Ok(Sha256::digest(serialized).into())
    }

    /// Returns the first eight bytes of [`Self::key_digest`] as a big-endian integer.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph cannot be serialized for hashing.
    pub fn key(&self, shape_policy: GraphKeyShapePolicy) -> Result<u64> {
        Ok(short_key(self.key_digest(shape_policy)?))
    }

    /// Returns the exact compilation-cache identity used for live or deviceless compilation.
    /// The key includes the target, cuDNN version and compile policy.
    /// Tensor dimensions and strides are omitted when dynamic-shape compilation is enabled.
    /// Diagnostic names, runtime scalar defaults and mutable kernel-cache contents do not affect this key.
    /// Matmul zero-padding signs and pointwise NaN settings unused by native lowering are canonicalized.
    /// Attention auxiliary presence is determined by output identities rather than saved flags.
    /// Tensor identities follow operation-use order rather than caller or generated ID numbers.
    /// A live context supplies device properties when the graph has no attached property snapshot.
    /// Includes the same resolved SM version used by attention implementation selection.
    ///
    /// # Errors
    ///
    /// Returns an error if graph serialization or identity canonicalization fails, target properties or the compilation SM version cannot be resolved, the attention workspace-limit environment value is invalid, the loaded cuDNN version is invalid, or compile-policy serialization fails.
    pub fn plan_cache_key_digest(
        &self,
        compile_config: &CompileConfig,
        context: Option<&Context>,
    ) -> Result<[u8; 32]> {
        let mut graph = key::to_value(self)?;
        let shape_policy = self.resolve_key_shape_policy(GraphKeyShapePolicy::Auto);
        project_key(&mut graph, KeyProjection::Compilation(shape_policy));
        key::canonicalize(&mut graph)?;
        let graph_digest: [u8; 32] = Sha256::digest(serde_json::to_vec(&graph)?).into();
        let context_device_properties_json = if self.device_properties_json.is_none() {
            context
                .map(|context| {
                    DeviceProperties::create(DevicePropertiesSource::Device(
                        context.cuda_context().device(),
                    ))?
                    .json_representation()
                })
                .transpose()?
        } else {
            None
        };
        let cache_input = serde_json::json!({
            "schema": "singe-cudnn-frontend-plan-cache-v36",
            "resolved_sm_version": self.compilation_sm_version(context)?,
            "attention_dp_workspace_limit": self.attention_workspace_environment(context)?,
            "graph_digest": hex::encode(graph_digest),
            "cudnn_version": version()?.raw(),
            "context_device_properties_json": context_device_properties_json,
            "compile_config": compile_config.cache_key_json()?,
        });
        let serialized = serde_json::to_vec(&cache_input)?;
        Ok(Sha256::digest(serialized).into())
    }

    /// Returns the hexadecimal compilation-cache key from [`Self::plan_cache_key_digest`].
    ///
    /// # Errors
    ///
    /// Returns an error if graph serialization or identity canonicalization fails, target properties or the compilation SM version cannot be resolved, the attention workspace-limit environment value is invalid, the loaded cuDNN version is invalid, or compile-policy serialization fails.
    pub fn plan_cache_key(
        &self,
        compile_config: &CompileConfig,
        context: Option<&Context>,
    ) -> Result<String> {
        Ok(hex::encode(
            self.plan_cache_key_digest(compile_config, context)?,
        ))
    }

    fn resolve_key_shape_policy(&self, shape_policy: GraphKeyShapePolicy) -> GraphKeyShapePolicy {
        match shape_policy {
            GraphKeyShapePolicy::Auto if self.config.dynamic_shape_enabled => {
                GraphKeyShapePolicy::Exclude
            }
            GraphKeyShapePolicy::Auto => GraphKeyShapePolicy::Include,
            policy => policy,
        }
    }
}

fn short_key(digest: [u8; 32]) -> u64 {
    let mut key = [0_u8; 8];
    key.copy_from_slice(&digest[..8]);
    u64::from_be_bytes(key)
}

fn project_key(value: &mut Value, projection: KeyProjection) {
    match value {
        Value::Object(object) => {
            let compilation = matches!(projection, KeyProjection::Compilation(_));
            if compilation {
                object.remove("name");
                object.remove("kernel_cache_json");
                object.remove("aux_gradients");
                object.remove("aux_outputs");
                project_operation(object);
            }
            // Operand groups may also use `tensors`; only graphs own tensor declarations.
            if object.get("operations").is_some_and(Value::is_array)
                && let Some(Value::Object(tensors)) = object.get_mut("tensors")
            {
                for tensor in tensors.values_mut() {
                    if !matches!(
                        projection,
                        KeyProjection::Compilation(GraphKeyShapePolicy::Include)
                    ) {
                        tensor["shape"]["dimensions"] = Value::Array(Vec::new());
                        tensor["shape"]["strides"] = Value::Array(Vec::new());
                    }
                    if compilation {
                        if let Some(scalar) = tensor["placement"].get_mut("RuntimeScalar") {
                            scalar["default"] = Value::Null;
                        }
                        // Native tensors ignore the vector axis when the lane count is omitted or one.
                        if tensor["vector_count"].is_null() || tensor["vector_count"] == 1 {
                            tensor["vector_count"] = Value::Null;
                            tensor["vectorized_dimension"] = Value::Null;
                        }
                        if tensor["reordering"] == "None" {
                            tensor["reordering"] = Value::Null;
                        }
                        // A unit multiplier has the same native meaning as an omitted value.
                        if tensor["ragged_offset_multiplier"] == 1 {
                            tensor["ragged_offset_multiplier"] = Value::Null;
                        }
                    }
                }
            }
            // Reducer bodies and attention score modifiers can contain nested graphs.
            for value in object.values_mut() {
                project_key(value, projection);
            }
        }
        Value::Array(values) => {
            for value in values {
                project_key(value, projection);
            }
        }
        _ => {}
    }
}

fn project_operation(object: &mut Map<String, Value>) {
    match object.get("kind").and_then(Value::as_str) {
        Some("Matmul") => {
            if let Some(padding) = object
                .get_mut("operation")
                .and_then(|operation| operation.pointer_mut("/config/padding_value"))
                && padding.as_f64() == Some(0.0)
            {
                // Both signs omit the native padding attribute.
                *padding = Value::from(0.0);
            }
        }
        Some("Pointwise") => {
            if let Some(Value::Object(variants)) = object.get_mut("operation") {
                for parameters in variants.values_mut() {
                    let relu = parameters
                        .get("activation")
                        .is_some_and(|activation| activation.get("Relu").is_some())
                        || matches!(
                            parameters.get("mode").and_then(Value::as_str),
                            Some("ReluFwd" | "ReluBwd")
                        );
                    if !relu && let Some(parameters) = parameters.as_object_mut() {
                        parameters.remove("nan_propagation");
                    }
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::tensor::{Shape, TensorSpec},
        data_type::DataType,
        frontend::graph::GraphConfig,
    };

    fn graph_with_shape(dimensions: [i64; 2], dynamic_shape: bool) -> Result<Graph> {
        let mut graph = Graph::with_config(GraphConfig::new().with_dynamic_shape(dynamic_shape));
        graph.tensor(TensorSpec::new(
            DataType::F32,
            Shape::contiguous(dimensions)?,
        ));
        Ok(graph)
    }

    #[test]
    fn test_auto_policy_includes_shapes_for_static_graphs() -> Result<()> {
        let graph = graph_with_shape([2, 3], false)?;
        assert_eq!(
            graph.key_digest(GraphKeyShapePolicy::Auto)?,
            graph.key_digest(GraphKeyShapePolicy::Include)?
        );
        Ok(())
    }

    #[test]
    fn test_auto_policy_excludes_shapes_for_dynamic_graphs() -> Result<()> {
        let graph = graph_with_shape([2, 3], true)?;
        assert_eq!(
            graph.key_digest(GraphKeyShapePolicy::Auto)?,
            graph.key_digest(GraphKeyShapePolicy::Exclude)?
        );
        Ok(())
    }
}
