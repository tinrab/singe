use sha2::{Digest, Sha256};

use crate::{
    context::Context,
    device::DeviceProperties,
    error::Result,
    frontend::{graph::Graph, operation::CompileConfig},
    version,
};

impl Graph {
    pub fn key_digest(&self) -> Result<[u8; 32]> {
        let serialized = serde_json::to_vec(self)?;
        Ok(Sha256::digest(serialized).into())
    }

    pub fn key(&self) -> Result<u64> {
        let digest = self.key_digest()?;
        let mut key = [0_u8; 8];
        key.copy_from_slice(&digest[..8]);
        Ok(u64::from_be_bytes(key))
    }

    pub fn plan_cache_key_digest(&self, compile_config: &CompileConfig) -> Result<[u8; 32]> {
        self.plan_cache_key_digest_with_context(compile_config, None)
    }

    pub(crate) fn plan_cache_key_digest_with_context(
        &self,
        compile_config: &CompileConfig,
        ctx: Option<&Context>,
    ) -> Result<[u8; 32]> {
        let graph_digest = self.key_digest()?;
        let context_device_properties_json = if self.device_properties_json.is_none() {
            ctx.map(|ctx| DeviceProperties::create(ctx)?.json_representation())
                .transpose()?
        } else {
            None
        };
        let cache_input = serde_json::json!({
            "schema": "singe-cudnn-frontend-plan-cache-v1",
            "graph_digest": hex::encode(graph_digest),
            "cudnn_version": version()?.raw(),
            "context_device_properties_json": context_device_properties_json,
            "compile_config": compile_config.cache_key_json()?,
        });
        let serialized = serde_json::to_vec(&cache_input)?;
        Ok(Sha256::digest(serialized).into())
    }

    pub fn plan_cache_key(&self, compile_config: &CompileConfig) -> Result<String> {
        Ok(hex::encode(self.plan_cache_key_digest(compile_config)?))
    }

    pub(crate) fn plan_cache_key_with_context(
        &self,
        compile_config: &CompileConfig,
        ctx: Option<&Context>,
    ) -> Result<String> {
        Ok(hex::encode(
            self.plan_cache_key_digest_with_context(compile_config, ctx)?,
        ))
    }
}
