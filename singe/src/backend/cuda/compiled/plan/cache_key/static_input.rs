use crate::{error::Result, graph::ir::Graph};

/// Static CUDA input storage identity material included in cache keys.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CudaStaticInputIdentityKey {
    /// Static input name.
    input: String,
    /// Process-local storage owner identity.
    storage_identity: usize,
    /// Process-local allocation data pointer identity.
    data_pointer_identity: Option<usize>,
    /// `CudaGraphValue` storage element offset.
    offset: usize,
    /// Backing storage allocation byte size.
    storage_byte_size: usize,
    /// Storage write version.
    version: u64,
}

impl CudaStaticInputIdentityKey {
    fn new(
        input: impl Into<String>,
        storage_identity: usize,
        data_pointer_identity: Option<usize>,
        offset: usize,
        storage_byte_size: usize,
        version: u64,
    ) -> Self {
        Self {
            input: input.into(),
            storage_identity,
            data_pointer_identity,
            offset,
            storage_byte_size,
            version,
        }
    }

    fn input(&self) -> &str {
        &self.input
    }
}

pub fn static_input_identity_key_material_from_ir(
    ir: &Graph,
) -> Result<Vec<CudaStaticInputIdentityKey>> {
    let mut keys = Vec::new();
    for (name, tensor) in ir.static_input_initializer_tensors()? {
        keys.push(CudaStaticInputIdentityKey::new(
            name,
            tensor.storage().storage_identity(),
            tensor.storage().data_pointer_identity(),
            tensor.storage_offset(),
            tensor.storage().byte_size(),
            tensor.version(),
        ));
    }

    keys.sort_by(|lhs, rhs| lhs.input().cmp(rhs.input()));
    Ok(keys)
}
