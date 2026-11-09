#[cfg(test)]
use crate::graph::{
    ValueId,
    guards::{ConstantGuard, ConstantGuardKind},
};
use crate::{
    data_type::DataType, error::Result, graph::guards::StaticInputStorageGuard, layout::Layout,
    tensor::Tensor,
};

pub fn storage_span_bytes(layout: &Layout, data_type: DataType) -> Result<usize> {
    layout.storage_span_byte_size(data_type.size_bytes())
}

#[cfg(test)]
pub fn constant_guard(value: ValueId, tensor: &Tensor) -> Result<ConstantGuard> {
    let layout = Layout::from_tensor(tensor);
    let byte_size = tensor.byte_size()?;
    if tensor.device().is_cpu()
        && tensor.is_contiguous()
        && tensor.storage_offset() == 0
        && tensor.storage().byte_size() == byte_size
    {
        let mut bytes = vec![0_u8; byte_size];
        tensor.storage().copy_to_host(&mut bytes)?;
        return Ok(ConstantGuard {
            value,
            data_type: tensor.data_type(),
            layout,
            kind: ConstantGuardKind::HostContent {
                digest: digest_bytes(&bytes),
            },
        });
    }

    Ok(ConstantGuard {
        value,
        data_type: tensor.data_type(),
        layout,
        kind: ConstantGuardKind::GraphLocal {
            device: tensor.device(),
        },
    })
}

pub fn static_input_storage_guard(tensor: &Tensor) -> StaticInputStorageGuard {
    StaticInputStorageGuard {
        storage_identity: tensor.storage().storage_identity(),
        data_pointer_identity: tensor.storage().data_pointer_identity(),
        offset: tensor.storage_offset(),
        storage_byte_size: tensor.storage().byte_size(),
        version: tensor.version(),
    }
}

#[cfg(test)]
fn digest_bytes(bytes: &[u8]) -> u64 {
    let mut state = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        state ^= u64::from(*byte);
        state = state.wrapping_mul(0x0000_0100_0000_01b3);
    }
    state
}
