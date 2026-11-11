use crate::{data_type::DataType, device::Device, error::Error, storage::cpu};

#[test]
fn test_cpu_storage_rejects_byte_size_overflow() {
    assert!(matches!(
        cpu::CpuStorage::create(DataType::F32, usize::MAX),
        Err(Error::OutOfRange { name }) if name == "cpu storage byte size"
    ));
}

#[test]
fn test_cpu_storage_rejects_partial_element_bytes() {
    assert!(matches!(
        cpu::CpuStorage::create_from_bytes(Device::Cpu, DataType::F32, &[0, 1, 2]),
        Err(Error::ByteLengthMismatch {
            name,
            element_size: 4,
            actual: 3,
        }) if name == "cpu storage bytes"
    ));
}

#[test]
fn test_cpu_storage_boolean_views_find_invalid_bytes_past_the_first_chunk() {
    let mut bytes = vec![1_u8; 10_000];
    bytes[9_000] = 2;
    assert!(matches!(
        cpu::CpuStorage::create_from_bytes(Device::Cpu, DataType::Bool, &bytes),
        Err(Error::InvalidArgument { .. })
    ));
    let storage = cpu::CpuStorage::create_from_bytes(Device::Cpu, DataType::U8, &bytes).unwrap();
    assert!(matches!(
        storage.as_typed_slice::<bool>(),
        Err(Error::InvalidArgument { name }) if name == "boolean byte 9000 (value 2)"
    ));
    bytes[9_000] = 0;
    let storage = cpu::CpuStorage::create_from_bytes(Device::Cpu, DataType::U8, &bytes).unwrap();
    assert_eq!(storage.as_typed_slice::<bool>().unwrap().len(), 10_000);
}
