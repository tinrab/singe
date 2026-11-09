use crate::{data_type::DataType, device::Device, error::Error, storage::cpu};

#[test]
fn test_cpu_storage_zeroes() {
    let storage = cpu::CpuStorage::create_zeroed(Device::Cpu, DataType::F32, 4).unwrap();
    assert_eq!(storage.byte_size(), 16);
    assert!(storage.as_slice().unwrap().iter().all(|&b| b == 0));
}

#[test]
fn test_cpu_storage_from_slice() {
    let data: Vec<u8> = vec![0, 0, 128, 63];
    let storage = cpu::CpuStorage::create_from_bytes(Device::Cpu, DataType::F32, &data).unwrap();
    assert_eq!(storage.byte_size(), 4);
}

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
fn test_cpu_storage_copy_to_host() {
    let mut storage = cpu::CpuStorage::create(DataType::I32, 2).unwrap();
    let typed = storage.as_typed_slice_mut::<i32>().unwrap();
    typed[0] = 42;
    typed[1] = -7;

    let mut dst = vec![0_u8; 8];
    storage.copy_to_host(&mut dst).unwrap();
    let dst_i32 = dst
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| i32::from_ne_bytes(*bytes))
        .collect::<Vec<_>>();
    assert_eq!(dst_i32, [42, -7]);
}

#[test]
fn test_cpu_storage_typed_slice_rejects_partial_elements() {
    let mut storage = cpu::CpuStorage::create(DataType::U8, 6).unwrap();

    assert!(matches!(
        storage.as_typed_slice::<f32>(),
        Err(Error::ByteLengthMismatch {
            element_size: 4,
            actual: 6,
            ..
        })
    ));
    assert!(matches!(
        storage.as_typed_slice_mut::<f32>(),
        Err(Error::ByteLengthMismatch {
            element_size: 4,
            actual: 6,
            ..
        })
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
