use crate::{data_type::DataType, device::Device, error::Error, storage::cpu::CpuStorage};

#[test]
fn test_constructors_reject_non_cpu_device() {
    assert!(matches!(
        CpuStorage::create_zeroed(Device::Cuda(0), DataType::F32, 1),
        Err(Error::DeviceMismatch { .. })
    ));
    assert!(matches!(
        CpuStorage::create_from_bytes(Device::Cuda(0), DataType::F32, &[0; 4]),
        Err(Error::DeviceMismatch { .. })
    ));
}
