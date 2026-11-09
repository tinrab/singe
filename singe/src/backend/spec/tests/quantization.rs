use crate::{
    backend::{
        registry::PackedFormat,
        spec::{unpacked_shape, unpacked_value},
    },
    data_type::DataType,
    shape::Shape,
};

#[test]
fn test_packed_values_decode_both_nibbles_of_each_byte() {
    // 0x8F holds 15 in its low nibble and 8 in its high one.
    let byte = 0x8F;
    assert_eq!(
        unpacked_value(byte, 0, DataType::U8, PackedFormat::Int4),
        15.0
    );
    assert_eq!(
        unpacked_value(byte, 1, DataType::U8, PackedFormat::Int4),
        8.0
    );
    assert_eq!(
        unpacked_value(byte, 0, DataType::I8, PackedFormat::Int4),
        -1.0
    );
    assert_eq!(
        unpacked_value(byte, 1, DataType::I8, PackedFormat::Int4),
        -8.0
    );
    assert_eq!(
        unpacked_value(0x07, 0, DataType::I8, PackedFormat::Int4),
        7.0
    );

    let fp4 = (0..16_u8)
        .map(|code| unpacked_value(code, 0, DataType::U8, PackedFormat::Float4E2M1))
        .collect::<Vec<_>>();
    assert_eq!(
        fp4,
        [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
        ]
    );
}

#[test]
fn test_packed_dequantization_doubles_the_last_axis() {
    let shape = Shape::create([3, 4]).unwrap();
    assert_eq!(unpacked_shape(&shape, None).unwrap(), shape);
    assert_eq!(
        unpacked_shape(&shape, Some(PackedFormat::Int4)).unwrap(),
        Shape::create([3, 8]).unwrap()
    );
    assert!(unpacked_shape(&Shape::scalar(), Some(PackedFormat::Int4)).is_err());
}
