use crate::{
    backend::cuda::provider_selection::{KernelChoice, Operand, matmul_kernel},
    data_type::DataType,
    layout::Layout,
    shape::Shape,
};

fn layout(dimensions: &[i64]) -> Layout {
    Layout::from_shape(&Shape::create(dimensions.to_vec()).unwrap())
}

fn operand(data_type: DataType, layout: &Layout) -> Operand<'_> {
    Operand { data_type, layout }
}

#[test]
fn test_matmul_prefers_libraries_or_the_general_kernel_by_operand() {
    let matrix = layout(&[4, 8]);
    let weight = layout(&[16, 8]).transpose(0, 1).unwrap();
    let batched = layout(&[2, 4, 8]);
    let every_other = layout(&[4, 16]).slice(1, 0, 8, 2).unwrap();
    for data_type in [DataType::F16, DataType::Bf16, DataType::F32] {
        for lhs in [&matrix, &batched] {
            assert_eq!(
                matmul_kernel(operand(data_type, lhs), operand(data_type, &weight)),
                KernelChoice::Library
            );
        }
    }
    assert_eq!(
        matmul_kernel(
            operand(DataType::F64, &matrix),
            operand(DataType::F64, &weight)
        ),
        KernelChoice::General
    );
    assert_eq!(
        matmul_kernel(
            operand(DataType::F32, &every_other),
            operand(DataType::F32, &weight)
        ),
        KernelChoice::General
    );
}
