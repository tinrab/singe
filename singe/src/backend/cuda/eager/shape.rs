use crate::tensor::Tensor;

pub fn noop_contiguous(input: &Tensor) -> Option<Tensor> {
    (input.is_contiguous() && input.storage_offset() == 0).then(|| input.clone())
}
