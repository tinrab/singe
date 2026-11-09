use singe_cudnn::backend::tensor::Tensor as CudnnBackendTensor;

#[derive(Debug)]
pub struct CudnnBackendConv2dBiasReluTensors {
    pub(crate) input: CudnnBackendTensor,
    pub(crate) weight: CudnnBackendTensor,
    pub(crate) bias: CudnnBackendTensor,
    pub(crate) conv_output: CudnnBackendTensor,
    pub(crate) biased: CudnnBackendTensor,
    pub(crate) output: CudnnBackendTensor,
}

impl CudnnBackendConv2dBiasReluTensors {
    pub fn new(
        input: CudnnBackendTensor,
        weight: CudnnBackendTensor,
        bias: CudnnBackendTensor,
        conv_output: CudnnBackendTensor,
        biased: CudnnBackendTensor,
        output: CudnnBackendTensor,
    ) -> Self {
        Self {
            input,
            weight,
            bias,
            conv_output,
            biased,
            output,
        }
    }
}

#[derive(Debug)]
pub struct CudnnBackendMatmulAddTensors {
    pub(crate) a: CudnnBackendTensor,
    pub(crate) b: CudnnBackendTensor,
    pub(crate) bias: CudnnBackendTensor,
    pub(crate) matmul_output: CudnnBackendTensor,
    pub(crate) output: CudnnBackendTensor,
}

impl CudnnBackendMatmulAddTensors {
    pub fn new(
        a: CudnnBackendTensor,
        b: CudnnBackendTensor,
        bias: CudnnBackendTensor,
        matmul_output: CudnnBackendTensor,
        output: CudnnBackendTensor,
    ) -> Self {
        Self {
            a,
            b,
            bias,
            matmul_output,
            output,
        }
    }
}

#[derive(Debug)]
pub struct CudnnBackendMatmulAddReluTensors {
    pub(crate) a: CudnnBackendTensor,
    pub(crate) b: CudnnBackendTensor,
    pub(crate) bias: CudnnBackendTensor,
    pub(crate) matmul_output: CudnnBackendTensor,
    pub(crate) biased: CudnnBackendTensor,
    pub(crate) output: CudnnBackendTensor,
}

impl CudnnBackendMatmulAddReluTensors {
    pub fn new(
        a: CudnnBackendTensor,
        b: CudnnBackendTensor,
        bias: CudnnBackendTensor,
        matmul_output: CudnnBackendTensor,
        biased: CudnnBackendTensor,
        output: CudnnBackendTensor,
    ) -> Self {
        Self {
            a,
            b,
            bias,
            matmul_output,
            biased,
            output,
        }
    }
}
