use crate::{
    backend::cuda::{
        cutensor::descriptor::cutensor_tensor_alignment,
        tensor_region::{CudaTensorRegion, CudaTensorSource},
    },
    data_type::DataType,
    error::Result,
    graph::NumericalPolicy,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CutensorContractionPlanKey {
    numerics: NumericalPolicy,
    data_type: DataType,
    a: CutensorTensorKey,
    a_mode: Vec<i64>,
    b: CutensorTensorKey,
    b_mode: Vec<i64>,
    c: CutensorTensorKey,
    c_mode: Vec<i64>,
    output: CutensorTensorKey,
    d_mode: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CutensorTensorKey {
    dimensions: Vec<i64>,
    strides: Vec<i64>,
    alignment: u32,
}

impl CutensorContractionPlanKey {
    pub(crate) fn accounted_bytes(&self) -> usize {
        [&self.a_mode, &self.b_mode, &self.c_mode, &self.d_mode]
            .into_iter()
            .map(Vec::capacity)
            .chain(
                [&self.a, &self.b, &self.c, &self.output]
                    .into_iter()
                    .flat_map(|key| [key.dimensions.capacity(), key.strides.capacity()]),
            )
            .fold(size_of::<Self>(), |bytes, capacity| {
                bytes.saturating_add(capacity.saturating_mul(size_of::<i64>()))
            })
    }
}

pub fn cutensor_contraction_plan_key(
    a: &dyn CudaTensorSource,
    a_mode: &[i64],
    b: &dyn CudaTensorSource,
    b_mode: &[i64],
    c: &dyn CudaTensorSource,
    c_mode: &[i64],
    d_mode: &[i64],
    output: CudaTensorRegion<'_>,
) -> Result<CutensorContractionPlanKey> {
    Ok(CutensorContractionPlanKey {
        numerics: NumericalPolicy::current(),
        data_type: DataType::F32,
        a: cutensor_region_key(a.tensor_region()?),
        a_mode: a_mode.to_vec(),
        b: cutensor_region_key(b.tensor_region()?),
        b_mode: b_mode.to_vec(),
        c: cutensor_region_key(c.tensor_region()?),
        c_mode: c_mode.to_vec(),
        output: cutensor_region_key(output),
        d_mode: d_mode.to_vec(),
    })
}

fn cutensor_region_key(tensor: CudaTensorRegion<'_>) -> CutensorTensorKey {
    CutensorTensorKey {
        dimensions: tensor.shape().dimensions().to_vec(),
        strides: tensor.layout().strides().to_vec(),
        alignment: cutensor_tensor_alignment(tensor),
    }
}
