mod info;
mod raw;
mod validation;

use singe_cuda::{
    data_type::DataTypeLike,
    memory::DeviceMemory,
    types::{Complex32, Complex64},
};

use crate::{
    context::Context,
    error::Result,
    layout::{
        ByteWorkspaceMut, MatrixMut, MatrixRef, StridedBatchedMatrixMut, StridedBatchedMatrixRef,
        StridedBatchedVectorMut, StridedBatchedVectorRef, WorkspaceSizes,
    },
    params::Params,
    types::{EigenMode, SvdMode, TruncatedSvdMode},
};

use raw::*;

pub use info::GesvdjInfo;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gesvd {
    pub job_u: SvdMode,
    pub job_vt: SvdMode,
    pub rows: usize,
    pub columns: usize,
}

impl Gesvd {
    pub fn new(job_u: SvdMode, job_vt: SvdMode, rows: usize, columns: usize) -> Self {
        Self {
            job_u,
            job_vt,
            rows,
            columns,
        }
    }

    pub fn workspace_size<
        TA: DataTypeLike,
        TS: DataTypeLike,
        TU: DataTypeLike,
        TVT: DataTypeLike,
    >(
        self,
        ctx: &Context,
        params: &Params,
        input: GesvdInput<'_, TA, TS, TU, TVT>,
    ) -> Result<WorkspaceSizes> {
        xgesvd_buffer_size(
            ctx,
            params,
            self.job_u,
            self.job_vt,
            self.rows,
            self.columns,
            input.a,
            input.singular_values,
            input.left_vectors,
            input.right_vectors_transposed,
        )
    }

    pub fn execute<TA: DataTypeLike, TS: DataTypeLike, TU: DataTypeLike, TVT: DataTypeLike>(
        self,
        ctx: &Context,
        params: &Params,
        bindings: GesvdBindings<'_, TA, TS, TU, TVT>,
    ) -> Result<()> {
        xgesvd(
            ctx,
            params,
            self.job_u,
            self.job_vt,
            self.rows,
            self.columns,
            bindings.a,
            bindings.singular_values,
            bindings.left_vectors,
            bindings.right_vectors_transposed,
            bindings.workspace,
            bindings.dev_info,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GesvdInput<'a, TA, TS, TU, TVT> {
    pub a: MatrixRef<'a, TA>,
    pub singular_values: &'a DeviceMemory<TS>,
    pub left_vectors: Option<MatrixRef<'a, TU>>,
    pub right_vectors_transposed: Option<MatrixRef<'a, TVT>>,
}

#[derive(Debug)]
pub struct GesvdBindings<'a, TA, TS, TU, TVT> {
    pub a: MatrixMut<'a, TA>,
    pub singular_values: &'a mut DeviceMemory<TS>,
    pub left_vectors: Option<MatrixMut<'a, TU>>,
    pub right_vectors_transposed: Option<MatrixMut<'a, TVT>>,
    pub workspace: ByteWorkspaceMut<'a>,
    pub dev_info: &'a mut DeviceMemory<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gesvdp {
    pub mode: EigenMode,
    pub economy: bool,
    pub rows: usize,
    pub columns: usize,
}

impl Gesvdp {
    pub fn new(mode: EigenMode, economy: bool, rows: usize, columns: usize) -> Self {
        Self {
            mode,
            economy,
            rows,
            columns,
        }
    }

    pub fn workspace_size<
        TA: DataTypeLike,
        TS: DataTypeLike,
        TU: DataTypeLike,
        TV: DataTypeLike,
    >(
        self,
        ctx: &Context,
        params: &Params,
        input: GesvdpInput<'_, TA, TS, TU, TV>,
    ) -> Result<WorkspaceSizes> {
        xgesvdp_buffer_size(
            ctx,
            params,
            self.mode,
            self.economy,
            self.rows,
            self.columns,
            input.a,
            input.singular_values,
            input.left_vectors,
            input.right_vectors,
        )
    }

    pub fn execute<TA: DataTypeLike, TS: DataTypeLike, TU: DataTypeLike, TV: DataTypeLike>(
        self,
        ctx: &Context,
        params: &Params,
        bindings: GesvdpBindings<'_, TA, TS, TU, TV>,
    ) -> Result<()> {
        xgesvdp(
            ctx,
            params,
            self.mode,
            self.economy,
            self.rows,
            self.columns,
            bindings.a,
            bindings.singular_values,
            bindings.left_vectors,
            bindings.right_vectors,
            bindings.workspace,
            bindings.dev_info,
            bindings.residual,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GesvdpInput<'a, TA, TS, TU, TV> {
    pub a: MatrixRef<'a, TA>,
    pub singular_values: &'a DeviceMemory<TS>,
    pub left_vectors: Option<MatrixRef<'a, TU>>,
    pub right_vectors: Option<MatrixRef<'a, TV>>,
}

#[derive(Debug)]
pub struct GesvdpBindings<'a, TA, TS, TU, TV> {
    pub a: MatrixMut<'a, TA>,
    pub singular_values: &'a mut DeviceMemory<TS>,
    pub left_vectors: Option<MatrixMut<'a, TU>>,
    pub right_vectors: Option<MatrixMut<'a, TV>>,
    pub workspace: ByteWorkspaceMut<'a>,
    pub dev_info: &'a mut DeviceMemory<i32>,
    pub residual: Option<&'a mut f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gesvdr {
    pub left_vectors: TruncatedSvdMode,
    pub right_vectors: TruncatedSvdMode,
    pub rows: usize,
    pub columns: usize,
    pub rank: usize,
    pub oversamples: usize,
    pub iterations: usize,
}

impl Gesvdr {
    pub fn new(
        left_vectors: TruncatedSvdMode,
        right_vectors: TruncatedSvdMode,
        rows: usize,
        columns: usize,
        rank: usize,
        oversamples: usize,
        iterations: usize,
    ) -> Self {
        Self {
            left_vectors,
            right_vectors,
            rows,
            columns,
            rank,
            oversamples,
            iterations,
        }
    }

    pub fn workspace_size<
        TA: DataTypeLike,
        TS: DataTypeLike,
        TU: DataTypeLike,
        TV: DataTypeLike,
    >(
        self,
        ctx: &Context,
        params: &Params,
        input: GesvdrInput<'_, TA, TS, TU, TV>,
    ) -> Result<WorkspaceSizes> {
        xgesvdr_buffer_size(
            ctx,
            params,
            self.left_vectors,
            self.right_vectors,
            self.rows,
            self.columns,
            self.rank,
            self.oversamples,
            self.iterations,
            input.a,
            input.singular_values,
            input.left_vectors,
            input.right_vectors,
        )
    }

    pub fn execute<TA: DataTypeLike, TS: DataTypeLike, TU: DataTypeLike, TV: DataTypeLike>(
        self,
        ctx: &Context,
        params: &Params,
        bindings: GesvdrBindings<'_, TA, TS, TU, TV>,
    ) -> Result<()> {
        xgesvdr(
            ctx,
            params,
            self.left_vectors,
            self.right_vectors,
            self.rows,
            self.columns,
            self.rank,
            self.oversamples,
            self.iterations,
            bindings.a,
            bindings.singular_values,
            bindings.left_vectors,
            bindings.right_vectors,
            bindings.workspace,
            bindings.dev_info,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GesvdrInput<'a, TA, TS, TU, TV> {
    pub a: MatrixRef<'a, TA>,
    pub singular_values: &'a DeviceMemory<TS>,
    pub left_vectors: Option<MatrixRef<'a, TU>>,
    pub right_vectors: Option<MatrixRef<'a, TV>>,
}

#[derive(Debug)]
pub struct GesvdrBindings<'a, TA, TS, TU, TV> {
    pub a: MatrixMut<'a, TA>,
    pub singular_values: &'a mut DeviceMemory<TS>,
    pub left_vectors: Option<MatrixMut<'a, TU>>,
    pub right_vectors: Option<MatrixMut<'a, TV>>,
    pub workspace: ByteWorkspaceMut<'a>,
    pub dev_info: &'a mut DeviceMemory<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gesvdj {
    pub mode: EigenMode,
    pub economy: bool,
    pub rows: usize,
    pub columns: usize,
}

impl Gesvdj {
    pub fn new(mode: EigenMode, economy: bool, rows: usize, columns: usize) -> Self {
        Self {
            mode,
            economy,
            rows,
            columns,
        }
    }

    pub fn workspace_size_f32(
        self,
        ctx: &Context,
        input: GesvdjInput<'_, f32, f32>,
        params: &GesvdjInfo,
    ) -> Result<usize> {
        sgesvdj_buffer_size(
            ctx,
            self.mode,
            self.economy,
            self.rows,
            self.columns,
            input.a,
            input.singular_values,
            input.left_vectors,
            input.right_vectors,
            params,
        )
    }

    pub fn execute_f32(
        self,
        ctx: &Context,
        bindings: GesvdjBindings<'_, f32, f32>,
        params: &GesvdjInfo,
    ) -> Result<()> {
        sgesvdj(
            ctx,
            self.mode,
            self.economy,
            self.rows,
            self.columns,
            bindings.a,
            bindings.singular_values,
            bindings.left_vectors,
            bindings.right_vectors,
            bindings.workspace,
            bindings.dev_info,
            params,
        )
    }

    pub fn workspace_size_f64(
        self,
        ctx: &Context,
        input: GesvdjInput<'_, f64, f64>,
        params: &GesvdjInfo,
    ) -> Result<usize> {
        dgesvdj_buffer_size(
            ctx,
            self.mode,
            self.economy,
            self.rows,
            self.columns,
            input.a,
            input.singular_values,
            input.left_vectors,
            input.right_vectors,
            params,
        )
    }

    pub fn execute_f64(
        self,
        ctx: &Context,
        bindings: GesvdjBindings<'_, f64, f64>,
        params: &GesvdjInfo,
    ) -> Result<()> {
        dgesvdj(
            ctx,
            self.mode,
            self.economy,
            self.rows,
            self.columns,
            bindings.a,
            bindings.singular_values,
            bindings.left_vectors,
            bindings.right_vectors,
            bindings.workspace,
            bindings.dev_info,
            params,
        )
    }

    pub fn workspace_size_complex_f32(
        self,
        ctx: &Context,
        input: GesvdjInput<'_, Complex32, f32>,
        params: &GesvdjInfo,
    ) -> Result<usize> {
        cgesvdj_buffer_size(
            ctx,
            self.mode,
            self.economy,
            self.rows,
            self.columns,
            input.a,
            input.singular_values,
            input.left_vectors,
            input.right_vectors,
            params,
        )
    }

    pub fn execute_complex_f32(
        self,
        ctx: &Context,
        bindings: GesvdjBindings<'_, Complex32, f32>,
        params: &GesvdjInfo,
    ) -> Result<()> {
        cgesvdj(
            ctx,
            self.mode,
            self.economy,
            self.rows,
            self.columns,
            bindings.a,
            bindings.singular_values,
            bindings.left_vectors,
            bindings.right_vectors,
            bindings.workspace,
            bindings.dev_info,
            params,
        )
    }

    pub fn workspace_size_complex_f64(
        self,
        ctx: &Context,
        input: GesvdjInput<'_, Complex64, f64>,
        params: &GesvdjInfo,
    ) -> Result<usize> {
        zgesvdj_buffer_size(
            ctx,
            self.mode,
            self.economy,
            self.rows,
            self.columns,
            input.a,
            input.singular_values,
            input.left_vectors,
            input.right_vectors,
            params,
        )
    }

    pub fn execute_complex_f64(
        self,
        ctx: &Context,
        bindings: GesvdjBindings<'_, Complex64, f64>,
        params: &GesvdjInfo,
    ) -> Result<()> {
        zgesvdj(
            ctx,
            self.mode,
            self.economy,
            self.rows,
            self.columns,
            bindings.a,
            bindings.singular_values,
            bindings.left_vectors,
            bindings.right_vectors,
            bindings.workspace,
            bindings.dev_info,
            params,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GesvdjInput<'a, TA, TS> {
    pub a: MatrixRef<'a, TA>,
    pub singular_values: &'a DeviceMemory<TS>,
    pub left_vectors: Option<MatrixRef<'a, TA>>,
    pub right_vectors: Option<MatrixRef<'a, TA>>,
}

#[derive(Debug)]
pub struct GesvdjBindings<'a, TA, TS> {
    pub a: MatrixMut<'a, TA>,
    pub singular_values: &'a mut DeviceMemory<TS>,
    pub left_vectors: Option<MatrixMut<'a, TA>>,
    pub right_vectors: Option<MatrixMut<'a, TA>>,
    pub workspace: &'a mut DeviceMemory<TA>,
    pub dev_info: &'a mut DeviceMemory<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GesvdjBatched {
    pub mode: EigenMode,
    pub rows: usize,
    pub columns: usize,
    pub batch_size: usize,
}

impl GesvdjBatched {
    pub fn new(mode: EigenMode, rows: usize, columns: usize, batch_size: usize) -> Self {
        Self {
            mode,
            rows,
            columns,
            batch_size,
        }
    }

    pub fn workspace_size_f32(
        self,
        ctx: &Context,
        input: GesvdjInput<'_, f32, f32>,
        params: &GesvdjInfo,
    ) -> Result<usize> {
        sgesvdj_batched_buffer_size(
            ctx,
            self.mode,
            self.rows,
            self.columns,
            input.a,
            input.singular_values,
            input.left_vectors,
            input.right_vectors,
            params,
            self.batch_size,
        )
    }

    pub fn execute_f32(
        self,
        ctx: &Context,
        bindings: GesvdjBindings<'_, f32, f32>,
        params: &GesvdjInfo,
    ) -> Result<()> {
        sgesvdj_batched(
            ctx,
            self.mode,
            self.rows,
            self.columns,
            bindings.a,
            bindings.singular_values,
            bindings.left_vectors,
            bindings.right_vectors,
            bindings.workspace,
            bindings.dev_info,
            params,
            self.batch_size,
        )
    }

    pub fn workspace_size_f64(
        self,
        ctx: &Context,
        input: GesvdjInput<'_, f64, f64>,
        params: &GesvdjInfo,
    ) -> Result<usize> {
        dgesvdj_batched_buffer_size(
            ctx,
            self.mode,
            self.rows,
            self.columns,
            input.a,
            input.singular_values,
            input.left_vectors,
            input.right_vectors,
            params,
            self.batch_size,
        )
    }

    pub fn execute_f64(
        self,
        ctx: &Context,
        bindings: GesvdjBindings<'_, f64, f64>,
        params: &GesvdjInfo,
    ) -> Result<()> {
        dgesvdj_batched(
            ctx,
            self.mode,
            self.rows,
            self.columns,
            bindings.a,
            bindings.singular_values,
            bindings.left_vectors,
            bindings.right_vectors,
            bindings.workspace,
            bindings.dev_info,
            params,
            self.batch_size,
        )
    }

    pub fn workspace_size_complex_f32(
        self,
        ctx: &Context,
        input: GesvdjInput<'_, Complex32, f32>,
        params: &GesvdjInfo,
    ) -> Result<usize> {
        cgesvdj_batched_buffer_size(
            ctx,
            self.mode,
            self.rows,
            self.columns,
            input.a,
            input.singular_values,
            input.left_vectors,
            input.right_vectors,
            params,
            self.batch_size,
        )
    }

    pub fn execute_complex_f32(
        self,
        ctx: &Context,
        bindings: GesvdjBindings<'_, Complex32, f32>,
        params: &GesvdjInfo,
    ) -> Result<()> {
        cgesvdj_batched(
            ctx,
            self.mode,
            self.rows,
            self.columns,
            bindings.a,
            bindings.singular_values,
            bindings.left_vectors,
            bindings.right_vectors,
            bindings.workspace,
            bindings.dev_info,
            params,
            self.batch_size,
        )
    }

    pub fn workspace_size_complex_f64(
        self,
        ctx: &Context,
        input: GesvdjInput<'_, Complex64, f64>,
        params: &GesvdjInfo,
    ) -> Result<usize> {
        zgesvdj_batched_buffer_size(
            ctx,
            self.mode,
            self.rows,
            self.columns,
            input.a,
            input.singular_values,
            input.left_vectors,
            input.right_vectors,
            params,
            self.batch_size,
        )
    }

    pub fn execute_complex_f64(
        self,
        ctx: &Context,
        bindings: GesvdjBindings<'_, Complex64, f64>,
        params: &GesvdjInfo,
    ) -> Result<()> {
        zgesvdj_batched(
            ctx,
            self.mode,
            self.rows,
            self.columns,
            bindings.a,
            bindings.singular_values,
            bindings.left_vectors,
            bindings.right_vectors,
            bindings.workspace,
            bindings.dev_info,
            params,
            self.batch_size,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GesvdaStridedBatched {
    pub mode: EigenMode,
    pub rank: usize,
    pub rows: usize,
    pub columns: usize,
    pub batch_size: usize,
}

impl GesvdaStridedBatched {
    pub fn new(
        mode: EigenMode,
        rank: usize,
        rows: usize,
        columns: usize,
        batch_size: usize,
    ) -> Self {
        Self {
            mode,
            rank,
            rows,
            columns,
            batch_size,
        }
    }

    pub fn workspace_size_f32(
        self,
        ctx: &Context,
        input: GesvdaStridedBatchedInput<'_, f32, f32>,
    ) -> Result<usize> {
        sgesvda_strided_batched_buffer_size(
            ctx,
            self.mode,
            self.rank,
            self.rows,
            self.columns,
            input.a,
            input.singular_values,
            input.left_vectors,
            input.right_vectors,
            self.batch_size,
        )
    }

    pub fn execute_f32(
        self,
        ctx: &Context,
        bindings: GesvdaStridedBatchedBindings<'_, f32, f32>,
    ) -> Result<()> {
        sgesvda_strided_batched(
            ctx,
            self.mode,
            self.rank,
            self.rows,
            self.columns,
            bindings.a,
            bindings.singular_values,
            bindings.left_vectors,
            bindings.right_vectors,
            bindings.workspace,
            bindings.dev_info,
            bindings.residual,
            self.batch_size,
        )
    }

    pub fn workspace_size_f64(
        self,
        ctx: &Context,
        input: GesvdaStridedBatchedInput<'_, f64, f64>,
    ) -> Result<usize> {
        dgesvda_strided_batched_buffer_size(
            ctx,
            self.mode,
            self.rank,
            self.rows,
            self.columns,
            input.a,
            input.singular_values,
            input.left_vectors,
            input.right_vectors,
            self.batch_size,
        )
    }

    pub fn execute_f64(
        self,
        ctx: &Context,
        bindings: GesvdaStridedBatchedBindings<'_, f64, f64>,
    ) -> Result<()> {
        dgesvda_strided_batched(
            ctx,
            self.mode,
            self.rank,
            self.rows,
            self.columns,
            bindings.a,
            bindings.singular_values,
            bindings.left_vectors,
            bindings.right_vectors,
            bindings.workspace,
            bindings.dev_info,
            bindings.residual,
            self.batch_size,
        )
    }

    pub fn workspace_size_complex_f32(
        self,
        ctx: &Context,
        input: GesvdaStridedBatchedInput<'_, Complex32, f32>,
    ) -> Result<usize> {
        cgesvda_strided_batched_buffer_size(
            ctx,
            self.mode,
            self.rank,
            self.rows,
            self.columns,
            input.a,
            input.singular_values,
            input.left_vectors,
            input.right_vectors,
            self.batch_size,
        )
    }

    pub fn execute_complex_f32(
        self,
        ctx: &Context,
        bindings: GesvdaStridedBatchedBindings<'_, Complex32, f32>,
    ) -> Result<()> {
        cgesvda_strided_batched(
            ctx,
            self.mode,
            self.rank,
            self.rows,
            self.columns,
            bindings.a,
            bindings.singular_values,
            bindings.left_vectors,
            bindings.right_vectors,
            bindings.workspace,
            bindings.dev_info,
            bindings.residual,
            self.batch_size,
        )
    }

    pub fn workspace_size_complex_f64(
        self,
        ctx: &Context,
        input: GesvdaStridedBatchedInput<'_, Complex64, f64>,
    ) -> Result<usize> {
        zgesvda_strided_batched_buffer_size(
            ctx,
            self.mode,
            self.rank,
            self.rows,
            self.columns,
            input.a,
            input.singular_values,
            input.left_vectors,
            input.right_vectors,
            self.batch_size,
        )
    }

    pub fn execute_complex_f64(
        self,
        ctx: &Context,
        bindings: GesvdaStridedBatchedBindings<'_, Complex64, f64>,
    ) -> Result<()> {
        zgesvda_strided_batched(
            ctx,
            self.mode,
            self.rank,
            self.rows,
            self.columns,
            bindings.a,
            bindings.singular_values,
            bindings.left_vectors,
            bindings.right_vectors,
            bindings.workspace,
            bindings.dev_info,
            bindings.residual,
            self.batch_size,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GesvdaStridedBatchedInput<'a, TA, TS> {
    pub a: StridedBatchedMatrixRef<'a, TA>,
    pub singular_values: StridedBatchedVectorRef<'a, TS>,
    pub left_vectors: Option<StridedBatchedMatrixRef<'a, TA>>,
    pub right_vectors: Option<StridedBatchedMatrixRef<'a, TA>>,
}

#[derive(Debug)]
pub struct GesvdaStridedBatchedBindings<'a, TA, TS> {
    pub a: StridedBatchedMatrixRef<'a, TA>,
    pub singular_values: StridedBatchedVectorMut<'a, TS>,
    pub left_vectors: Option<StridedBatchedMatrixMut<'a, TA>>,
    pub right_vectors: Option<StridedBatchedMatrixMut<'a, TA>>,
    pub workspace: &'a mut DeviceMemory<TA>,
    pub dev_info: &'a mut DeviceMemory<i32>,
    pub residual: Option<&'a mut f64>,
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use singe_core::assert_close;
    use singe_cuda::memory::DeviceMemory;

    use super::*;
    use crate::{params::Params, testing::setup_context_if_available};

    #[test]
    fn gesvd_descriptor_returns_expected_singular_values() -> Result<()> {
        let Some(ctx) = setup_context_if_available()? else {
            return Ok(());
        };
        let params = Params::create()?;

        let mut a = DeviceMemory::from_slice(&[
            3.0_f32, 0.0, //
            0.0, 2.0,
        ])?;
        let mut s = DeviceMemory::<f32>::create(2)?;
        let operation = Gesvd::new(SvdMode::None, SvdMode::None, 2, 2);
        let workspace_sizes = operation.workspace_size(
            &ctx,
            &params,
            GesvdInput {
                a: MatrixRef::new(&a, 2),
                singular_values: &s,
                left_vectors: None::<MatrixRef<'_, f32>>,
                right_vectors_transposed: None::<MatrixRef<'_, f32>>,
            },
        )?;
        let mut workspace = DeviceMemory::<u8>::create(workspace_sizes.device_bytes)?;
        let mut host_workspace = vec![0_u8; workspace_sizes.host_bytes];
        let mut dev_info = DeviceMemory::create(1)?;

        operation.execute(
            &ctx,
            &params,
            GesvdBindings {
                a: MatrixMut::new(&mut a, 2),
                singular_values: &mut s,
                left_vectors: None::<MatrixMut<'_, f32>>,
                right_vectors_transposed: None::<MatrixMut<'_, f32>>,
                workspace: ByteWorkspaceMut::new(&mut workspace, &mut host_workspace),
                dev_info: &mut dev_info,
            },
        )?;

        let singular_values = s.copy_to_host_vec()?;
        let info = dev_info.copy_to_host_vec()?;

        assert_eq!(info, vec![0]);
        assert_close!(&singular_values, &[3.0, 2.0], 1.0e-5);
        Ok(())
    }

    #[test]
    fn gesvdp_descriptor_preserves_operation_shape() {
        let operation = Gesvdp::new(EigenMode::Vector, true, 32, 16);

        assert_eq!(operation.mode, EigenMode::Vector);
        assert!(operation.economy);
        assert_eq!(operation.rows, 32);
        assert_eq!(operation.columns, 16);
    }

    #[test]
    fn gesvdr_descriptor_preserves_operation_shape() {
        let operation = Gesvdr::new(
            TruncatedSvdMode::Some,
            TruncatedSvdMode::None,
            128,
            64,
            12,
            24,
            3,
        );

        assert_eq!(operation.left_vectors, TruncatedSvdMode::Some);
        assert_eq!(operation.right_vectors, TruncatedSvdMode::None);
        assert_eq!(operation.rows, 128);
        assert_eq!(operation.columns, 64);
        assert_eq!(operation.rank, 12);
        assert_eq!(operation.oversamples, 24);
        assert_eq!(operation.iterations, 3);
    }

    #[test]
    fn test_xgesvd_returns_expected_singular_values() -> Result<()> {
        let Some(ctx) = setup_context_if_available()? else {
            return Ok(());
        };
        let params = Params::create()?;

        let mut a = DeviceMemory::from_slice(&[
            3.0_f32, 0.0, //
            0.0, 2.0,
        ])?;
        let mut s = DeviceMemory::create(2)?;
        let workspace_sizes = xgesvd_buffer_size::<f32, f32, f32, f32>(
            &ctx,
            &params,
            SvdMode::None,
            SvdMode::None,
            2,
            2,
            MatrixRef::new(&a, 2),
            &s,
            None,
            None,
        )?;
        let mut device_workspace = DeviceMemory::create(workspace_sizes.device_bytes.max(1))?;
        let mut host_workspace = vec![0_u8; workspace_sizes.host_bytes.max(1)];
        let mut dev_info = DeviceMemory::create(1)?;

        xgesvd::<f32, f32, f32, f32>(
            &ctx,
            &params,
            SvdMode::None,
            SvdMode::None,
            2,
            2,
            MatrixMut::new(&mut a, 2),
            &mut s,
            None,
            None,
            ByteWorkspaceMut::new(&mut device_workspace, &mut host_workspace),
            &mut dev_info,
        )?;

        let singular_values = s.copy_to_host_vec()?;
        let info = dev_info.copy_to_host_vec()?;

        assert_eq!(info, vec![0]);
        assert_close!(&singular_values, &[3.0, 2.0], 1.0e-5);
        Ok(())
    }
}
