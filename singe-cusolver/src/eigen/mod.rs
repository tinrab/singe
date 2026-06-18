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
    layout::{ByteWorkspaceMut, MatrixMut, MatrixRef, SelectionWorkspaceSizes, WorkspaceSizes},
    params::Params,
    types::{EigenMode, EigenType, FillMode},
};

use raw::*;

pub use info::SyevjInfo;

#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum EigenSelection<T> {
    All,
    ByValue { lower: T, upper: T },
    ByIndex { start: usize, end: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Syevd {
    pub mode: EigenMode,
    pub fill_mode: FillMode,
    pub n: usize,
    pub leading_dimension: usize,
}

impl Syevd {
    pub fn new(mode: EigenMode, fill_mode: FillMode, n: usize, leading_dimension: usize) -> Self {
        Self {
            mode,
            fill_mode,
            n,
            leading_dimension,
        }
    }

    pub fn workspace_size<TA: DataTypeLike, TW: DataTypeLike>(
        self,
        ctx: &Context,
        params: &Params,
        input: SyevdInput<'_, TA, TW>,
    ) -> Result<WorkspaceSizes> {
        xsyevd_buffer_size(
            ctx,
            params,
            self.mode,
            self.fill_mode,
            self.n,
            input.a,
            self.leading_dimension,
            input.eigenvalues,
        )
    }

    pub fn execute<TA: DataTypeLike, TW: DataTypeLike>(
        self,
        ctx: &Context,
        params: &Params,
        bindings: SyevdBindings<'_, TA, TW>,
    ) -> Result<()> {
        xsyevd(
            ctx,
            params,
            self.mode,
            self.fill_mode,
            self.n,
            bindings.a,
            self.leading_dimension,
            bindings.eigenvalues,
            bindings.workspace,
            bindings.dev_info,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SyevdInput<'a, TA, TW> {
    pub a: &'a DeviceMemory<TA>,
    pub eigenvalues: &'a DeviceMemory<TW>,
}

#[derive(Debug)]
pub struct SyevdBindings<'a, TA, TW> {
    pub a: &'a mut DeviceMemory<TA>,
    pub eigenvalues: &'a mut DeviceMemory<TW>,
    pub workspace: ByteWorkspaceMut<'a>,
    pub dev_info: &'a mut DeviceMemory<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geev {
    pub n: usize,
    pub leading_dimension: usize,
}

impl Geev {
    pub fn new(n: usize, leading_dimension: usize) -> Self {
        Self {
            n,
            leading_dimension,
        }
    }

    pub fn workspace_size<TA: DataTypeLike, TW: DataTypeLike, TV: DataTypeLike>(
        self,
        ctx: &Context,
        params: &Params,
        input: GeevInput<'_, TA, TW, TV>,
    ) -> Result<WorkspaceSizes> {
        xgeev_buffer_size(
            ctx,
            params,
            self.n,
            MatrixRef::new(input.a, self.leading_dimension),
            input.eigenvalues,
            input.right_vectors,
        )
    }

    pub fn execute<TA: DataTypeLike, TW: DataTypeLike, TV: DataTypeLike>(
        self,
        ctx: &Context,
        params: &Params,
        bindings: GeevBindings<'_, TA, TW, TV>,
    ) -> Result<()> {
        xgeev(
            ctx,
            params,
            self.n,
            MatrixMut::new(bindings.a, self.leading_dimension),
            bindings.eigenvalues,
            bindings.right_vectors,
            bindings.workspace,
            bindings.dev_info,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GeevInput<'a, TA, TW, TV> {
    pub a: &'a DeviceMemory<TA>,
    pub eigenvalues: &'a DeviceMemory<TW>,
    pub right_vectors: Option<MatrixRef<'a, TV>>,
}

#[derive(Debug)]
pub struct GeevBindings<'a, TA, TW, TV> {
    pub a: &'a mut DeviceMemory<TA>,
    pub eigenvalues: &'a mut DeviceMemory<TW>,
    pub right_vectors: Option<MatrixMut<'a, TV>>,
    pub workspace: ByteWorkspaceMut<'a>,
    pub dev_info: &'a mut DeviceMemory<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyevBatched {
    pub mode: EigenMode,
    pub fill_mode: FillMode,
    pub n: usize,
    pub leading_dimension: usize,
    pub batch_count: usize,
}

impl SyevBatched {
    pub fn new(
        mode: EigenMode,
        fill_mode: FillMode,
        n: usize,
        leading_dimension: usize,
        batch_count: usize,
    ) -> Self {
        Self {
            mode,
            fill_mode,
            n,
            leading_dimension,
            batch_count,
        }
    }

    pub fn workspace_size<TA: DataTypeLike, TW: DataTypeLike>(
        self,
        ctx: &Context,
        params: &Params,
        input: SyevdInput<'_, TA, TW>,
    ) -> Result<WorkspaceSizes> {
        xsyev_batched_buffer_size(
            ctx,
            params,
            self.mode,
            self.fill_mode,
            self.n,
            MatrixRef::new(input.a, self.leading_dimension),
            input.eigenvalues,
            self.batch_count,
        )
    }

    pub fn execute<TA: DataTypeLike, TW: DataTypeLike>(
        self,
        ctx: &Context,
        params: &Params,
        bindings: SyevdBindings<'_, TA, TW>,
    ) -> Result<()> {
        xsyev_batched(
            ctx,
            params,
            self.mode,
            self.fill_mode,
            self.n,
            MatrixMut::new(bindings.a, self.leading_dimension),
            bindings.eigenvalues,
            self.batch_count,
            bindings.workspace,
            bindings.dev_info,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Syevdx<T> {
    pub mode: EigenMode,
    pub fill_mode: FillMode,
    pub selection: EigenSelection<T>,
    pub n: usize,
    pub leading_dimension: usize,
}

impl<T> Syevdx<T> {
    pub fn new(
        mode: EigenMode,
        fill_mode: FillMode,
        selection: EigenSelection<T>,
        n: usize,
        leading_dimension: usize,
    ) -> Self {
        Self {
            mode,
            fill_mode,
            selection,
            n,
            leading_dimension,
        }
    }
}

impl<T: Copy + Default> Syevdx<T> {
    pub fn workspace_size<TA: DataTypeLike, TW: DataTypeLike>(
        self,
        ctx: &Context,
        params: &Params,
        input: SyevdInput<'_, TA, TW>,
    ) -> Result<SelectionWorkspaceSizes> {
        xsyevdx_buffer_size(
            ctx,
            params,
            self.mode,
            self.fill_mode,
            self.selection,
            self.n,
            input.a,
            self.leading_dimension,
            input.eigenvalues,
        )
    }

    pub fn execute<TA: DataTypeLike, TW: DataTypeLike>(
        self,
        ctx: &Context,
        params: &Params,
        bindings: SyevdBindings<'_, TA, TW>,
    ) -> Result<usize> {
        xsyevdx(
            ctx,
            params,
            self.mode,
            self.fill_mode,
            self.selection,
            self.n,
            bindings.a,
            self.leading_dimension,
            bindings.eigenvalues,
            bindings.workspace,
            bindings.dev_info,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Syevj {
    pub mode: EigenMode,
    pub fill_mode: FillMode,
    pub n: usize,
    pub leading_dimension: usize,
}

impl Syevj {
    pub fn new(mode: EigenMode, fill_mode: FillMode, n: usize, leading_dimension: usize) -> Self {
        Self {
            mode,
            fill_mode,
            n,
            leading_dimension,
        }
    }

    pub fn workspace_size_f32(
        self,
        ctx: &Context,
        input: SyevjInput<'_, f32, f32>,
        params: &SyevjInfo,
    ) -> Result<usize> {
        ssyevj_buffer_size(
            ctx,
            self.mode,
            self.fill_mode,
            self.n,
            input.a,
            self.leading_dimension,
            input.eigenvalues,
            params,
        )
    }

    pub fn execute_f32(
        self,
        ctx: &Context,
        bindings: SyevjBindings<'_, f32, f32>,
        params: &SyevjInfo,
    ) -> Result<()> {
        ssyevj(
            ctx,
            self.mode,
            self.fill_mode,
            self.n,
            bindings.a,
            self.leading_dimension,
            bindings.eigenvalues,
            bindings.workspace,
            bindings.dev_info,
            params,
        )
    }

    pub fn workspace_size_f64(
        self,
        ctx: &Context,
        input: SyevjInput<'_, f64, f64>,
        params: &SyevjInfo,
    ) -> Result<usize> {
        dsyevj_buffer_size(
            ctx,
            self.mode,
            self.fill_mode,
            self.n,
            input.a,
            self.leading_dimension,
            input.eigenvalues,
            params,
        )
    }

    pub fn execute_f64(
        self,
        ctx: &Context,
        bindings: SyevjBindings<'_, f64, f64>,
        params: &SyevjInfo,
    ) -> Result<()> {
        dsyevj(
            ctx,
            self.mode,
            self.fill_mode,
            self.n,
            bindings.a,
            self.leading_dimension,
            bindings.eigenvalues,
            bindings.workspace,
            bindings.dev_info,
            params,
        )
    }

    pub fn workspace_size_complex_f32(
        self,
        ctx: &Context,
        input: SyevjInput<'_, Complex32, f32>,
        params: &SyevjInfo,
    ) -> Result<usize> {
        cheevj_buffer_size(
            ctx,
            self.mode,
            self.fill_mode,
            self.n,
            input.a,
            self.leading_dimension,
            input.eigenvalues,
            params,
        )
    }

    pub fn execute_complex_f32(
        self,
        ctx: &Context,
        bindings: SyevjBindings<'_, Complex32, f32>,
        params: &SyevjInfo,
    ) -> Result<()> {
        cheevj(
            ctx,
            self.mode,
            self.fill_mode,
            self.n,
            bindings.a,
            self.leading_dimension,
            bindings.eigenvalues,
            bindings.workspace,
            bindings.dev_info,
            params,
        )
    }

    pub fn workspace_size_complex_f64(
        self,
        ctx: &Context,
        input: SyevjInput<'_, Complex64, f64>,
        params: &SyevjInfo,
    ) -> Result<usize> {
        zheevj_buffer_size(
            ctx,
            self.mode,
            self.fill_mode,
            self.n,
            input.a,
            self.leading_dimension,
            input.eigenvalues,
            params,
        )
    }

    pub fn execute_complex_f64(
        self,
        ctx: &Context,
        bindings: SyevjBindings<'_, Complex64, f64>,
        params: &SyevjInfo,
    ) -> Result<()> {
        zheevj(
            ctx,
            self.mode,
            self.fill_mode,
            self.n,
            bindings.a,
            self.leading_dimension,
            bindings.eigenvalues,
            bindings.workspace,
            bindings.dev_info,
            params,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyevjBatched {
    pub mode: EigenMode,
    pub fill_mode: FillMode,
    pub n: usize,
    pub leading_dimension: usize,
    pub batch_count: usize,
}

impl SyevjBatched {
    pub fn new(
        mode: EigenMode,
        fill_mode: FillMode,
        n: usize,
        leading_dimension: usize,
        batch_count: usize,
    ) -> Self {
        Self {
            mode,
            fill_mode,
            n,
            leading_dimension,
            batch_count,
        }
    }

    pub fn workspace_size_f32(
        self,
        ctx: &Context,
        input: SyevjInput<'_, f32, f32>,
        params: &SyevjInfo,
    ) -> Result<usize> {
        ssyevj_batched_buffer_size(
            ctx,
            self.mode,
            self.fill_mode,
            self.n,
            input.a,
            self.leading_dimension,
            input.eigenvalues,
            params,
            self.batch_count,
        )
    }

    pub fn execute_f32(
        self,
        ctx: &Context,
        bindings: SyevjBindings<'_, f32, f32>,
        params: &SyevjInfo,
    ) -> Result<()> {
        ssyevj_batched(
            ctx,
            self.mode,
            self.fill_mode,
            self.n,
            bindings.a,
            self.leading_dimension,
            bindings.eigenvalues,
            bindings.workspace,
            bindings.dev_info,
            params,
            self.batch_count,
        )
    }

    pub fn workspace_size_f64(
        self,
        ctx: &Context,
        input: SyevjInput<'_, f64, f64>,
        params: &SyevjInfo,
    ) -> Result<usize> {
        dsyevj_batched_buffer_size(
            ctx,
            self.mode,
            self.fill_mode,
            self.n,
            input.a,
            self.leading_dimension,
            input.eigenvalues,
            params,
            self.batch_count,
        )
    }

    pub fn execute_f64(
        self,
        ctx: &Context,
        bindings: SyevjBindings<'_, f64, f64>,
        params: &SyevjInfo,
    ) -> Result<()> {
        dsyevj_batched(
            ctx,
            self.mode,
            self.fill_mode,
            self.n,
            bindings.a,
            self.leading_dimension,
            bindings.eigenvalues,
            bindings.workspace,
            bindings.dev_info,
            params,
            self.batch_count,
        )
    }

    pub fn workspace_size_complex_f32(
        self,
        ctx: &Context,
        input: SyevjInput<'_, Complex32, f32>,
        params: &SyevjInfo,
    ) -> Result<usize> {
        cheevj_batched_buffer_size(
            ctx,
            self.mode,
            self.fill_mode,
            self.n,
            input.a,
            self.leading_dimension,
            input.eigenvalues,
            params,
            self.batch_count,
        )
    }

    pub fn execute_complex_f32(
        self,
        ctx: &Context,
        bindings: SyevjBindings<'_, Complex32, f32>,
        params: &SyevjInfo,
    ) -> Result<()> {
        cheevj_batched(
            ctx,
            self.mode,
            self.fill_mode,
            self.n,
            bindings.a,
            self.leading_dimension,
            bindings.eigenvalues,
            bindings.workspace,
            bindings.dev_info,
            params,
            self.batch_count,
        )
    }

    pub fn workspace_size_complex_f64(
        self,
        ctx: &Context,
        input: SyevjInput<'_, Complex64, f64>,
        params: &SyevjInfo,
    ) -> Result<usize> {
        zheevj_batched_buffer_size(
            ctx,
            self.mode,
            self.fill_mode,
            self.n,
            input.a,
            self.leading_dimension,
            input.eigenvalues,
            params,
            self.batch_count,
        )
    }

    pub fn execute_complex_f64(
        self,
        ctx: &Context,
        bindings: SyevjBindings<'_, Complex64, f64>,
        params: &SyevjInfo,
    ) -> Result<()> {
        zheevj_batched(
            ctx,
            self.mode,
            self.fill_mode,
            self.n,
            bindings.a,
            self.leading_dimension,
            bindings.eigenvalues,
            bindings.workspace,
            bindings.dev_info,
            params,
            self.batch_count,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SyevjInput<'a, TA, TW> {
    pub a: &'a DeviceMemory<TA>,
    pub eigenvalues: &'a DeviceMemory<TW>,
}

#[derive(Debug)]
pub struct SyevjBindings<'a, TA, TW> {
    pub a: &'a mut DeviceMemory<TA>,
    pub eigenvalues: &'a mut DeviceMemory<TW>,
    pub workspace: &'a mut DeviceMemory<TA>,
    pub dev_info: &'a mut DeviceMemory<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sygvj {
    pub eig_type: EigenType,
    pub mode: EigenMode,
    pub fill_mode: FillMode,
    pub n: usize,
    pub leading_dimension_a: usize,
    pub leading_dimension_b: usize,
}

impl Sygvj {
    pub fn new(
        eig_type: EigenType,
        mode: EigenMode,
        fill_mode: FillMode,
        n: usize,
        leading_dimension_a: usize,
        leading_dimension_b: usize,
    ) -> Self {
        Self {
            eig_type,
            mode,
            fill_mode,
            n,
            leading_dimension_a,
            leading_dimension_b,
        }
    }

    pub fn workspace_size_f32(
        self,
        ctx: &Context,
        input: SygvInput<'_, f32, f32>,
        params: &SyevjInfo,
    ) -> Result<usize> {
        ssygvj_buffer_size(
            ctx,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.n,
            input.a,
            self.leading_dimension_a,
            input.b,
            self.leading_dimension_b,
            input.eigenvalues,
            params,
        )
    }

    pub fn execute_f32(
        self,
        ctx: &Context,
        bindings: SygvBindings<'_, f32, f32>,
        params: &SyevjInfo,
    ) -> Result<()> {
        ssygvj(
            ctx,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.n,
            bindings.a,
            self.leading_dimension_a,
            bindings.b,
            self.leading_dimension_b,
            bindings.eigenvalues,
            bindings.workspace,
            bindings.dev_info,
            params,
        )
    }

    pub fn workspace_size_f64(
        self,
        ctx: &Context,
        input: SygvInput<'_, f64, f64>,
        params: &SyevjInfo,
    ) -> Result<usize> {
        dsygvj_buffer_size(
            ctx,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.n,
            input.a,
            self.leading_dimension_a,
            input.b,
            self.leading_dimension_b,
            input.eigenvalues,
            params,
        )
    }

    pub fn execute_f64(
        self,
        ctx: &Context,
        bindings: SygvBindings<'_, f64, f64>,
        params: &SyevjInfo,
    ) -> Result<()> {
        dsygvj(
            ctx,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.n,
            bindings.a,
            self.leading_dimension_a,
            bindings.b,
            self.leading_dimension_b,
            bindings.eigenvalues,
            bindings.workspace,
            bindings.dev_info,
            params,
        )
    }

    pub fn workspace_size_complex_f32(
        self,
        ctx: &Context,
        input: SygvInput<'_, Complex32, f32>,
        params: &SyevjInfo,
    ) -> Result<usize> {
        chegvj_buffer_size(
            ctx,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.n,
            input.a,
            self.leading_dimension_a,
            input.b,
            self.leading_dimension_b,
            input.eigenvalues,
            params,
        )
    }

    pub fn execute_complex_f32(
        self,
        ctx: &Context,
        bindings: SygvBindings<'_, Complex32, f32>,
        params: &SyevjInfo,
    ) -> Result<()> {
        chegvj(
            ctx,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.n,
            bindings.a,
            self.leading_dimension_a,
            bindings.b,
            self.leading_dimension_b,
            bindings.eigenvalues,
            bindings.workspace,
            bindings.dev_info,
            params,
        )
    }

    pub fn workspace_size_complex_f64(
        self,
        ctx: &Context,
        input: SygvInput<'_, Complex64, f64>,
        params: &SyevjInfo,
    ) -> Result<usize> {
        zhegvj_buffer_size(
            ctx,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.n,
            input.a,
            self.leading_dimension_a,
            input.b,
            self.leading_dimension_b,
            input.eigenvalues,
            params,
        )
    }

    pub fn execute_complex_f64(
        self,
        ctx: &Context,
        bindings: SygvBindings<'_, Complex64, f64>,
        params: &SyevjInfo,
    ) -> Result<()> {
        zhegvj(
            ctx,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.n,
            bindings.a,
            self.leading_dimension_a,
            bindings.b,
            self.leading_dimension_b,
            bindings.eigenvalues,
            bindings.workspace,
            bindings.dev_info,
            params,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sygvd {
    pub eig_type: EigenType,
    pub mode: EigenMode,
    pub fill_mode: FillMode,
    pub n: usize,
    pub leading_dimension_a: usize,
    pub leading_dimension_b: usize,
}

impl Sygvd {
    pub fn new(
        eig_type: EigenType,
        mode: EigenMode,
        fill_mode: FillMode,
        n: usize,
        leading_dimension_a: usize,
        leading_dimension_b: usize,
    ) -> Self {
        Self {
            eig_type,
            mode,
            fill_mode,
            n,
            leading_dimension_a,
            leading_dimension_b,
        }
    }

    pub fn workspace_size_f32(
        self,
        ctx: &Context,
        input: SygvInput<'_, f32, f32>,
    ) -> Result<usize> {
        ssygvd_buffer_size(
            ctx,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.n,
            input.a,
            self.leading_dimension_a,
            input.b,
            self.leading_dimension_b,
            input.eigenvalues,
        )
    }

    pub fn execute_f32(self, ctx: &Context, bindings: SygvBindings<'_, f32, f32>) -> Result<()> {
        ssygvd(
            ctx,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.n,
            bindings.a,
            self.leading_dimension_a,
            bindings.b,
            self.leading_dimension_b,
            bindings.eigenvalues,
            bindings.workspace,
            bindings.dev_info,
        )
    }

    pub fn workspace_size_f64(
        self,
        ctx: &Context,
        input: SygvInput<'_, f64, f64>,
    ) -> Result<usize> {
        dsygvd_buffer_size(
            ctx,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.n,
            input.a,
            self.leading_dimension_a,
            input.b,
            self.leading_dimension_b,
            input.eigenvalues,
        )
    }

    pub fn execute_f64(self, ctx: &Context, bindings: SygvBindings<'_, f64, f64>) -> Result<()> {
        dsygvd(
            ctx,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.n,
            bindings.a,
            self.leading_dimension_a,
            bindings.b,
            self.leading_dimension_b,
            bindings.eigenvalues,
            bindings.workspace,
            bindings.dev_info,
        )
    }

    pub fn workspace_size_complex_f32(
        self,
        ctx: &Context,
        input: SygvInput<'_, Complex32, f32>,
    ) -> Result<usize> {
        chegvd_buffer_size(
            ctx,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.n,
            input.a,
            self.leading_dimension_a,
            input.b,
            self.leading_dimension_b,
            input.eigenvalues,
        )
    }

    pub fn execute_complex_f32(
        self,
        ctx: &Context,
        bindings: SygvBindings<'_, Complex32, f32>,
    ) -> Result<()> {
        chegvd(
            ctx,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.n,
            bindings.a,
            self.leading_dimension_a,
            bindings.b,
            self.leading_dimension_b,
            bindings.eigenvalues,
            bindings.workspace,
            bindings.dev_info,
        )
    }

    pub fn workspace_size_complex_f64(
        self,
        ctx: &Context,
        input: SygvInput<'_, Complex64, f64>,
    ) -> Result<usize> {
        zhegvd_buffer_size(
            ctx,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.n,
            input.a,
            self.leading_dimension_a,
            input.b,
            self.leading_dimension_b,
            input.eigenvalues,
        )
    }

    pub fn execute_complex_f64(
        self,
        ctx: &Context,
        bindings: SygvBindings<'_, Complex64, f64>,
    ) -> Result<()> {
        zhegvd(
            ctx,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.n,
            bindings.a,
            self.leading_dimension_a,
            bindings.b,
            self.leading_dimension_b,
            bindings.eigenvalues,
            bindings.workspace,
            bindings.dev_info,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sygvdx<T> {
    pub eig_type: EigenType,
    pub mode: EigenMode,
    pub fill_mode: FillMode,
    pub selection: EigenSelection<T>,
    pub n: usize,
    pub leading_dimension_a: usize,
    pub leading_dimension_b: usize,
}

impl<T> Sygvdx<T> {
    pub fn new(
        eig_type: EigenType,
        mode: EigenMode,
        fill_mode: FillMode,
        selection: EigenSelection<T>,
        n: usize,
        leading_dimension_a: usize,
        leading_dimension_b: usize,
    ) -> Self {
        Self {
            eig_type,
            mode,
            fill_mode,
            selection,
            n,
            leading_dimension_a,
            leading_dimension_b,
        }
    }
}

impl Sygvdx<f32> {
    pub fn workspace_size_f32(
        self,
        ctx: &Context,
        input: SygvInput<'_, f32, f32>,
    ) -> Result<(usize, usize)> {
        ssygvdx_selected_buffer_size(
            ctx,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.selection,
            self.n,
            input.a,
            self.leading_dimension_a,
            input.b,
            self.leading_dimension_b,
            input.eigenvalues,
        )
    }

    pub fn execute_f32(self, ctx: &Context, bindings: SygvBindings<'_, f32, f32>) -> Result<usize> {
        ssygvdx_selected(
            ctx,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.selection,
            self.n,
            bindings.a,
            self.leading_dimension_a,
            bindings.b,
            self.leading_dimension_b,
            bindings.eigenvalues,
            bindings.workspace,
            bindings.dev_info,
        )
    }

    pub fn workspace_size_complex_f32(
        self,
        ctx: &Context,
        input: SygvInput<'_, Complex32, f32>,
    ) -> Result<(usize, usize)> {
        chegvdx_selected_buffer_size(
            ctx,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.selection,
            self.n,
            input.a,
            self.leading_dimension_a,
            input.b,
            self.leading_dimension_b,
            input.eigenvalues,
        )
    }

    pub fn execute_complex_f32(
        self,
        ctx: &Context,
        bindings: SygvBindings<'_, Complex32, f32>,
    ) -> Result<usize> {
        chegvdx_selected(
            ctx,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.selection,
            self.n,
            bindings.a,
            self.leading_dimension_a,
            bindings.b,
            self.leading_dimension_b,
            bindings.eigenvalues,
            bindings.workspace,
            bindings.dev_info,
        )
    }
}

impl Sygvdx<f64> {
    pub fn workspace_size_f64(
        self,
        ctx: &Context,
        input: SygvInput<'_, f64, f64>,
    ) -> Result<(usize, usize)> {
        dsygvdx_selected_buffer_size(
            ctx,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.selection,
            self.n,
            input.a,
            self.leading_dimension_a,
            input.b,
            self.leading_dimension_b,
            input.eigenvalues,
        )
    }

    pub fn execute_f64(self, ctx: &Context, bindings: SygvBindings<'_, f64, f64>) -> Result<usize> {
        dsygvdx_selected(
            ctx,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.selection,
            self.n,
            bindings.a,
            self.leading_dimension_a,
            bindings.b,
            self.leading_dimension_b,
            bindings.eigenvalues,
            bindings.workspace,
            bindings.dev_info,
        )
    }

    pub fn workspace_size_complex_f64(
        self,
        ctx: &Context,
        input: SygvInput<'_, Complex64, f64>,
    ) -> Result<(usize, usize)> {
        zhegvdx_selected_buffer_size(
            ctx,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.selection,
            self.n,
            input.a,
            self.leading_dimension_a,
            input.b,
            self.leading_dimension_b,
            input.eigenvalues,
        )
    }

    pub fn execute_complex_f64(
        self,
        ctx: &Context,
        bindings: SygvBindings<'_, Complex64, f64>,
    ) -> Result<usize> {
        zhegvdx_selected(
            ctx,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.selection,
            self.n,
            bindings.a,
            self.leading_dimension_a,
            bindings.b,
            self.leading_dimension_b,
            bindings.eigenvalues,
            bindings.workspace,
            bindings.dev_info,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SygvInput<'a, TA, TW> {
    pub a: &'a DeviceMemory<TA>,
    pub b: &'a DeviceMemory<TA>,
    pub eigenvalues: &'a DeviceMemory<TW>,
}

#[derive(Debug)]
pub struct SygvBindings<'a, TA, TW> {
    pub a: &'a mut DeviceMemory<TA>,
    pub b: &'a mut DeviceMemory<TA>,
    pub eigenvalues: &'a mut DeviceMemory<TW>,
    pub workspace: &'a mut DeviceMemory<TA>,
    pub dev_info: &'a mut DeviceMemory<i32>,
}

#[cfg(test)]
mod tests {
    use super::{EigenSelection, Geev, SyevBatched};
    use crate::{
        eigen::validation::selection_parts,
        types::{EigenMode, EigenRange, FillMode},
    };

    #[test]
    fn eigen_selection_all_maps_cleanly() {
        let (range, values, indices) = selection_parts::<f32>(EigenSelection::All);
        assert_eq!(range, EigenRange::All);
        assert_eq!(values, None);
        assert_eq!(indices, None);
    }

    #[test]
    fn eigen_selection_index_maps_cleanly() {
        let (range, values, indices) =
            selection_parts::<f64>(EigenSelection::ByIndex { start: 2, end: 5 });
        assert_eq!(range, EigenRange::Index);
        assert_eq!(values, None);
        assert_eq!(indices, Some((2, 5)));
    }

    #[test]
    fn syev_batched_descriptor_preserves_operation_shape() {
        let op = SyevBatched::new(EigenMode::Vector, FillMode::Lower, 16, 20, 7);

        assert_eq!(op.mode, EigenMode::Vector);
        assert_eq!(op.fill_mode, FillMode::Lower);
        assert_eq!(op.n, 16);
        assert_eq!(op.leading_dimension, 20);
        assert_eq!(op.batch_count, 7);
    }

    #[test]
    fn geev_descriptor_preserves_operation_shape() {
        let op = Geev::new(12, 16);

        assert_eq!(op.n, 12);
        assert_eq!(op.leading_dimension, 16);
    }
}
