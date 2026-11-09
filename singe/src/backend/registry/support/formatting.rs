use crate::{
    backend::registry::{
        CudaGraphProvider,
        support::{BackendExecution, LayoutSupport, OperationSupport},
    },
    data_type::DataType,
};

impl OperationSupport {
    /// Returns supported output data-type names separated by commas, or `"none"` when the mode has none.
    #[must_use]
    pub fn data_type_list(&self, execution: BackendExecution) -> String {
        data_type_list(self.data_types(execution))
    }

    /// Returns supported primary input data-type names separated by commas, or `"none"` when the mode has none.
    #[must_use]
    pub fn input_data_type_list(&self, execution: BackendExecution) -> String {
        data_type_list(self.input_data_types(execution))
    }

    /// Returns registered CUDA graph provider names separated by commas, or `"none"` when the list is empty.
    #[must_use]
    pub fn provider_list(&self) -> String {
        if self.cuda_graph_providers.is_empty() {
            return "none".into();
        }

        self.cuda_graph_providers
            .iter()
            .map(|provider| match provider {
                CudaGraphProvider::Alias => "alias",
                CudaGraphProvider::Kernel => "kernel",
                CudaGraphProvider::CudnnBackend => "cudnn",
                CudaGraphProvider::CublasLt => "cublaslt",
                CudaGraphProvider::Cublas => "cublas",
                CudaGraphProvider::Cutensor => "cutensor",
                CudaGraphProvider::Cufft => "cufft",
                CudaGraphProvider::Fusion => "fusion",
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Returns a descriptive name for the operation's physical layout requirement.
    #[must_use]
    pub fn layout_name(&self) -> &'static str {
        match self.layout {
            LayoutSupport::AnyStrided => "any strided",
            LayoutSupport::MaterializesToContiguous => "materializes to contiguous",
            LayoutSupport::RequiresContiguous => "requires contiguous",
        }
    }
}

fn data_type_list(data_types: &[DataType]) -> String {
    if data_types.is_empty() {
        return "none".into();
    }

    data_types
        .iter()
        .map(|data_type| format!("{data_type:?}"))
        .collect::<Vec<_>>()
        .join(", ")
}
