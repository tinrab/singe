use std::{
    collections::HashMap,
    env, fs, io,
    path::{Path, PathBuf},
    process::Command,
};

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use sha2::{Digest, Sha256};
use singe_ptx::{
    ast::{Parameter, ScalarType, TopLevelItem},
    parser::parse_module,
};
use syn::{
    Error as SynError, Ident, LitStr, Result as SynResult, Token, Visibility, braced, bracketed,
    parse::{Parse, ParseStream},
};
use thiserror::Error;
use tree_sitter::{Node, Parser};

pub fn expand_module(input: TokenStream) -> SynResult<TokenStream> {
    let input = syn::parse2::<CudaModuleInput>(input)?;
    let metadata = extract_metadata(&input.config)?;
    generate_module(&input.visibility, &input.module, &input.config, &metadata)
}

struct CudaModuleInput {
    visibility: Visibility,
    module: Ident,
    config: ModuleConfig,
}

impl Parse for CudaModuleInput {
    fn parse(input: ParseStream) -> SynResult<Self> {
        let visibility = input.parse::<Visibility>()?;
        input.parse::<Token![mod]>()?;
        let module = input.parse::<Ident>()?;

        let content;
        braced!(content in input);

        Ok(Self {
            visibility,
            module,
            config: ModuleConfig::parse_body(&content)?,
        })
    }
}

struct ModuleConfig {
    source: LitStr,
    exports: Vec<ExportSpec>,
    headers: Vec<HeaderSpec>,
    nvcc_args: Vec<LitStr>,
    nvrtc_args: Vec<LitStr>,
}

impl ModuleConfig {
    fn parse_body(input: ParseStream) -> SynResult<Self> {
        let mut source = None;
        let mut exports = Vec::new();
        let mut headers = Vec::new();
        let mut nvcc_args = Vec::new();
        let mut nvrtc_args = Vec::new();
        let mut compile = false;

        while !input.is_empty() {
            let key = input.parse::<Ident>()?;
            input.parse::<Token![:]>()?;

            match key.to_string().as_str() {
                "source" => {
                    if source.is_some() {
                        return Err(SynError::new(key.span(), "duplicate `source` field"));
                    }
                    source = Some(input.parse::<LitStr>()?);
                }
                "exports" => {
                    if !exports.is_empty() {
                        return Err(SynError::new(key.span(), "duplicate `exports` field"));
                    }
                    exports = parse_export_specs(input)?;
                }
                "headers" => {
                    if !headers.is_empty() {
                        return Err(SynError::new(key.span(), "duplicate `headers` field"));
                    }
                    headers = parse_header_specs(input)?;
                }
                "compile" => {
                    if compile {
                        return Err(SynError::new(key.span(), "duplicate `compile` field"));
                    }
                    compile = true;
                    let config = parse_compile_config(input)?;
                    nvcc_args = config.nvcc_args;
                    nvrtc_args = config.nvrtc_args;
                }
                _ => return Err(SynError::new(key.span(), format!("unknown field `{key}`"))),
            }

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        let Some(source) = source else {
            return Err(SynError::new(Span::call_site(), "missing `source` field"));
        };

        Ok(Self {
            source,
            exports,
            headers,
            nvcc_args,
            nvrtc_args,
        })
    }
}

struct CompileConfig {
    nvcc_args: Vec<LitStr>,
    nvrtc_args: Vec<LitStr>,
}

struct ExportSpec {
    source_name: Ident,
    rust_name: Option<Ident>,
}

#[derive(Clone)]
struct HeaderSpec {
    include_name: LitStr,
    source: LitStr,
}

struct ModuleMetadata {
    cache_key: String,
    kernels: Vec<KernelMetadata>,
}

struct NvccToolchain {
    path: PathBuf,
    identity: String,
}

struct KernelMetadata {
    ptx_name: String,
    rust_name: Ident,
    params: Vec<KernelParam>,
}

struct KernelParam {
    rust_name: Ident,
    rust_type: TokenStream,
    memory_type: Option<TokenStream>,
    memory_arg: Option<TokenStream>,
}

struct SourceKernel {
    name: String,
    exported_name: String,
    extern_c: bool,
    params: Vec<SourceParam>,
}

struct SourceParam {
    name: String,
    declaration: String,
    kind: SourceParamKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum SourceParamKind {
    Scalar(Option<RustScalar>),
    Pointer {
        levels: Vec<bool>,
        element: Option<RustScalar>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RustScalar {
    Bool,
    F16,
    Bf16,
    F32,
    F64,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    USize,
    ISize,
}

#[derive(Debug, Error)]
enum MacroError {
    #[error("failed to create kernel cache directory: {0}")]
    CreateCacheDir(io::Error),
    #[error("failed to write temporary cuda source: {0}")]
    WriteSource(io::Error),
    #[error("failed to write temporary header `{header}`: {source}")]
    WriteHeader { header: String, source: io::Error },
    #[error("failed to read generated ptx: {0}")]
    ReadPtx(io::Error),
    #[error("nvcc not found")]
    NvccNotFound,
    #[error("failed to invoke nvcc: {0}")]
    NvccInvocation(io::Error),
    #[error("failed to query nvcc version: {0}")]
    NvccVersion(io::Error),
    #[error("nvcc failed: {0}")]
    NvccFailed(String),
    #[error("failed to parse ptx: {0}")]
    ParsePtx(String),
    #[error("failed to parse cuda source: {0}")]
    ParseSource(String),
    #[error("kernel `{kernel}` from source was not found in ptx entries ({available})")]
    MissingSourceKernel { kernel: String, available: String },
    #[error(
        "kernel `{kernel}` matched multiple ptx entries ({matches}); use `extern \"C\"` or simplify the declaration"
    )]
    AmbiguousPtxKernel { kernel: String, matches: String },
    #[error("kernel `{0}` parameter count mismatch between source and ptx")]
    ParameterCountMismatch(String),
    #[error("unsupported ptx parameter type for `{kernel}`: {detail}")]
    UnsupportedParameter { kernel: String, detail: String },
    #[error("failed to resolve cargo manifest directory")]
    MissingManifestDir,
    #[error("header include name must be a relative include path")]
    InvalidHeaderName,
    #[error("multiple source kernels named `{0}` require a more robust parser")]
    DuplicateSourceKernel(String),
    #[error(
        "exports field references unknown source kernel `{kernel}`; available source kernels: {available}"
    )]
    UnknownExport { kernel: String, available: String },
    #[error("duplicate rust export name `{0}`")]
    DuplicateExport(String),
    #[error("no kernels were found in source")]
    MissingSourceKernels,
}

fn extract_metadata(config: &ModuleConfig) -> SynResult<ModuleMetadata> {
    let source = config.source.value();
    let nvcc = resolve_nvcc_toolchain()?;
    let headers = config
        .headers
        .iter()
        .map(|header| (header.include_name.value(), header.source.value()))
        .collect::<Vec<_>>();
    let nvcc_args = config
        .nvcc_args
        .iter()
        .map(LitStr::value)
        .collect::<Vec<_>>();

    let cache_key = cache_key(
        &source,
        &nvcc.identity,
        &config.exports,
        &headers,
        &nvcc_args,
        &config.nvrtc_args,
    );
    let cache_dir = kernel_cache_dir().map_err(to_syn_error)?;
    fs::create_dir_all(&cache_dir).map_err(|err| to_syn_error(MacroError::CreateCacheDir(err)))?;

    let source_path = cache_dir.join(format!("{cache_key}.cu"));
    let ptx_path = cache_dir.join(format!("{cache_key}.ptx"));
    let header_dir = cache_dir.join(format!("{cache_key}-headers"));

    fs::write(&source_path, &source).map_err(|err| to_syn_error(MacroError::WriteSource(err)))?;
    materialize_headers(&header_dir, &headers)?;

    if !ptx_path.is_file() {
        compile_to_ptx(&nvcc.path, &source_path, &ptx_path, &header_dir, &nvcc_args)?;
    }

    let ptx =
        fs::read_to_string(&ptx_path).map_err(|err| to_syn_error(MacroError::ReadPtx(err)))?;
    let parsed =
        parse_module(&ptx).map_err(|err| to_syn_error(MacroError::ParsePtx(err.to_string())))?;
    let source_kernels = parse_source_kernels(&source)?;
    if source_kernels.is_empty() {
        return Err(to_syn_error(MacroError::MissingSourceKernels));
    }

    let mut source_names = HashMap::new();
    for kernel in &source_kernels {
        if source_names.insert(kernel.name.clone(), ()).is_some() {
            return Err(to_syn_error(MacroError::DuplicateSourceKernel(
                kernel.name.clone(),
            )));
        }
    }

    let selected_kernels = if config.exports.is_empty() {
        source_kernels
            .iter()
            .map(|kernel| SelectedKernel {
                kernel,
                rust_name: None,
            })
            .collect::<Vec<_>>()
    } else {
        let exports = config
            .exports
            .iter()
            .map(|export| {
                let source_name = export.source_name.to_string();
                let source_kernel = source_kernels
                    .iter()
                    .find(|kernel| kernel.exported_name == source_name)
                    .ok_or_else(|| {
                        to_syn_error(MacroError::UnknownExport {
                            kernel: source_name.clone(),
                            available: join_kernel_names(
                                source_kernels
                                    .iter()
                                    .map(|kernel| kernel.exported_name.as_str()),
                            ),
                        })
                    })?;
                Ok((source_kernel, export.rust_name.clone()))
            })
            .collect::<SynResult<Vec<_>>>()?;

        let mut rust_names = HashMap::new();
        for (kernel, rust_name) in &exports {
            let rust_name = rust_name
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_else(|| sanitize_identifier(&kernel.exported_name));
            if rust_names.insert(rust_name.clone(), ()).is_some() {
                return Err(to_syn_error(MacroError::DuplicateExport(rust_name)));
            }
        }

        exports
            .into_iter()
            .map(|(kernel, rust_name)| SelectedKernel { kernel, rust_name })
            .collect::<Vec<_>>()
    };

    let mut kernels = Vec::new();
    for selected in selected_kernels {
        let source_kernel = selected.kernel;
        let matches = parsed
            .items
            .iter()
            .filter_map(|item| match item {
                TopLevelItem::Function(function) if function.entry => Some(function),
                _ => None,
            })
            .filter(|function| {
                function.name == source_kernel.name
                    || demangle_like_name(&function.name) == Some(source_kernel.name.as_str())
            })
            .collect::<Vec<_>>();

        let ptx_function = match matches.as_slice() {
            [] => {
                return Err(to_syn_error(MacroError::MissingSourceKernel {
                    kernel: source_kernel.name.clone(),
                    available: join_kernel_names(parsed_entry_names(&parsed.items)),
                }));
            }
            [function] => *function,
            _ if source_kernel.extern_c => {
                let match_names =
                    join_kernel_names(matches.iter().map(|function| function.name.as_str()));
                matches
                    .iter()
                    .copied()
                    .find(|function| function.name == source_kernel.name)
                    .ok_or_else(|| {
                        to_syn_error(MacroError::AmbiguousPtxKernel {
                            kernel: source_kernel.name.clone(),
                            matches: match_names.clone(),
                        })
                    })?
            }
            _ => {
                let match_names =
                    join_kernel_names(matches.iter().map(|function| function.name.as_str()));
                return Err(to_syn_error(MacroError::AmbiguousPtxKernel {
                    kernel: source_kernel.name.clone(),
                    matches: match_names,
                }));
            }
        };

        if source_kernel.params.len() != ptx_function.params.len() {
            return Err(to_syn_error(MacroError::ParameterCountMismatch(
                source_kernel.name.clone(),
            )));
        }

        let params = ptx_function
            .params
            .iter()
            .zip(&source_kernel.params)
            .map(|(param, source_param)| map_param(&source_kernel.name, param, source_param))
            .collect::<SynResult<Vec<_>>>()?;

        kernels.push(KernelMetadata {
            ptx_name: ptx_function.name.clone(),
            rust_name: selected.rust_name.unwrap_or_else(|| {
                format_ident!("{}", sanitize_identifier(&source_kernel.exported_name))
            }),
            params,
        });
    }

    Ok(ModuleMetadata { cache_key, kernels })
}

fn generate_module(
    visibility: &Visibility,
    module: &Ident,
    config: &ModuleConfig,
    metadata: &ModuleMetadata,
) -> SynResult<TokenStream> {
    let source = &config.source;
    let cache_key = &metadata.cache_key;
    let header_literals = config.headers.iter().map(|header| {
        let include_name = &header.include_name;
        let source = &header.source;
        quote! {
            Header {
                include_name: #include_name,
                source: #source,
            }
        }
    });
    let nvrtc_args = config.nvrtc_args.iter();

    let methods = metadata.kernels.iter().map(|kernel| {
        let ptx_name = &kernel.ptx_name;
        let name = &kernel.rust_name;
        let on_name = format_ident!("{}_on", name);
        let record_name = format_ident!("{}_record", name);
        let node_name = format_ident!("{}_node", name);
        let set_node_params_name = format_ident!("{}_set_node_params", name);
        let memory_name = format_ident!("{}_with_memory", name);
        let memory_on_name = format_ident!("{}_with_memory_on", name);
        let memory_record_name = format_ident!("{}_with_memory_record", name);
        let memory_node_name = format_ident!("{}_with_memory_node", name);
        let memory_set_node_params_name = format_ident!("{}_with_memory_set_node_params", name);
        let args = kernel
            .params
            .iter()
            .map(|param| {
                let name = &param.rust_name;
                let ty = &param.rust_type;
                quote! { #name: #ty }
            })
            .collect::<Vec<_>>();
        let launch_args = args.clone();
        let on_args = args.clone();
        let node_args = args.clone();
        let arg_names = kernel
            .params
            .iter()
            .map(|param| {
                let name = &param.rust_name;
                quote! { params.arg(&#name); }
            })
            .collect::<Vec<_>>();
        let launch_arg_names = arg_names.clone();
        let on_arg_names = arg_names.clone();
        let node_arg_names = arg_names.clone();

        let memory_methods = if kernel
            .params
            .iter()
            .all(|param| param.memory_type.is_some() && param.memory_arg.is_some())
        {
            let memory_args = kernel
                .params
                .iter()
                .map(|param| {
                    let name = &param.rust_name;
                    let ty = param.memory_type.as_ref().expect("checked above");
                    quote! { #name: #ty }
                })
                .collect::<Vec<_>>();
            let memory_launch_args = memory_args.clone();
            let memory_on_args = memory_args.clone();
            let memory_record_args = memory_args.clone();
            let memory_node_args = memory_args.clone();
            let memory_arg_names = kernel
                .params
                .iter()
                .map(|param| param.memory_arg.as_ref().expect("checked above"))
                .collect::<Vec<_>>();
            let memory_launch_arg_names = memory_arg_names.clone();
            let memory_on_arg_names = memory_arg_names.clone();
            let memory_node_arg_names = memory_arg_names.clone();

            quote! {
                /// Launches the kernel using `DeviceMemory` parameters for pointer arguments.
                ///
                /// # Safety
                ///
                /// The caller must ensure the kernel only accesses memory within the
                /// provided allocations and that all referenced device memory remains
                /// alive until the launch has finished executing.
                pub unsafe fn #memory_name(
                    &self,
                    config: &LaunchConfig,
                    #(#memory_launch_args),*
                ) -> Result<()> {
                    unsafe {
                        self.#name(
                            config,
                            #(#memory_launch_arg_names),*
                        )
                    }
                }

                /// Launches the kernel on `stream` using `DeviceMemory` parameters for pointer arguments.
                ///
                /// # Safety
                ///
                /// The caller must ensure the kernel only accesses memory within the
                /// provided allocations and that all referenced device memory remains
                /// alive until the stream has finished executing this launch.
                pub unsafe fn #memory_on_name(
                    &self,
                    config: &LaunchConfig,
                    stream: &Stream,
                    #(#memory_on_args),*
                ) -> Result<()> {
                    unsafe {
                        self.#on_name(
                            config,
                            stream,
                            #(#memory_on_arg_names),*
                        )
                    }
                }

                /// Records a captured kernel launch using `DeviceMemory` parameters for pointer arguments.
                ///
                /// # Safety
                ///
                /// The caller must ensure the graph execution only accesses memory within
                /// the provided allocations and that all referenced device memory remains
                /// alive until every captured graph execution has finished.
                pub unsafe fn #memory_record_name(
                    &self,
                    scope: &StreamCaptureScope<'_>,
                    config: &LaunchConfig,
                    #(#memory_record_args),*
                ) -> Result<()> {
                    unsafe {
                        self.#record_name(
                            scope,
                            config,
                            #(#memory_arg_names),*
                        )
                    }
                }

                /// Adds a graph node using `DeviceMemory` parameters for pointer arguments.
                ///
                /// # Safety
                ///
                /// CUDA copies each kernel argument value during this call. For
                /// device-memory arguments, CUDA stores only the device pointer
                /// address. The caller must ensure the graph execution only
                /// accesses memory within the provided allocations and that all
                /// referenced device memory remains alive until every execution
                /// using the node has finished.
                pub unsafe fn #memory_node_name(
                    &self,
                    graph: &mut Graph,
                    dependencies: &[GraphNode],
                    config: &LaunchConfig,
                    #(#memory_node_args),*
                ) -> Result<GraphNode> {
                    unsafe {
                        self.#node_name(
                            graph,
                            dependencies,
                            config,
                            #(#memory_node_arg_names),*
                        )
                    }
                }

                /// Updates graph node parameters using `DeviceMemory` parameters for pointer arguments.
                ///
                /// # Safety
                ///
                /// CUDA copies each kernel argument value during this call. For
                /// device-memory arguments, CUDA stores only the device pointer
                /// address. The caller must ensure the graph execution only
                /// accesses memory within the provided allocations and that all
                /// referenced device memory remains alive until every execution
                /// using the node has finished.
                pub unsafe fn #memory_set_node_params_name(
                    &self,
                    executable: &mut ExecutableGraph,
                    node: GraphNode,
                    config: &LaunchConfig,
                    #(#memory_node_args),*
                ) -> Result<()> {
                    unsafe {
                        self.#set_node_params_name(
                            executable,
                            node,
                            config,
                            #(#memory_node_arg_names),*
                        )
                    }
                }
            }
        } else {
            quote! {}
        };

        quote! {
            /// Launches the kernel with raw CUDA pointer arguments.
            ///
            /// # Safety
            ///
            /// The caller must ensure every raw pointer argument is valid for the
            /// kernel's accesses and remains valid until the launch has finished
            /// executing.
            pub unsafe fn #name(
                &self,
                config: &LaunchConfig,
                #(#launch_args),*
            ) -> Result<()> {
                let function = self.module.function(#ptx_name)?;
                let mut params = KernelParameters::new();
                #(#launch_arg_names)*
                function.launch(config, params)
            }

            /// Launches the kernel on `stream` with raw CUDA pointer arguments.
            ///
            /// # Safety
            ///
            /// The caller must ensure every raw pointer argument is valid for the
            /// kernel's accesses and remains valid until `stream` has finished
            /// executing this launch.
            pub unsafe fn #on_name(
                &self,
                config: &LaunchConfig,
                stream: &Stream,
                #(#on_args),*
            ) -> Result<()> {
                let function = self.module.function(#ptx_name)?;
                let mut params = KernelParameters::new();
                #(#on_arg_names)*
                function.launch_on(config, params, stream)
            }

            /// Records a captured kernel launch with raw CUDA pointer arguments.
            ///
            /// # Safety
            ///
            /// The caller must ensure every raw pointer argument is valid for the
            /// kernel's accesses and remains valid until every captured graph
            /// execution using this launch has finished.
            pub unsafe fn #record_name(
                &self,
                scope: &StreamCaptureScope<'_>,
                config: &LaunchConfig,
                #(#launch_args),*
            ) -> Result<()> {
                let function = self.module.function(#ptx_name)?;
                let mut params = KernelParameters::new();
                #(#launch_arg_names)*
                scope.record(unsafe { function.launch_operation(config, &mut params) })
            }

            /// Adds a graph node with raw CUDA pointer arguments.
            ///
            /// # Safety
            ///
            /// CUDA copies each kernel argument value during this call. For raw
            /// pointer arguments, CUDA stores only the pointer address. The
            /// caller must ensure every raw pointer argument is valid for the
            /// kernel's accesses and remains valid until every graph execution
            /// using the node has finished.
            pub unsafe fn #node_name(
                &self,
                graph: &mut Graph,
                dependencies: &[GraphNode],
                config: &LaunchConfig,
                #(#node_args),*
            ) -> Result<GraphNode> {
                let function = self.module.function(#ptx_name)?;
                let mut params = KernelParameters::new();
                #(#node_arg_names)*
                unsafe { function.add_to_graph(graph, dependencies, config, &mut params) }
            }

            /// Updates graph node parameters with raw CUDA pointer arguments.
            ///
            /// # Safety
            ///
            /// CUDA copies each kernel argument value during this call. For raw
            /// pointer arguments, CUDA stores only the pointer address. The
            /// caller must ensure every raw pointer argument is valid for the
            /// kernel's accesses and remains valid until every graph execution
            /// using the node has finished.
            pub unsafe fn #set_node_params_name(
                &self,
                executable: &mut ExecutableGraph,
                node: GraphNode,
                config: &LaunchConfig,
                #(#node_args),*
            ) -> Result<()> {
                let function = self.module.function(#ptx_name)?;
                let mut params = KernelParameters::new();
                #(#node_arg_names)*
                unsafe { function.set_graph_node_params(executable, node, config, &mut params) }
            }

            #memory_methods
        }
    });

    Ok(quote! {
        #visibility mod #module {
            #[allow(unused_extern_crates, clippy::useless_attribute)]
            extern crate singe_cuda as _singe_cuda;

            use std::{
                io,
                collections::HashMap,
                sync::{Arc, Mutex, OnceLock},
            };

            use _singe_cuda::{
                architecture::GpuArchitecture,
                context::Context,
                error::Result,
                graph::{ExecutableGraph, Graph, GraphNode},
                memory::DeviceMemory,
                module::{KernelParameters, LaunchConfig, Module as CudaModule},
                nvrtc::{CompileOptions, Header, OutputKind, Program, supported_architectures},
                stream::{Stream, StreamCaptureScope},
            };

            const MODULE_CACHE_KEY_SUFFIX: &str = #cache_key;

            fn module_cache_key_suffix() -> &'static str {
                MODULE_CACHE_KEY_SUFFIX
            }

            type ModuleCacheKey = (usize, &'static str);
            type ModuleCacheMap = HashMap<ModuleCacheKey, Arc<CudaModule>>;

            static MODULE_CACHE: OnceLock<Mutex<ModuleCacheMap>> = OnceLock::new();

            fn module_cache() -> &'static Mutex<ModuleCacheMap> {
                MODULE_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
            }

            fn compile_module(
                ctx: &Arc<Context>,
                source: &str,
                headers: &[Header<'_>],
                nvrtc_args: &[&str],
            ) -> Result<CudaModule> {
                let properties = ctx.device().properties()?;
                let (architecture, output_kind) =
                    supported_gpu_architecture(properties.major, properties.minor);
                let program = Program::from_source(source)
                    .with_name("cuda_module.cu")
                    .with_headers(headers);
                let mut options = CompileOptions::new().gpu_architecture(architecture);

                for arg in nvrtc_args {
                    options = options.raw_option(arg);
                }

                program.compile_with_options(&options)?;
                ctx.load_nvrtc_module(&program, output_kind)
            }

            fn supported_gpu_architecture(major: i32, minor: i32) -> (GpuArchitecture, OutputKind) {
                let requested = major * 10 + minor;

                let supported = supported_architectures()
                    .unwrap_or_default()
                    .into_iter()
                    .collect::<Vec<_>>();

                if supported.contains(&requested) {
                    if let Some(architecture) = gpu_architecture(major, minor, true) {
                        return (architecture, OutputKind::Cubin);
                    }
                }

                let fallback = supported
                    .into_iter()
                    .filter(|value| *value <= requested)
                    .max()
                    .and_then(|value| gpu_architecture(value / 10, value % 10, false))
                    .unwrap_or(GpuArchitecture::Compute75);
                (fallback, OutputKind::Ptx)
            }

            fn gpu_architecture(major: i32, minor: i32, real: bool) -> Option<GpuArchitecture> {
                match (major, minor, real) {
                    (7, 5, false) => Some(GpuArchitecture::Compute75),
                    (7, 5, true) => Some(GpuArchitecture::Sm75),
                    (8, 0, false) => Some(GpuArchitecture::Compute80),
                    (8, 0, true) => Some(GpuArchitecture::Sm80),
                    (8, 6, false) => Some(GpuArchitecture::Compute86),
                    (8, 6, true) => Some(GpuArchitecture::Sm86),
                    (8, 7, false) => Some(GpuArchitecture::Compute87),
                    (8, 7, true) => Some(GpuArchitecture::Sm87),
                    (8, 9, false) => Some(GpuArchitecture::Compute89),
                    (8, 9, true) => Some(GpuArchitecture::Sm89),
                    (9, 0, false) => Some(GpuArchitecture::Compute90),
                    (9, 0, true) => Some(GpuArchitecture::Sm90),
                    (10, 0, false) => Some(GpuArchitecture::Compute100),
                    (10, 0, true) => Some(GpuArchitecture::Sm100),
                    (10, 1, false) => Some(GpuArchitecture::Compute101),
                    (10, 1, true) => Some(GpuArchitecture::Sm101),
                    (10, 3, false) => Some(GpuArchitecture::Compute103),
                    (10, 3, true) => Some(GpuArchitecture::Sm103),
                    (12, 0, false) => Some(GpuArchitecture::Compute120),
                    (12, 0, true) => Some(GpuArchitecture::Sm120),
                    (12, 1, false) => Some(GpuArchitecture::Compute121),
                    (12, 1, true) => Some(GpuArchitecture::Sm121),
                    _ => None,
                }
            }

            #[derive(Debug, Clone)]
            pub struct Module {
                module: Arc<CudaModule>,
            }

            impl Module {
                pub const SOURCE: &'static str = #source;

                pub fn create(ctx: &Arc<Context>) -> Result<Self> {
                    let headers = [#(#header_literals),*];
                    let nvrtc_args = [#(#nvrtc_args),*];
                    let context_key = unsafe { ctx.as_raw() as usize };
                    let cache_key = (context_key, module_cache_key_suffix());
                    let cache = module_cache();

                    if let Some(module) = cache.lock().expect("cuda module cache poisoned").get(&cache_key) {
                        return Ok(Self { module: Arc::clone(module) });
                    }

                    let module = Arc::new(compile_module(ctx, Self::SOURCE, &headers, &nvrtc_args)?);
                    let mut cache = cache.lock().expect("cuda module cache poisoned");
                    let entry = cache.entry(cache_key).or_insert_with(|| Arc::clone(&module));

                    let module = Arc::clone(entry);
                    Ok(Self { module })
                }

                pub fn raw(&self) -> &CudaModule {
                    self.module.as_ref()
                }

                #(#methods)*
            }
        }
    })
}

fn map_param(
    kernel_name: &str,
    param: &Parameter,
    source_param: &SourceParam,
) -> SynResult<KernelParam> {
    let rust_type = match &source_param.kind {
        SourceParamKind::Pointer { levels, element } => rust_pointer_type(levels, *element),
        SourceParamKind::Scalar(source_scalar) if param.ptr => {
            rust_pointer_type(&[true], *source_scalar)
        }
        SourceParamKind::Scalar(source_scalar) => {
            match (*source_scalar).or_else(|| ptx_scalar_type(param.ty)) {
                Some(scalar) => rust_scalar_type(scalar),
                None => {
                    return Err(to_syn_error(MacroError::UnsupportedParameter {
                        kernel: kernel_name.to_string(),
                        detail: format!("{} ({:?})", source_param.declaration, param.ty),
                    }));
                }
            }
        }
    };

    if !param.array_bounds.is_empty() {
        return Err(to_syn_error(MacroError::UnsupportedParameter {
            kernel: kernel_name.to_string(),
            detail: "by-value array parameters are not supported yet".to_string(),
        }));
    }

    Ok(KernelParam {
        rust_name: format_ident!("{}", sanitize_identifier(&source_param.name)),
        rust_type,
        memory_type: memory_param_type(&source_param.kind),
        memory_arg: memory_arg(&source_param.name, &source_param.kind),
    })
}

fn rust_pointer_type(levels: &[bool], element: Option<RustScalar>) -> TokenStream {
    let mut ty = element
        .map(rust_scalar_type)
        .unwrap_or_else(|| quote! { () });
    for mutable in levels {
        ty = if *mutable {
            quote! { *mut #ty }
        } else {
            quote! { *const #ty }
        };
    }
    ty
}

fn memory_param_type(kind: &SourceParamKind) -> Option<TokenStream> {
    match kind {
        SourceParamKind::Scalar(scalar) => scalar.map(rust_scalar_type),
        SourceParamKind::Pointer {
            levels,
            element: Some(element),
        } if levels.len() == 1 => {
            let ty = rust_scalar_type(*element);
            if levels[0] {
                Some(quote! { &mut DeviceMemory<#ty> })
            } else {
                Some(quote! { &DeviceMemory<#ty> })
            }
        }
        SourceParamKind::Pointer { .. } => None,
    }
}

fn memory_arg(name: &str, kind: &SourceParamKind) -> Option<TokenStream> {
    let name = format_ident!("{}", sanitize_identifier(name));
    match kind {
        SourceParamKind::Scalar(_) => Some(quote! { #name }),
        SourceParamKind::Pointer {
            levels,
            element: Some(_),
        } if levels.len() == 1 => {
            if levels[0] {
                Some(quote! { #name.as_mut_ptr() })
            } else {
                Some(quote! { #name.as_ptr() })
            }
        }
        SourceParamKind::Pointer { .. } => None,
    }
}

fn rust_scalar_type(scalar: RustScalar) -> TokenStream {
    match scalar {
        RustScalar::Bool => quote! { bool },
        RustScalar::F16 => quote! { _singe_cuda::types::f16 },
        RustScalar::Bf16 => quote! { _singe_cuda::types::bf16 },
        RustScalar::F32 => quote! { f32 },
        RustScalar::F64 => quote! { f64 },
        RustScalar::U8 => quote! { u8 },
        RustScalar::I8 => quote! { i8 },
        RustScalar::U16 => quote! { u16 },
        RustScalar::I16 => quote! { i16 },
        RustScalar::U32 => quote! { u32 },
        RustScalar::I32 => quote! { i32 },
        RustScalar::U64 => quote! { u64 },
        RustScalar::I64 => quote! { i64 },
        RustScalar::USize => quote! { usize },
        RustScalar::ISize => quote! { isize },
    }
}

fn ptx_scalar_type(ty: ScalarType) -> Option<RustScalar> {
    match ty {
        ScalarType::U8 | ScalarType::B8 => Some(RustScalar::U8),
        ScalarType::S8 => Some(RustScalar::I8),
        ScalarType::U16 | ScalarType::B16 => Some(RustScalar::U16),
        ScalarType::S16 => Some(RustScalar::I16),
        ScalarType::F16 => Some(RustScalar::F16),
        ScalarType::Bf16 => Some(RustScalar::Bf16),
        ScalarType::F32 => Some(RustScalar::F32),
        ScalarType::F64 => Some(RustScalar::F64),
        ScalarType::U32 | ScalarType::B32 => Some(RustScalar::U32),
        ScalarType::S32 => Some(RustScalar::I32),
        ScalarType::U64 | ScalarType::B64 => Some(RustScalar::U64),
        ScalarType::S64 => Some(RustScalar::I64),
        _ => None,
    }
}

fn compile_to_ptx(
    nvcc_path: &Path,
    source_path: &Path,
    ptx_path: &Path,
    header_dir: &Path,
    nvcc_args: &[String],
) -> SynResult<()> {
    let mut command = Command::new(nvcc_path);
    command.arg("--ptx");
    if header_dir.is_dir() {
        command.arg(format!("--include-path={}", header_dir.display()));
    }
    command.args(nvcc_args);
    command.arg(source_path).arg("-o").arg(ptx_path);

    let output = command
        .output()
        .map_err(|err| to_syn_error(MacroError::NvccInvocation(err)))?;
    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let message = if stderr.trim().is_empty() {
        stdout.trim().to_string()
    } else if stdout.trim().is_empty() {
        stderr.trim().to_string()
    } else {
        format!("{}\n{}", stdout.trim(), stderr.trim())
    };
    Err(to_syn_error(MacroError::NvccFailed(message)))
}

fn materialize_headers(header_dir: &Path, headers: &[(String, String)]) -> SynResult<()> {
    if headers.is_empty() {
        return Ok(());
    }

    fs::create_dir_all(header_dir).map_err(|err| to_syn_error(MacroError::CreateCacheDir(err)))?;
    for (include_name, source) in headers {
        validate_header_name(include_name)?;
        let path = header_dir.join(include_name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|err| to_syn_error(MacroError::CreateCacheDir(err)))?;
        }
        fs::write(&path, source).map_err(|err| {
            to_syn_error(MacroError::WriteHeader {
                header: include_name.clone(),
                source: err,
            })
        })?;
    }
    Ok(())
}

fn validate_header_name(include_name: &str) -> SynResult<()> {
    let path = Path::new(include_name);
    if include_name.is_empty() || path.is_absolute() || include_name.contains("..") {
        return Err(to_syn_error(MacroError::InvalidHeaderName));
    }
    Ok(())
}

fn kernel_cache_dir() -> Result<PathBuf, MacroError> {
    if let Some(out_dir) = env::var_os("OUT_DIR") {
        return Ok(PathBuf::from(out_dir).join("singe-kernels"));
    }

    let manifest_dir = env::var_os("CARGO_MANIFEST_DIR").ok_or(MacroError::MissingManifestDir)?;
    Ok(PathBuf::from(manifest_dir)
        .join("target")
        .join("singe-kernels"))
}

fn parsed_entry_names<'a>(items: &'a [TopLevelItem]) -> impl Iterator<Item = &'a str> + 'a {
    items.iter().filter_map(|item| match item {
        TopLevelItem::Function(function) if function.entry => Some(function.name.as_str()),
        _ => None,
    })
}

fn join_kernel_names<'a>(names: impl Iterator<Item = &'a str>) -> String {
    let names = names.collect::<Vec<_>>();
    if names.is_empty() {
        String::from("none")
    } else {
        names.join(", ")
    }
}

fn cache_key(
    source: &str,
    nvcc_identity: &str,
    exports: &[ExportSpec],
    headers: &[(String, String)],
    nvcc_args: &[String],
    nvrtc_args: &[LitStr],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    hasher.update(nvcc_identity.as_bytes());
    hasher.update([0]);
    for export in exports {
        hasher.update(export.source_name.to_string().as_bytes());
        hasher.update([0]);
        if let Some(rust_name) = &export.rust_name {
            hasher.update(rust_name.to_string().as_bytes());
        }
        hasher.update([0]);
    }
    for (include_name, header_source) in headers {
        hasher.update(include_name.as_bytes());
        hasher.update([0]);
        hasher.update(header_source.as_bytes());
        hasher.update([0]);
    }
    for arg in nvcc_args {
        hasher.update(arg.as_bytes());
        hasher.update([0]);
    }
    for arg in nvrtc_args {
        hasher.update(arg.value().as_bytes());
        hasher.update([0]);
    }
    hex::encode(hasher.finalize())
}

fn parse_export_specs(input: ParseStream) -> SynResult<Vec<ExportSpec>> {
    let content;
    braced!(content in input);
    let mut exports = Vec::new();

    while !content.is_empty() {
        let source_name = content.parse::<Ident>()?;
        let rust_name = if content.peek(Token![as]) {
            content.parse::<Token![as]>()?;
            Some(content.parse::<Ident>()?)
        } else {
            None
        };
        exports.push(ExportSpec {
            source_name,
            rust_name,
        });
        if content.peek(Token![,]) {
            content.parse::<Token![,]>()?;
        }
    }

    Ok(exports)
}

fn parse_header_specs(input: ParseStream) -> SynResult<Vec<HeaderSpec>> {
    let content;
    braced!(content in input);
    let mut headers = Vec::new();
    while !content.is_empty() {
        let include_name = content.parse::<LitStr>()?;
        content.parse::<Token![=>]>()?;
        let source = content.parse::<LitStr>()?;
        headers.push(HeaderSpec {
            include_name,
            source,
        });
        if content.peek(Token![,]) {
            content.parse::<Token![,]>()?;
        }
    }
    Ok(headers)
}

fn parse_compile_config(input: ParseStream) -> SynResult<CompileConfig> {
    let content;
    braced!(content in input);

    let mut nvcc_args = None;
    let mut nvrtc_args = None;

    while !content.is_empty() {
        let key = content.parse::<Ident>()?;
        content.parse::<Token![:]>()?;

        match key.to_string().as_str() {
            "nvcc_args" => {
                if nvcc_args.is_some() {
                    return Err(SynError::new(key.span(), "duplicate `nvcc_args` field"));
                }
                nvcc_args = Some(parse_string_list(&content)?);
            }
            "nvrtc_args" => {
                if nvrtc_args.is_some() {
                    return Err(SynError::new(key.span(), "duplicate `nvrtc_args` field"));
                }
                nvrtc_args = Some(parse_string_list(&content)?);
            }
            _ => {
                return Err(SynError::new(
                    key.span(),
                    format!("unknown compile field `{key}`"),
                ));
            }
        }

        if content.peek(Token![,]) {
            content.parse::<Token![,]>()?;
        }
    }

    Ok(CompileConfig {
        nvcc_args: nvcc_args.unwrap_or_default(),
        nvrtc_args: nvrtc_args.unwrap_or_default(),
    })
}

fn parse_string_list(input: ParseStream) -> SynResult<Vec<LitStr>> {
    let content;
    bracketed!(content in input);
    let mut values = Vec::new();
    while !content.is_empty() {
        values.push(content.parse::<LitStr>()?);
        if content.peek(Token![,]) {
            content.parse::<Token![,]>()?;
        }
    }
    Ok(values)
}

fn parse_source_kernels(source: &str) -> SynResult<Vec<SourceKernel>> {
    let mut parser = Parser::new();
    let language = tree_sitter_c::LANGUAGE;
    parser
        .set_language(&language.into())
        .map_err(|err| to_syn_error(MacroError::ParseSource(err.to_string())))?;
    let tree = parser
        .parse(source, None)
        .ok_or_else(|| to_syn_error(MacroError::ParseSource("parser returned no tree".into())))?;

    let mut kernels = Vec::new();
    collect_source_kernels(source, tree.root_node(), &mut kernels);
    Ok(kernels)
}

fn collect_source_kernels(source: &str, node: Node<'_>, kernels: &mut Vec<SourceKernel>) {
    if node.kind() == "function_definition"
        && let Some(kernel) = parse_source_kernel(source, node)
    {
        kernels.push(kernel);
    }

    for index in 0..node.child_count() {
        if let Some(child) = node_child(node, index) {
            collect_source_kernels(source, child, kernels);
        }
    }
}

fn parse_source_kernel(source: &str, node: Node<'_>) -> Option<SourceKernel> {
    let declarator = function_declarator(node)?;
    let prefix = &source[node.start_byte()..declarator.start_byte()];
    if !contains_identifier(prefix, "__global__") {
        return None;
    }

    let name_node = declarator
        .child_by_field_name("declarator")
        .and_then(declarator_name)?;
    let name = node_text(source, name_node).trim().to_string();
    let params = declarator
        .child_by_field_name("parameters")
        .or_else(|| find_descendant_kind(declarator, "parameter_list"))
        .map(|parameters| parse_source_parameters(source, parameters))
        .unwrap_or_default();

    Some(SourceKernel {
        exported_name: name.clone(),
        name,
        extern_c: has_extern_c_linkage(source, node) || prefix.contains("extern \"C\""),
        params,
    })
}

fn function_declarator<'a>(node: Node<'a>) -> Option<Node<'a>> {
    node.child_by_field_name("declarator")
        .and_then(|declarator| find_descendant_kind(declarator, "function_declarator"))
}

fn parse_source_parameters(source: &str, parameters: Node<'_>) -> Vec<SourceParam> {
    let mut params = Vec::new();
    for index in 0..parameters.child_count() {
        let Some(child) = node_child(parameters, index) else {
            continue;
        };
        if child.kind() != "parameter_declaration" {
            continue;
        }
        if let Some(param) = parse_source_parameter(source, child, params.len()) {
            params.push(param);
        }
    }
    params
}

fn parse_source_parameter(source: &str, node: Node<'_>, index: usize) -> Option<SourceParam> {
    let declaration = node_text(source, node).trim();
    if declaration.is_empty() || declaration == "void" {
        return None;
    }

    let declarator = node.child_by_field_name("declarator");
    let name_node = declarator.and_then(declarator_name);
    let name = name_node
        .map(|node| node_text(source, node).trim().to_string())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| format!("arg{index}"));
    let array = declarator.is_some_and(|node| declarator_has_kind(node, "array_declarator"));
    let levels = pointer_levels_from_declaration(declaration, array);
    let scalar = source_scalar_type(declaration, &name);
    let kind = if levels.is_empty() {
        SourceParamKind::Scalar(scalar)
    } else {
        SourceParamKind::Pointer {
            levels,
            element: scalar,
        }
    };

    Some(SourceParam {
        name,
        declaration: declaration.to_string(),
        kind,
    })
}

fn pointer_levels_from_declaration(declaration: &str, array: bool) -> Vec<bool> {
    let mut levels = Vec::new();
    let mut previous = 0usize;
    for (star, _) in declaration.match_indices('*') {
        levels.push(!contains_identifier(&declaration[previous..star], "const"));
        previous = star + 1;
    }
    if array {
        let segment = if levels.is_empty() {
            declaration
        } else {
            &declaration[previous..]
        };
        levels.push(!contains_identifier(segment, "const"));
    }
    levels
}

fn source_scalar_type(declaration: &str, name: &str) -> Option<RustScalar> {
    let tokens = declaration_type_tokens(declaration, name);
    if tokens.iter().any(|token| token == "void") {
        return None;
    }
    if tokens
        .iter()
        .any(|token| token == "bool" || token == "_bool")
    {
        return Some(RustScalar::Bool);
    }
    if tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "__half" | "half" | "__nv_half" | "cuda_fp16"
        )
    }) {
        return Some(RustScalar::F16);
    }
    if tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "__nv_bfloat16" | "nv_bfloat16" | "bfloat16" | "bf16"
        )
    }) {
        return Some(RustScalar::Bf16);
    }
    if tokens.iter().any(|token| token == "double") {
        return Some(RustScalar::F64);
    }
    if tokens.iter().any(|token| token == "float") {
        return Some(RustScalar::F32);
    }
    if tokens.iter().any(|token| token == "size_t") {
        return Some(RustScalar::USize);
    }
    if tokens
        .iter()
        .any(|token| matches!(token.as_str(), "ssize_t" | "ptrdiff_t"))
    {
        return Some(RustScalar::ISize);
    }
    if tokens
        .iter()
        .any(|token| matches!(token.as_str(), "uint64_t" | "uintptr_t"))
    {
        return Some(RustScalar::U64);
    }
    if tokens
        .iter()
        .any(|token| matches!(token.as_str(), "int64_t" | "intptr_t"))
    {
        return Some(RustScalar::I64);
    }
    if tokens.iter().any(|token| token == "uint32_t") {
        return Some(RustScalar::U32);
    }
    if tokens.iter().any(|token| token == "int32_t") {
        return Some(RustScalar::I32);
    }
    if tokens.iter().any(|token| token == "uint16_t") {
        return Some(RustScalar::U16);
    }
    if tokens.iter().any(|token| token == "int16_t") {
        return Some(RustScalar::I16);
    }
    if tokens.iter().any(|token| token == "uint8_t") {
        return Some(RustScalar::U8);
    }
    if tokens.iter().any(|token| token == "int8_t") {
        return Some(RustScalar::I8);
    }

    let unsigned = tokens.iter().any(|token| token == "unsigned");
    let signed = tokens.iter().any(|token| token == "signed");
    let long_count = tokens.iter().filter(|token| *token == "long").count();
    if long_count >= 2 {
        return Some(if unsigned {
            RustScalar::U64
        } else {
            RustScalar::I64
        });
    }
    if tokens.iter().any(|token| token == "short") {
        return Some(if unsigned {
            RustScalar::U16
        } else {
            RustScalar::I16
        });
    }
    if tokens.iter().any(|token| token == "char") {
        return match (unsigned, signed) {
            (true, _) => Some(RustScalar::U8),
            (false, true) => Some(RustScalar::I8),
            _ => None,
        };
    }
    if tokens.iter().any(|token| token == "int") {
        return Some(if unsigned {
            RustScalar::U32
        } else {
            RustScalar::I32
        });
    }
    None
}

fn declaration_type_tokens(declaration: &str, name: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let name = name.to_ascii_lowercase();
    for ch in declaration.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            current.push(ch.to_ascii_lowercase());
            continue;
        }
        push_type_token(&mut tokens, &mut current, &name);
    }
    push_type_token(&mut tokens, &mut current, &name);
    tokens
}

fn push_type_token(tokens: &mut Vec<String>, current: &mut String, name: &str) {
    if current.is_empty() {
        return;
    }
    if current != name && !is_ignored_type_token(current) {
        tokens.push(std::mem::take(current));
    } else {
        current.clear();
    }
}

fn is_ignored_type_token(token: &str) -> bool {
    matches!(
        token,
        "const"
            | "volatile"
            | "restrict"
            | "__restrict"
            | "__restrict__"
            | "static"
            | "register"
            | "extern"
            | "__global__"
            | "__device__"
            | "__host__"
            | "__forceinline__"
            | "inline"
    )
}

fn find_descendant_kind<'a>(node: Node<'a>, kind: &str) -> Option<Node<'a>> {
    if node.kind() == kind {
        return Some(node);
    }
    for index in 0..node.child_count() {
        if let Some(child) = node_child(node, index)
            && let Some(found) = find_descendant_kind(child, kind)
        {
            return Some(found);
        }
    }
    None
}

fn declarator_name<'a>(node: Node<'a>) -> Option<Node<'a>> {
    if matches!(node.kind(), "identifier" | "field_identifier") {
        return Some(node);
    }
    if let Some(declarator) = node.child_by_field_name("declarator")
        && let Some(name) = declarator_name(declarator)
    {
        return Some(name);
    }

    let mut last = None;
    for index in 0..node.child_count() {
        if let Some(child) = node_child(node, index)
            && child.is_named()
            && let Some(name) = declarator_name(child)
        {
            last = Some(name);
        }
    }
    last
}

fn declarator_has_kind(node: Node<'_>, kind: &str) -> bool {
    if node.kind() == kind {
        return true;
    }
    for index in 0..node.child_count() {
        if let Some(child) = node_child(node, index)
            && declarator_has_kind(child, kind)
        {
            return true;
        }
    }
    false
}

fn has_extern_c_linkage(source: &str, mut node: Node<'_>) -> bool {
    while let Some(parent) = node.parent() {
        if parent.kind() == "linkage_specification"
            && let Some(value) = parent.child_by_field_name("value")
            && node_text(source, value).contains("\"C\"")
        {
            return true;
        }
        node = parent;
    }
    false
}

fn contains_identifier(source: &str, needle: &str) -> bool {
    let mut start = 0usize;
    while let Some(relative) = source[start..].find(needle) {
        let index = start + relative;
        let end = index + needle.len();
        let before = source[..index]
            .chars()
            .next_back()
            .is_none_or(|ch| !is_c_identifier_char(ch));
        let after = source[end..]
            .chars()
            .next()
            .is_none_or(|ch| !is_c_identifier_char(ch));
        if before && after {
            return true;
        }
        start = end;
    }
    false
}

fn is_c_identifier_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

fn node_text<'a>(source: &'a str, node: Node<'_>) -> &'a str {
    &source[node.start_byte()..node.end_byte()]
}

fn node_child<'a>(node: Node<'a>, index: usize) -> Option<Node<'a>> {
    u32::try_from(index)
        .ok()
        .and_then(|index| node.child(index))
}

fn resolve_nvcc_toolchain() -> SynResult<NvccToolchain> {
    let path = singe_cuda_find::find_nvcc()
        .map_err(|_| to_syn_error(MacroError::NvccNotFound))?
        .ok_or_else(|| to_syn_error(MacroError::NvccNotFound))?;
    let output = Command::new(&path)
        .arg("--version")
        .output()
        .map_err(|err| to_syn_error(MacroError::NvccVersion(err)))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let identity = format!("{}\n{}\n{}", path.display(), stdout.trim(), stderr.trim());

    Ok(NvccToolchain { path, identity })
}

struct SelectedKernel<'a> {
    kernel: &'a SourceKernel,
    rust_name: Option<Ident>,
}

fn demangle_like_name(name: &str) -> Option<&str> {
    if !name.starts_with("_Z") {
        return None;
    }

    let mut index = 2usize;
    let bytes = name.as_bytes();
    let mut length = 0usize;
    while index < bytes.len() && bytes[index].is_ascii_digit() {
        length = length * 10 + (bytes[index] - b'0') as usize;
        index += 1;
    }
    if length == 0 || index + length > name.len() {
        return None;
    }
    Some(&name[index..index + length])
}

fn sanitize_identifier(name: &str) -> String {
    let mut output = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            output.push(ch);
        } else {
            output.push('_');
        }
    }
    if output.is_empty() {
        output.push_str("arg");
    }
    if output.as_bytes()[0].is_ascii_digit() {
        output.insert(0, '_');
    }
    if is_rust_keyword(&output) {
        format!("r#{output}")
    } else {
        output
    }
}

fn is_rust_keyword(value: &str) -> bool {
    matches!(
        value,
        "as" | "break"
            | "const"
            | "continue"
            | "crate"
            | "else"
            | "enum"
            | "extern"
            | "false"
            | "fn"
            | "for"
            | "if"
            | "impl"
            | "in"
            | "let"
            | "loop"
            | "match"
            | "mod"
            | "move"
            | "mut"
            | "pub"
            | "ref"
            | "return"
            | "self"
            | "Self"
            | "static"
            | "struct"
            | "super"
            | "trait"
            | "true"
            | "type"
            | "unsafe"
            | "use"
            | "where"
            | "while"
    )
}

fn to_syn_error(error: MacroError) -> SynError {
    SynError::new(Span::call_site(), error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_kernel_parameters() {
        let source = r#"
            __device__ void helper(float* value) {}

            extern "C" __global__ void scale_add(
                const float* input,
                float* output,
                float alpha,
                int len,
                size_t count,
                unsigned long long seed,
                const __half* halves,
                unsigned int flags,
                const float* const* batches
            ) {
                int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
                if (i < len) {
                    output[i] = input[i] * alpha + 1.0f;
                }
            }
        "#;
        let kernels = parse_source_kernels(source).unwrap();
        assert_eq!(kernels.len(), 1);

        let kernel = &kernels[0];
        assert_eq!(kernel.name, "scale_add");
        assert!(kernel.extern_c);
        assert_eq!(kernel.params.len(), 9);
        assert_pointer(&kernel.params[0], &[false], Some(RustScalar::F32));
        assert_pointer(&kernel.params[1], &[true], Some(RustScalar::F32));
        assert_scalar(&kernel.params[2], Some(RustScalar::F32));
        assert_scalar(&kernel.params[3], Some(RustScalar::I32));
        assert_scalar(&kernel.params[4], Some(RustScalar::USize));
        assert_scalar(&kernel.params[5], Some(RustScalar::U64));
        assert_pointer(&kernel.params[6], &[false], Some(RustScalar::F16));
        assert_scalar(&kernel.params[7], Some(RustScalar::U32));
        assert_pointer(&kernel.params[8], &[false, false], Some(RustScalar::F32));
    }

    #[test]
    fn test_handle_array_parameters() {
        let source = r#"
            extern "C" __global__ void arrays(
                const float input[],
                unsigned int output[static 4],
                void* scratch
            ) {}
        "#;
        let kernels = parse_source_kernels(source).unwrap();
        let params = &kernels[0].params;
        assert_pointer(&params[0], &[false], Some(RustScalar::F32));
        assert_pointer(&params[1], &[true], Some(RustScalar::U32));
        assert_pointer(&params[2], &[true], None);
    }

    fn assert_scalar(param: &SourceParam, expected: Option<RustScalar>) {
        assert_eq!(param.kind, SourceParamKind::Scalar(expected));
    }

    fn assert_pointer(param: &SourceParam, levels: &[bool], element: Option<RustScalar>) {
        assert_eq!(
            param.kind,
            SourceParamKind::Pointer {
                levels: levels.to_vec(),
                element,
            }
        );
    }
}
