use std::{
    collections::{HashMap, HashSet},
    mem,
    path::Path,
};

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use sha2::{Digest, Sha256};
use syn::{
    Error as SynError, Expr, Ident, LitStr, Result as SynResult, Token, Type, Visibility, braced,
    bracketed,
    parse::{Parse, ParseStream},
    token,
};
use thiserror::Error;
use tree_sitter::{Node, Parser};

pub fn expand_module(input: TokenStream) -> SynResult<TokenStream> {
    let input = syn::parse2::<CudaModuleInput>(input)?;
    let metadata = extract_metadata(&input.config)?;
    Ok(generate_module(
        &input.visibility,
        &input.module,
        &input.config,
        &metadata,
    ))
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
    nvrtc_args: Vec<LitStr>,
}

impl ModuleConfig {
    fn parse_body(input: ParseStream) -> SynResult<Self> {
        let mut source = None;
        let mut exports = Vec::new();
        let mut headers = Vec::new();
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
            nvrtc_args,
        })
    }
}

struct CompileConfig {
    nvrtc_args: Vec<LitStr>,
}

struct ExportSpec {
    source_name: Ident,
    rust_name: Option<Ident>,
    parameters: HashMap<String, ParameterSpec>,
    requirements: Option<Expr>,
}

struct ParameterSpec {
    access: ParameterMode,
    rust_type: Option<Type>,
}

#[derive(Clone, Copy)]
enum ParameterMode {
    Value,
    Read,
    Write,
    ReadWrite,
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

struct KernelMetadata {
    source_name: String,
    rust_name: Ident,
    params: Vec<KernelParam>,
    requirements: TokenStream,
}

struct KernelParam {
    rust_name: Ident,
    source_name: String,
    rust_type: TokenStream,
    typed_argument: Option<TypedArgument>,
    layout: TokenStream,
    context_check: Option<TokenStream>,
    view_type: TokenStream,
    view_binding: TokenStream,
    raw_value: TokenStream,
}

struct TypedArgument {
    rust_type: TokenStream,
    value: TokenStream,
}

struct SourceKernel {
    name: String,
    exported_name: String,
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
    #[error("failed to parse cuda source: {0}")]
    ParseSource(String),
    #[error("unsupported source parameter type for `{kernel}`: {detail}")]
    UnsupportedParameter { kernel: String, detail: String },
    #[error("header include name must be a relative include path")]
    InvalidHeaderName,
    #[error("multiple source kernels named `{0}` require a more robust parser")]
    DuplicateSourceKernel(String),
    #[error(
        "exports field references unknown source kernel `{kernel}`; available source kernels: {available}"
    )]
    UnknownExport { kernel: String, available: String },
    #[error("no kernels were found in source")]
    MissingSourceKernels,
}

fn extract_metadata(config: &ModuleConfig) -> SynResult<ModuleMetadata> {
    let source = config.source.value();
    let headers = config
        .headers
        .iter()
        .map(|header| (header.include_name.value(), header.source.value()))
        .collect::<Vec<_>>();

    for (include_name, _) in &headers {
        validate_header_name(include_name)?;
    }

    let cache_key = cache_key(&source, &config.exports, &headers, &config.nvrtc_args);
    let source_kernels = parse_source_kernels(&source)?;
    if source_kernels.is_empty() {
        return Err(to_syn_error(&MacroError::MissingSourceKernels));
    }

    let mut source_names = HashSet::new();
    for kernel in &source_kernels {
        if !source_names.insert(kernel.name.clone()) {
            return Err(to_syn_error(&MacroError::DuplicateSourceKernel(
                kernel.name.clone(),
            )));
        }
    }

    let selected_kernels = if config.exports.is_empty() {
        source_kernels
            .iter()
            .map(|kernel| SelectedKernel { kernel, spec: None })
            .collect::<Vec<_>>()
    } else {
        config
            .exports
            .iter()
            .map(|export| {
                let source_name = export
                    .source_name
                    .to_string()
                    .trim_start_matches("r#")
                    .to_owned();
                let source_kernel = source_kernels
                    .iter()
                    .find(|kernel| kernel.exported_name == source_name)
                    .ok_or_else(|| {
                        to_syn_error(&MacroError::UnknownExport {
                            kernel: source_name.clone(),
                            available: join_kernel_names(
                                source_kernels
                                    .iter()
                                    .map(|kernel| kernel.exported_name.as_str()),
                            ),
                        })
                    })?;
                Ok(SelectedKernel {
                    kernel: source_kernel,
                    spec: Some(export),
                })
            })
            .collect::<SynResult<Vec<_>>>()?
    };

    let mut kernels = Vec::new();
    for selected in selected_kernels {
        let source_kernel = selected.kernel;
        let spec = selected.spec;
        if let Some(spec) = spec {
            for name in spec.parameters.keys() {
                if !source_kernel.params.iter().any(|param| param.name == *name) {
                    return Err(SynError::new(
                        spec.source_name.span(),
                        format!("unknown parameter `{name}`"),
                    ));
                }
            }
        }
        let params = source_kernel
            .params
            .iter()
            .enumerate()
            .map(|(index, source_param)| {
                map_param(
                    index,
                    &source_kernel.name,
                    source_param,
                    spec.and_then(|s| s.parameters.get(&source_param.name)),
                )
            })
            .collect::<SynResult<Vec<_>>>()?;

        kernels.push(KernelMetadata {
            source_name: source_kernel.name.clone(),
            rust_name: spec
                .and_then(|spec| spec.rust_name.clone())
                .unwrap_or_else(|| {
                    format_ident!("{}", sanitize_identifier(&source_kernel.exported_name))
                }),
            params,
            requirements: spec.and_then(|s| s.requirements.as_ref()).map_or_else(
                || quote! { LaunchRequirements::DEFAULT },
                |expr| quote! { #expr },
            ),
        });
    }

    validate_generated_names(&kernels)?;
    Ok(ModuleMetadata { cache_key, kernels })
}

#[expect(
    clippy::too_many_lines,
    reason = "the generated module is a single quoted item tree whose shared names must stay visible"
)]
fn generate_module(
    visibility: &Visibility,
    module: &Ident,
    config: &ModuleConfig,
    metadata: &ModuleMetadata,
) -> TokenStream {
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

    let signatures = metadata.kernels.iter().map(generate_signature);
    let methods = metadata.kernels.iter().map(generate_methods);

    let kernel_names = metadata.kernels.iter().map(|kernel| &kernel.source_name);

    quote! {
        #visibility mod #module {
            #[expect(unused_extern_crates, clippy::useless_attribute, reason = "linked for side effects; required alongside unused_extern_crates")]
            extern crate singe_cuda as _singe_cuda;

            use std::{
                collections::{HashMap, HashSet},
                sync::{Arc, Mutex, OnceLock, Weak},
            };

            use _singe_cuda::{
                architecture::GpuArchitecture,
                context::Context,
                error::Result,
                graph::{ExecutableGraph, Graph, GraphNode},
                kernel::launch::{KernelSignature, LaunchRequirements, ParameterAccess, ParameterLayout, PreparedLaunch},
                module::{KernelParameters, LaunchConfig, LaunchGeometry, Module as CudaModule},
                nvrtc::{CompileOptions, Header, OutputKind, Program, supported_architectures},
                stream::{Stream, StreamCaptureScope},
                view::{DeviceInput, DeviceSlice, DeviceSliceMut, DeviceOutput, DeviceView, DeviceViewMut, DeviceOutputView},
            };

            const MODULE_CACHE_KEY_SUFFIX: &str = #cache_key;

            fn module_cache_key_suffix() -> &'static str {
                MODULE_CACHE_KEY_SUFFIX
            }

            type ModuleCacheKey = (usize, &'static str);
            type ModuleCacheMap = HashMap<ModuleCacheKey, Weak<CompiledModule>>;

            static MODULE_CACHE: OnceLock<Mutex<ModuleCacheMap>> = OnceLock::new();

            fn module_cache() -> &'static Mutex<ModuleCacheMap> {
                MODULE_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
            }

            #[derive(Debug)]
            struct CompiledModule {
                module: CudaModule,
                symbols: HashMap<&'static str, String>,
            }

            fn compile_module(
                context: &Arc<Context>,
                source: &str,
                headers: &[Header<'_>],
                nvrtc_args: &[&str],
                kernel_names: &[&'static str],
            ) -> Result<CompiledModule> {
                let properties = context.device().properties()?;
                let (architecture, output_kind) =
                    supported_gpu_architecture(properties.major, properties.minor);
                let program = Program::from_source(source)
                    .with_name("cuda_module.cu")
                    .with_headers(headers);
                let name_expressions = kernel_names
                    .iter()
                    .map(|name| format!("&{name}"))
                    .collect::<Vec<_>>();
                for name_expression in &name_expressions {
                    program.add_name_expression(name_expression)?;
                }

                let mut options = CompileOptions::new().gpu_architecture(architecture);

                for arg in nvrtc_args {
                    options = options.raw_option(arg);
                }

                program.compile_with_options(&options)?;
                let mut symbols = HashMap::new();
                for (kernel_name, name_expression) in kernel_names.iter().zip(&name_expressions) {
                    symbols.insert(*kernel_name, program.lowered_name(name_expression)?);
                }

                let module = context.load_nvrtc_module(&program, output_kind)?;
                Ok(CompiledModule { module, symbols })
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
                    (10, 7, false) => Some(GpuArchitecture::Compute107),
                    (10, 7, true) => Some(GpuArchitecture::Sm107),
                    (11, 0, false) => Some(GpuArchitecture::Compute110),
                    (11, 0, true) => Some(GpuArchitecture::Sm110),
                    (12, 0, false) => Some(GpuArchitecture::Compute120),
                    (12, 0, true) => Some(GpuArchitecture::Sm120),
                    (12, 1, false) => Some(GpuArchitecture::Compute121),
                    (12, 1, true) => Some(GpuArchitecture::Sm121),
                    _ => None,
                }
            }

            #(#signatures)*

            #[derive(Debug, Clone)]
            pub struct Module {
                module: Arc<CompiledModule>,
            }

            impl Module {
                pub const SOURCE: &'static str = #source;

                pub fn create(context: &Arc<Context>) -> Result<Self> {
                    let headers = [#(#header_literals),*];
                    let nvrtc_args = [#(#nvrtc_args),*];
                    let kernel_names = [#(#kernel_names),*];
                    // SAFETY: the raw context handle is only read to key the per-context module cache.
                    let context_key = unsafe { context.as_raw().addr() };
                    let cache_key = (context_key, module_cache_key_suffix());
                    let cache = module_cache();

                    if let Some(module) = cache.lock().expect("cuda module cache poisoned").get(&cache_key).and_then(Weak::upgrade) {
                        return Ok(Self { module });
                    }

                    let module = Arc::new(compile_module(
                        context,
                        Self::SOURCE,
                        &headers,
                        &nvrtc_args,
                        &kernel_names,
                    )?);
                    let mut cache = cache.lock().expect("cuda module cache poisoned");
                    let module = match cache.get(&cache_key).and_then(Weak::upgrade) {
                        Some(cached) => cached,
                        None => {
                            cache.insert(cache_key, Arc::downgrade(&module));
                            module
                        },
                    };
                    Ok(Self { module })
                }

                pub fn raw(&self) -> &CudaModule {
                    &self.module.module
                }

                fn kernel_symbol(&self, name: &'static str) -> &str {
                    self.module
                        .symbols
                        .get(name)
                        .map(String::as_str)
                        .expect("cuda module kernel symbol missing")
                }

                #(#methods)*
            }
        }
    }
}

fn map_param(
    index: usize,
    kernel_name: &str,
    source_param: &SourceParam,
    spec: Option<&ParameterSpec>,
) -> SynResult<KernelParam> {
    let name = Ident::new(&format!("__singe_arg_{index}"), Span::mixed_site());
    match &source_param.kind {
        SourceParamKind::Scalar(scalar) => {
            if spec.is_some_and(|spec| !matches!(spec.access, ParameterMode::Value)) {
                return Err(SynError::new(
                    name.span(),
                    "by-value parameters require `value(Type)`",
                ));
            }
            let ty = spec
                .and_then(|spec| spec.rust_type.as_ref())
                .map(|ty| quote! { #ty })
                .or_else(|| scalar.map(rust_scalar_type))
                .ok_or_else(|| {
                    to_syn_error(&MacroError::UnsupportedParameter {
                        kernel: kernel_name.to_string(),
                        detail: format!(
                            "{}; supply `value(RustType)` in `parameters`",
                            source_param.declaration
                        ),
                    })
                })?;
            Ok(KernelParam {
                rust_name: name.clone(),
                source_name: source_param.name.clone(),
                rust_type: ty.clone(),
                typed_argument: Some(TypedArgument {
                    rust_type: ty.clone(),
                    value: quote! { #name },
                }),
                layout: quote! { ParameterLayout::value::<#ty>() },
                context_check: None,
                view_type: ty,
                view_binding: quote! { #name },
                raw_value: quote! { #name },
            })
        }
        SourceParamKind::Pointer { levels, element } => {
            map_pointer_param(&name, source_param, levels, *element, spec)
        }
    }
}

fn map_pointer_param(
    name: &Ident,
    source_param: &SourceParam,
    levels: &[bool],
    element: Option<RustScalar>,
    spec: Option<&ParameterSpec>,
) -> SynResult<KernelParam> {
    if spec.is_some_and(|spec| matches!(spec.access, ParameterMode::Value)) {
        return Err(SynError::new(
            name.span(),
            "pointer parameters require `read`, `write` or `read_write`",
        ));
    }
    if spec.is_some() && levels.len() != 1 {
        return Err(SynError::new(
            name.span(),
            "access annotations require a single-level pointer",
        ));
    }
    let element_type = spec
        .and_then(|spec| spec.rust_type.as_ref())
        .map(|ty| quote! { #ty })
        .or_else(|| element.map(rust_scalar_type));
    if spec.is_some() && element_type.is_none() {
        return Err(SynError::new(
            name.span(),
            "access annotations on untyped pointers require an explicit Rust element type",
        ));
    }
    let mut raw = element_type.clone().unwrap_or_else(|| quote! { () });
    for mutable in levels {
        raw = if *mutable {
            quote! { *mut #raw }
        } else {
            quote! { *const #raw }
        };
    }
    let mut param = KernelParam {
        rust_name: name.clone(),
        source_name: source_param.name.clone(),
        rust_type: raw.clone(),
        typed_argument: None,
        layout: quote! { ParameterLayout::pointer(ParameterAccess::OpaquePointer) },
        context_check: None,
        view_type: raw.clone(),
        view_binding: quote! { #name },
        raw_value: quote! { #name },
    };
    if let Some(ty) = element_type.filter(|_| levels.len() == 1) {
        let mode = spec.map_or(
            if levels[0] {
                ParameterMode::ReadWrite
            } else {
                ParameterMode::Read
            },
            |spec| spec.access,
        );
        if !levels[0] && !matches!(mode, ParameterMode::Read) {
            return Err(SynError::new(
                name.span(),
                "a const pointer cannot declare writable access",
            ));
        }
        let mapping = pointer_access_tokens(name, &ty, mode);
        param.view_type = mapping.view;
        param.typed_argument = Some(mapping.argument);
        let access = mapping.access;
        param.layout = quote! { ParameterLayout::pointer(#access) };
        param.context_check = Some(quote! {
            if #name.context() != context {
                return Err(_singe_cuda::error::Error::MemoryContextMismatch);
            }
        });
        if matches!(mode, ParameterMode::Read) {
            param.raw_value = quote! { #name.as_device_ptr() as #raw };
        } else {
            param.view_binding = quote! { mut #name };
            param.raw_value = quote! { #name.as_mut_ptr() };
        }
    }
    Ok(param)
}

struct PointerAccessTokens {
    view: TokenStream,
    argument: TypedArgument,
    access: TokenStream,
}

fn pointer_access_tokens(
    name: &Ident,
    ty: &TokenStream,
    mode: ParameterMode,
) -> PointerAccessTokens {
    let (view, rust_type, value, access) = match mode {
        ParameterMode::Read => (
            quote! { DeviceView<'a, #ty> },
            quote! { &(impl DeviceSlice<#ty> + ?Sized) },
            quote! { DeviceView::from_buffer(#name) },
            quote! { ParameterAccess::Read },
        ),
        ParameterMode::Write => (
            quote! { DeviceOutputView<'a, #ty> },
            quote! { &mut (impl DeviceOutput<#ty> + ?Sized) },
            quote! { DeviceOutputView::from_output(#name) },
            quote! { ParameterAccess::Write },
        ),
        ParameterMode::ReadWrite => (
            quote! { DeviceViewMut<'a, #ty> },
            quote! { &mut (impl DeviceSliceMut<#ty> + ?Sized) },
            quote! { DeviceViewMut::from_buffer(#name) },
            quote! { ParameterAccess::ReadWrite },
        ),
        ParameterMode::Value => {
            unreachable!("pointer value mapping was rejected before token generation")
        }
    };
    PointerAccessTokens {
        view,
        argument: TypedArgument { rust_type, value },
        access,
    }
}

fn signature_name(name: &Ident) -> Ident {
    let span = name.span();
    let name = name
        .to_string()
        .trim_start_matches("r#")
        .split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            chars
                .next()
                .map(|first| first.to_uppercase().to_string() + chars.as_str())
                .unwrap_or_default()
        })
        .collect::<String>();
    format_ident!("{}Signature", name, span = span)
}

fn validate_generated_names(kernels: &[KernelMetadata]) -> SynResult<()> {
    fn register(names: &mut HashMap<String, Span>, name: &Ident) -> SynResult<()> {
        let key = name.to_string().trim_start_matches("r#").to_owned();
        if let Some(previous) = names.insert(key.clone(), name.span()) {
            let mut error = SynError::new(
                name.span(),
                format!(
                    "generated name `{key}` conflicts with another item; use distinct export aliases"
                ),
            );
            error.combine(SynError::new(
                previous,
                "conflicting item is generated here",
            ));
            return Err(error);
        }
        Ok(())
    }

    let mut types = HashMap::new();
    let mut methods = HashMap::new();
    for name in ["create", "raw", "kernel_symbol", "SOURCE"] {
        register(&mut methods, &Ident::new(name, Span::call_site()))?;
    }
    for kernel in kernels {
        register(&mut types, &signature_name(&kernel.rust_name))?;
        register(&mut methods, &format_ident!("prepare_{}", kernel.rust_name))?;
        let has_typed_arguments = kernel.params.iter().all(|p| p.typed_argument.is_some());
        for suffix in ["", "_on", "_record", "_node", "_set_node_params"] {
            register(
                &mut methods,
                &format_ident!("{}{}_raw", kernel.rust_name, suffix),
            )?;
            if has_typed_arguments {
                let name = if suffix.is_empty() {
                    kernel.rust_name.clone()
                } else {
                    format_ident!("{}{}", kernel.rust_name, suffix)
                };
                register(&mut methods, &name)?;
            }
        }
    }
    Ok(())
}

fn generate_signature(kernel: &KernelMetadata) -> TokenStream {
    let signature = signature_name(&kernel.rust_name);
    let types = kernel.params.iter().map(|p| &p.rust_type);
    let names = kernel
        .params
        .iter()
        .map(|p| &p.rust_name)
        .collect::<Vec<_>>();
    let layouts = kernel.params.iter().map(|p| &p.layout);
    let view_types = kernel.params.iter().map(|p| &p.view_type);
    let bindings = kernel.params.iter().map(|p| &p.view_binding);
    let raw_values = kernel.params.iter().map(|p| &p.raw_value);
    let checks = kernel
        .params
        .iter()
        .filter_map(|p| p.context_check.as_ref());
    let requirements = &kernel.requirements;
    quote! {
        /// ABI and launch requirements extracted from this module's CUDA declaration.
        #[derive(Debug)]
        pub struct #signature;

        // SAFETY: argument order and representations come from the CUDA declaration and explicit type mappings.
        // Semantic access annotations are obligations of the unsafe launch caller.
        unsafe impl KernelSignature for #signature {
            type Arguments<'a> = (#(#view_types,)*);
            type RawArguments = (#(#types,)*);
            const PARAMETERS: &'static [ParameterLayout] = &[#(#layouts),*];
            const REQUIREMENTS: LaunchRequirements = #requirements;
            fn raw_arguments(arguments: Self::Arguments<'_>, context: &Arc<Context>) -> Result<Self::RawArguments> {
                let (#(#bindings,)*) = arguments;
                #(#checks)*
                Ok((#(#raw_values,)*))
            }
            fn encode<R>(arguments: Self::RawArguments, f: impl FnOnce(KernelParameters<'_>) -> R) -> R {
                let (#(#names,)*) = arguments;
                let mut params = KernelParameters::new();
                #(params.arg(&#names);)*
                f(params)
            }
        }
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "raw and typed entry points share one quoted launch-family template"
)]
fn generate_methods(kernel: &KernelMetadata) -> TokenStream {
    let signature = signature_name(&kernel.rust_name);
    let prepare_name = format_ident!("prepare_{}", kernel.rust_name);
    let source_name = &kernel.source_name;
    let names = kernel
        .params
        .iter()
        .map(|p| &p.rust_name)
        .collect::<Vec<_>>();
    let typed_arguments = kernel
        .params
        .iter()
        .map(|p| p.typed_argument.as_ref())
        .collect::<Option<Vec<_>>>();
    let parameter_description = format!(
        "CUDA parameters in declaration order: {}.",
        kernel
            .params
            .iter()
            .map(|p| p.source_name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let mut methods = Vec::new();
    for (suffix, extra_args, dispatch, raw_dispatch) in [
        (
            "",
            quote! {},
            quote! { prepared.launch(arguments) },
            quote! { prepared.launch_raw(arguments) },
        ),
        (
            "_on",
            quote! { stream: &Stream, },
            quote! { prepared.launch_on(arguments, stream) },
            quote! { prepared.launch_on_raw(arguments, stream) },
        ),
        (
            "_record",
            quote! { scope: &StreamCaptureScope<'_>, },
            quote! { prepared.record(arguments, scope) },
            quote! { prepared.record_raw(arguments, scope) },
        ),
        (
            "_node",
            quote! { graph: &mut Graph, dependencies: &[GraphNode], },
            quote! { prepared.add_to_graph(arguments, graph, dependencies) },
            quote! { prepared.add_to_graph_raw(arguments, graph, dependencies) },
        ),
        (
            "_set_node_params",
            quote! { executable: &mut ExecutableGraph, node: &GraphNode, },
            quote! { function.set_typed_graph_node_params::<#signature>(executable, node, geometry, arguments) },
            quote! { function.set_typed_graph_node_params_raw::<#signature>(executable, node, geometry, arguments) },
        ),
    ] {
        let raw_name = format_ident!("{}{}_raw", kernel.rust_name, suffix);
        let typed_name = if suffix.is_empty() {
            kernel.rust_name.clone()
        } else {
            format_ident!("{}{}", kernel.rust_name, suffix)
        };
        let raw_args = kernel.params.iter().map(|p| {
            let name = &p.rust_name;
            let ty = &p.rust_type;
            quote! { #name: #ty }
        });
        let description = match suffix {
            "" => format!(
                "Launches `{}` on its context's default stream.",
                kernel.rust_name
            ),
            "_on" => format!("Launches `{}` on `stream`.", kernel.rust_name),
            "_record" => format!(
                "Records `{}` in the active capture and retains its code.",
                kernel.rust_name
            ),
            "_node" => format!(
                "Adds a `{}` kernel node and retains its code.",
                kernel.rust_name
            ),
            "_set_node_params" => format!(
                "Updates a `{}` executable node and retains its code.",
                kernel.rust_name
            ),
            _ => unreachable!("entry points use a fixed suffix list"),
        };
        let is_update = suffix == "_set_node_params";
        let attribute_safety = if is_update {
            quote! { /// Existing executable launch attributes must satisfy this kernel's requirements.
            }
        } else {
            quote! {}
        };
        let setup = if is_update {
            quote! { let function = self.raw().function(self.kernel_symbol(#source_name))?; }
        } else {
            quote! { let prepared = self.#prepare_name(config.clone())?; }
        };
        let configuration = if is_update {
            quote! { geometry: &LaunchGeometry, }
        } else {
            quote! { config: &LaunchConfig, }
        };
        let code_lifetime = if matches!(suffix, "_record" | "_node" | "_set_node_params") {
            quote! { /// This operation retains its code owner; keep the graph or executable alive until queued executions complete.
            }
        } else {
            quote! { /// Keep this module or another code owner alive until execution completes.
            }
        };
        let output = if suffix == "_node" {
            quote! { GraphNode }
        } else {
            quote! { () }
        };
        let (leading, trailing) = if matches!(suffix, "_record" | "_node" | "_set_node_params") {
            (quote! { #extra_args #configuration }, quote! {})
        } else {
            (configuration, extra_args.clone())
        };
        methods.push(quote! {
            #[doc = #description]
            #[doc = #parameter_description]
            ///
            /// # Safety
            /// Values and pointers must satisfy the CUDA kernel's bounds, initialization and aliasing requirements.
            /// Keep storage alive and order conflicting accesses until execution completes, including every graph replay.
            #code_lifetime
            #attribute_safety
            /// Submission does not establish output initialization or completion.
            pub unsafe fn #raw_name(&self, #leading #trailing #(#raw_args),*) -> Result<#output> {
                #setup
                let arguments = (#(#names,)*);
                // SAFETY: the caller upholds this function's safety contract.
                unsafe { #raw_dispatch }
            }
        });
        if let Some(typed_arguments) = &typed_arguments {
            let view_args = kernel
                .params
                .iter()
                .zip(typed_arguments)
                .map(|(p, argument)| {
                    let name = &p.rust_name;
                    let ty = &argument.rust_type;
                    quote! { #name: #ty }
                });
            let values = typed_arguments.iter().map(|argument| &argument.value);
            methods.push(quote! {
                #[doc = #description]
                #[doc = #parameter_description]
                ///
                /// # Safety
                /// Values and views must satisfy the CUDA kernel's bounds and semantic requirements.
                /// Keep storage alive and preserve exclusive writable access until execution completes, including every graph replay.
                #code_lifetime
                #attribute_safety
                /// Writable output views do not certify initialization coverage or completion.
                pub unsafe fn #typed_name(&self, #leading #trailing #(#view_args),*) -> Result<#output> {
                    #setup
                    let arguments = (#(#values,)*);
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe { #dispatch }
                }
            });
        }
    }
    quote! {
        /// Prepares this kernel's signature and execution configuration in the loaded context.
        pub fn #prepare_name(&self, config: LaunchConfig) -> Result<PreparedLaunch<#signature>> {
            let function = self.raw().function(self.kernel_symbol(#source_name))?;
            // SAFETY: this function and signature originate from the same compiled CUDA declaration.
            unsafe { PreparedLaunch::create(&function, config) }
        }
        #(#methods)*
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

fn validate_header_name(include_name: &str) -> SynResult<()> {
    let path = Path::new(include_name);
    if include_name.is_empty() || path.is_absolute() || include_name.contains("..") {
        return Err(to_syn_error(&MacroError::InvalidHeaderName));
    }
    Ok(())
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
    exports: &[ExportSpec],
    headers: &[(String, String)],
    nvrtc_args: &[LitStr],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
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
        let mut parameters = HashMap::new();
        let mut requirements = None;
        if content.peek(token::Brace) {
            let body;
            braced!(body in content);
            while !body.is_empty() {
                let key = body.parse::<Ident>()?;
                body.parse::<Token![:]>()?;
                match key.to_string().as_str() {
                    "requirements" => {
                        if requirements.is_some() {
                            return Err(SynError::new(
                                key.span(),
                                "duplicate `requirements` field",
                            ));
                        }
                        requirements = Some(body.parse::<Expr>()?);
                    }
                    "parameters" => {
                        parse_parameter_specs(&body, &mut parameters)?;
                    }
                    _ => {
                        return Err(SynError::new(
                            key.span(),
                            "expected `requirements` or `parameters`",
                        ));
                    }
                }
                if !body.is_empty() {
                    body.parse::<Token![,]>()?;
                }
            }
        }
        exports.push(ExportSpec {
            source_name,
            rust_name,
            parameters,
            requirements,
        });
        if content.peek(Token![,]) {
            content.parse::<Token![,]>()?;
        }
    }

    Ok(exports)
}

fn parse_parameter_specs(
    input: ParseStream,
    parameters: &mut HashMap<String, ParameterSpec>,
) -> SynResult<()> {
    let args;
    braced!(args in input);
    while !args.is_empty() {
        let name = args.parse::<Ident>()?;
        args.parse::<Token![:]>()?;
        let mode = args.parse::<Ident>()?;
        let access = match mode.to_string().as_str() {
            "value" => ParameterMode::Value,
            "read" => ParameterMode::Read,
            "write" => ParameterMode::Write,
            "read_write" => ParameterMode::ReadWrite,
            _ => {
                return Err(SynError::new(
                    mode.span(),
                    "expected `value`, `read`, `write` or `read_write`",
                ));
            }
        };
        let rust_type = if args.peek(token::Paren) {
            let ty;
            syn::parenthesized!(ty in args);
            let parsed = ty.parse::<Type>()?;
            if !ty.is_empty() {
                return Err(ty.error("expected a single Rust type"));
            }
            Some(parsed)
        } else {
            None
        };
        if matches!(access, ParameterMode::Value) && rust_type.is_none() {
            return Err(SynError::new(
                mode.span(),
                "`value` requires an explicit Rust type",
            ));
        }
        if parameters
            .insert(
                name.to_string().trim_start_matches("r#").to_owned(),
                ParameterSpec { access, rust_type },
            )
            .is_some()
        {
            return Err(SynError::new(name.span(), "duplicate parameter mapping"));
        }
        if !args.is_empty() {
            args.parse::<Token![,]>()?;
        }
    }
    Ok(())
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

    let mut nvrtc_args = None;

    while !content.is_empty() {
        let key = content.parse::<Ident>()?;
        content.parse::<Token![:]>()?;

        match key.to_string().as_str() {
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
        .map_err(|error| to_syn_error(&MacroError::ParseSource(error.to_string())))?;
    let tree = parser
        .parse(source, None)
        .ok_or_else(|| to_syn_error(&MacroError::ParseSource("parser returned no tree".into())))?;

    let aliases = collect_type_aliases(source, tree.root_node());
    let mut kernels = Vec::new();
    collect_source_kernels(source, tree.root_node(), &aliases, &mut kernels);
    Ok(kernels)
}

fn collect_source_kernels(
    source: &str,
    node: Node<'_>,
    aliases: &HashMap<String, RustScalar>,
    kernels: &mut Vec<SourceKernel>,
) {
    if node.kind() == "function_definition"
        && let Some(kernel) = parse_source_kernel(source, node, aliases)
    {
        kernels.push(kernel);
    }

    for index in 0..node.child_count() {
        if let Some(child) = node.child(index) {
            collect_source_kernels(source, child, aliases, kernels);
        }
    }
}

fn parse_source_kernel(
    source: &str,
    node: Node<'_>,
    aliases: &HashMap<String, RustScalar>,
) -> Option<SourceKernel> {
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
        .map(|parameters| parse_source_parameters(source, parameters, aliases))
        .unwrap_or_default();

    Some(SourceKernel {
        exported_name: name.clone(),
        name,
        params,
    })
}

fn function_declarator(node: Node<'_>) -> Option<Node<'_>> {
    node.child_by_field_name("declarator")
        .and_then(|declarator| find_descendant_kind(declarator, "function_declarator"))
}

fn parse_source_parameters(
    source: &str,
    parameters: Node<'_>,
    aliases: &HashMap<String, RustScalar>,
) -> Vec<SourceParam> {
    let mut params = Vec::new();
    for index in 0..parameters.child_count() {
        let Some(child) = parameters.child(index) else {
            continue;
        };
        if child.kind() != "parameter_declaration" {
            continue;
        }
        if let Some(param) = parse_source_parameter(source, child, params.len(), aliases) {
            params.push(param);
        }
    }
    params
}

fn parse_source_parameter(
    source: &str,
    node: Node<'_>,
    index: usize,
    aliases: &HashMap<String, RustScalar>,
) -> Option<SourceParam> {
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
    let scalar = source_scalar_type(declaration, &name, aliases);
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

fn source_scalar_type(
    declaration: &str,
    name: &str,
    aliases: &HashMap<String, RustScalar>,
) -> Option<RustScalar> {
    let tokens = declaration_type_tokens(declaration, name);
    resolve_scalar_tokens(&tokens, aliases)
}

fn resolve_scalar_tokens(
    tokens: &[String],
    aliases: &HashMap<String, RustScalar>,
) -> Option<RustScalar> {
    if tokens.iter().any(|token| token == "void") {
        return None;
    }
    if let Some(scalar) = tokens.iter().find_map(|token| aliases.get(token).copied()) {
        return Some(scalar);
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

    resolve_integer_tokens(tokens)
}

fn resolve_integer_tokens(tokens: &[String]) -> Option<RustScalar> {
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

fn collect_type_aliases(source: &str, root: Node<'_>) -> HashMap<String, RustScalar> {
    let mut aliases = HashMap::new();
    collect_typedef_aliases(source, root, &mut aliases);
    collect_using_aliases(source, &mut aliases);
    aliases
}

fn collect_typedef_aliases(
    source: &str,
    node: Node<'_>,
    aliases: &mut HashMap<String, RustScalar>,
) {
    if node.kind() == "type_definition" {
        collect_typedef_alias(source, node, aliases);
    }

    for index in 0..node.child_count() {
        if let Some(child) = node.child(index) {
            collect_typedef_aliases(source, child, aliases);
        }
    }
}

fn collect_typedef_alias(source: &str, node: Node<'_>, aliases: &mut HashMap<String, RustScalar>) {
    let Some(type_node) = node.child_by_field_name("type") else {
        return;
    };
    let base = node_text(source, type_node).trim();
    let scalar = source_scalar_type(base, "", aliases);
    for index in 0..node.child_count() {
        let Some(child) = node.child(index) else {
            continue;
        };
        if child.kind() == "type_identifier"
            && let Some(scalar) = scalar
        {
            aliases.insert(node_text(source, child).trim().to_ascii_lowercase(), scalar);
        }
    }
}

fn collect_using_aliases(source: &str, aliases: &mut HashMap<String, RustScalar>) {
    for statement in source.split(';') {
        let statement = statement.trim();
        let Some(rest) = statement.strip_prefix("using ") else {
            continue;
        };
        let Some((alias, ty)) = rest.split_once('=') else {
            continue;
        };
        let alias = alias.trim();
        if alias.is_empty() || !alias.chars().all(is_c_identifier_char) {
            continue;
        }
        let ty = ty.trim();
        if ty.contains('*') || ty.contains('&') {
            continue;
        }
        let Some(scalar) = source_scalar_type(ty, "", aliases) else {
            continue;
        };
        aliases.insert(alias.to_ascii_lowercase(), scalar);
    }
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
        tokens.push(mem::take(current));
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
        if let Some(child) = node.child(index)
            && let Some(found) = find_descendant_kind(child, kind)
        {
            return Some(found);
        }
    }
    None
}

fn declarator_name(node: Node<'_>) -> Option<Node<'_>> {
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
        if let Some(child) = node.child(index)
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
        if let Some(child) = node.child(index)
            && declarator_has_kind(child, kind)
        {
            return true;
        }
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

struct SelectedKernel<'a> {
    kernel: &'a SourceKernel,
    spec: Option<&'a ExportSpec>,
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

fn to_syn_error(error: &MacroError) -> SynError {
    SynError::new(Span::call_site(), error.to_string())
}

#[cfg(test)]
mod tests {
    use syn::parse::Parser as _;

    use super::*;

    #[test]
    fn test_export_parameter_blocks_accumulate_mappings() {
        let exports = parse_export_specs
            .parse2(quote! {{
                scale {
                    parameters: { input: read(f32) },
                    parameters: { output: write(f32) },
                }
            }})
            .unwrap();
        let parameters = &exports[0].parameters;
        assert_eq!(parameters.len(), 2);
        assert!(matches!(parameters["input"].access, ParameterMode::Read));
        assert!(matches!(parameters["output"].access, ParameterMode::Write));
    }

    #[test]
    fn test_export_parameter_blocks_reject_duplicate_mappings() {
        let Err(error) = parse_export_specs.parse2(quote! {{
            scale {
                parameters: { input: read(f32) },
                parameters: { input: write(f32) },
            }
        }}) else {
            panic!("duplicate parameter mapping was accepted");
        };
        assert_eq!(error.to_string(), "duplicate parameter mapping");
    }

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

    #[test]
    fn test_resolve_scalar_alias_parameters() {
        let source = r#"
            using index_t = int;
            using count_t = index_t;
            typedef unsigned int flags_t;
            typedef unsigned long long seed_t;

            extern "C" __global__ void aliases(
                index_t index,
                count_t count,
                flags_t flags,
                seed_t seed
            ) {}
        "#;
        let kernels = parse_source_kernels(source).unwrap();
        let params = &kernels[0].params;
        assert_scalar(&params[0], Some(RustScalar::I32));
        assert_scalar(&params[1], Some(RustScalar::I32));
        assert_scalar(&params[2], Some(RustScalar::U32));
        assert_scalar(&params[3], Some(RustScalar::U64));
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
