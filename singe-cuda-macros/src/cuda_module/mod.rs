// `cuda_module!`: parses the input, reads the kernels from the CUDA source, and emits a module of data and launch wrappers.
// The runtime work, compiling and caching the module per context, is `singe_cuda::nvrtc::compile_cached`.

mod generate;
mod input;
mod name;
mod parameter;
mod source;

use std::{collections::HashSet, path::Path};

use proc_macro2::{Span, TokenStream};
use quote::quote;
use sha2::{Digest, Sha256};
use syn::{Error, Ident, LitStr, Result};

use self::{
    generate::generate_module,
    input::{CudaModuleInput, ExportSpec, ModuleConfig},
    name::{kernel_identifier, validate_generated_names},
    parameter::{KernelParam, map_param},
    source::{SourceKernel, parse_source_kernels},
};

pub(crate) fn expand_module(input: TokenStream) -> Result<TokenStream> {
    let input = syn::parse2::<CudaModuleInput>(input)?;
    let metadata = extract_metadata(&input.config)?;
    Ok(generate_module(
        &input.visibility,
        &input.module,
        &input.config,
        &metadata,
    ))
}

pub(crate) struct ModuleMetadata {
    pub(crate) cache_key: String,
    pub(crate) kernels: Vec<KernelMetadata>,
}

pub(crate) struct KernelMetadata {
    pub(crate) source_name: String,
    pub(crate) rust_name: Ident,
    pub(crate) params: Vec<KernelParam>,
    pub(crate) requirements: TokenStream,
}

fn extract_metadata(config: &ModuleConfig) -> Result<ModuleMetadata> {
    for header in &config.headers {
        validate_header_name(&header.include_name)?;
    }

    let source = config.source.value();
    let source_span = config.source.span();
    let source_kernels = parse_source_kernels(&source, source_span)?;
    if source_kernels.is_empty() {
        return Err(Error::new(
            source_span,
            "no `__global__` kernels were found in the source",
        ));
    }

    let mut source_names = HashSet::new();
    for kernel in &source_kernels {
        if !source_names.insert(kernel.name.as_str()) {
            return Err(Error::new(
                source_span,
                format!(
                    "the source defines more than one kernel named `{}`",
                    kernel.name
                ),
            ));
        }
    }

    let selected = if config.exports.is_empty() {
        source_kernels.iter().map(|kernel| (kernel, None)).collect()
    } else {
        config
            .exports
            .iter()
            .map(|export| Ok((find_export(&source_kernels, export)?, Some(export))))
            .collect::<Result<Vec<_>>>()?
    };

    let kernels = selected
        .into_iter()
        .map(|(kernel, export)| kernel_metadata(kernel, export, source_span))
        .collect::<Result<Vec<_>>>()?;
    validate_generated_names(&kernels)?;
    Ok(ModuleMetadata {
        cache_key: cache_key(config),
        kernels,
    })
}

fn find_export<'a>(kernels: &'a [SourceKernel], export: &ExportSpec) -> Result<&'a SourceKernel> {
    let name = export.source_name.to_string();
    let name = name.trim_start_matches("r#");
    kernels
        .iter()
        .find(|kernel| kernel.name == name)
        .ok_or_else(|| {
            let available = kernels
                .iter()
                .map(|kernel| kernel.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            Error::new(
                export.source_name.span(),
                format!("the source has no kernel `{name}`; its kernels are: {available}"),
            )
        })
}

// Maps one kernel and its optional export, reporting problems at the export when there is one and at the source otherwise.
fn kernel_metadata(
    kernel: &SourceKernel,
    export: Option<&ExportSpec>,
    source_span: Span,
) -> Result<KernelMetadata> {
    let span = export.map_or(source_span, |export| export.source_name.span());
    if let Some(export) = export {
        for (key, spec) in &export.parameters {
            if !kernel.params.iter().any(|param| param.name == *key) {
                return Err(Error::new(
                    spec.name.span(),
                    format!("kernel `{}` has no parameter `{}`", kernel.name, spec.name),
                ));
            }
        }
    }

    let params = kernel
        .params
        .iter()
        .enumerate()
        .map(|(index, param)| {
            let spec = export.and_then(|export| export.parameters.get(&param.name));
            map_param(index, &kernel.name, param, spec, span)
        })
        .collect::<Result<Vec<_>>>()?;
    let rust_name = match export.and_then(|export| export.rust_name.clone()) {
        Some(rust_name) => rust_name,
        None => kernel_identifier(&kernel.name, span)?,
    };
    let requirements = export
        .and_then(|export| export.requirements.as_ref())
        .map_or_else(
            || quote! { LaunchRequirements::DEFAULT },
            |expr| quote! { #expr },
        );
    Ok(KernelMetadata {
        source_name: kernel.name.clone(),
        rust_name,
        params,
        requirements,
    })
}

fn validate_header_name(include_name: &LitStr) -> Result<()> {
    let value = include_name.value();
    if value.is_empty() || Path::new(&value).is_absolute() || value.contains("..") {
        return Err(Error::new(
            include_name.span(),
            "a header include name must be a relative path without `..`",
        ));
    }
    Ok(())
}

// Hashes everything that determines the compiled module, so modules with equal keys are interchangeable.
fn cache_key(config: &ModuleConfig) -> String {
    let mut hasher = Sha256::new();
    hasher.update(config.source.value().as_bytes());
    for export in &config.exports {
        hasher.update(export.source_name.to_string().as_bytes());
        hasher.update([0]);
        if let Some(rust_name) = &export.rust_name {
            hasher.update(rust_name.to_string().as_bytes());
        }

        hasher.update([0]);
    }

    for header in &config.headers {
        hasher.update(header.include_name.value().as_bytes());
        hasher.update([0]);
        hasher.update(header.source.value().as_bytes());
        hasher.update([0]);
    }

    for arg in &config.nvrtc_args {
        hasher.update(arg.value().as_bytes());
        hasher.update([0]);
    }
    hex::encode(hasher.finalize())
}
