// The Rust form of each kernel parameter: its method argument, its view in the signature, and its raw value.

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{Error, Ident, Result};

use super::{
    input::{ParameterMode, ParameterSpec},
    source::{RustScalar, SourceParam, SourceParamKind},
};

pub(crate) struct KernelParam {
    pub(crate) rust_name: Ident,
    pub(crate) source_name: String,
    // The CUDA parameter value's type, such as `*const f32`.
    pub(crate) raw_type: TokenStream,
    // The generated method's argument type, and the expression that turns the argument into its view.
    pub(crate) argument_type: TokenStream,
    pub(crate) argument_value: TokenStream,
    pub(crate) layout: TokenStream,
    pub(crate) context_check: Option<TokenStream>,
    pub(crate) view_type: TokenStream,
    pub(crate) view_binding: TokenStream,
    pub(crate) raw_value: TokenStream,
}

// Maps the parameter at `index`, reporting a type the source does not determine at `span`.
pub(crate) fn map_param(
    index: usize,
    kernel_name: &str,
    source_param: &SourceParam,
    spec: Option<&ParameterSpec>,
    span: Span,
) -> Result<KernelParam> {
    let name = Ident::new(&format!("__singe_arg_{index}"), Span::mixed_site());
    match &source_param.kind {
        SourceParamKind::Scalar(scalar) => {
            if let Some(spec) = spec
                && !matches!(spec.access, ParameterMode::Value)
            {
                return Err(Error::new(
                    spec.name.span(),
                    "by-value parameters require `value(Type)`",
                ));
            }

            let ty = spec
                .and_then(|spec| spec.rust_type.as_ref())
                .map(|ty| quote! { #ty })
                .or_else(|| scalar.map(rust_scalar_type))
                .ok_or_else(|| {
                    Error::new(
                        span,
                        format!(
                            "unsupported source parameter type for `{kernel_name}`: {}; supply `value(RustType)` in `parameters`",
                            source_param.declaration
                        ),
                    )
                })?;
            Ok(KernelParam {
                rust_name: name.clone(),
                source_name: source_param.name.clone(),
                raw_type: ty.clone(),
                argument_type: ty.clone(),
                argument_value: quote! { #name },
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
) -> Result<KernelParam> {
    if let Some(spec) = spec {
        if matches!(spec.access, ParameterMode::Value) {
            return Err(Error::new(
                spec.name.span(),
                "pointer parameters require `read`, `write` or `read_write`",
            ));
        }

        if levels.len() != 1 {
            return Err(Error::new(
                spec.name.span(),
                "access annotations require a single-level pointer",
            ));
        }
    }

    let element_type = spec
        .and_then(|spec| spec.rust_type.as_ref())
        .map(|ty| quote! { #ty })
        .or_else(|| element.map(rust_scalar_type));
    if let Some(spec) = spec
        && element_type.is_none()
    {
        return Err(Error::new(
            spec.name.span(),
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
        raw_type: raw.clone(),
        argument_type: raw.clone(),
        argument_value: quote! { #name },
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
            let span = spec.map_or_else(Span::call_site, |spec| spec.name.span());
            return Err(Error::new(
                span,
                "a const pointer cannot declare writable access",
            ));
        }

        let PointerAccess {
            view,
            argument_type,
            argument_value,
            access,
        } = pointer_access(name, &ty, mode);
        param.view_type = view;
        param.argument_type = argument_type;
        param.argument_value = argument_value;
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

struct PointerAccess {
    view: TokenStream,
    argument_type: TokenStream,
    argument_value: TokenStream,
    access: TokenStream,
}

fn pointer_access(name: &Ident, ty: &TokenStream, mode: ParameterMode) -> PointerAccess {
    let (view, argument_type, argument_value, access) = match mode {
        ParameterMode::Read | ParameterMode::Value => (
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
    };
    PointerAccess {
        view,
        argument_type,
        argument_value,
        access,
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
