use proc_macro2::TokenStream;

use quote::{quote, quote_spanned};

use syn::{Data, DeriveInput, Error, Path, Result, parse_quote, spanned::Spanned};

#[derive(Clone, Copy)]
pub enum Representation {
    Transfer,
    NoUninit,
    Zeroable,
    AnyBitPattern,
}

pub fn expand(input: TokenStream, representation: Representation) -> Result<TokenStream> {
    let input: DeriveInput = syn::parse2(input)?;
    let mut cuda: Path = parse_quote!(::singe_cuda);
    let mut stable_layout = false;
    for attribute in &input.attrs {
        if attribute.path().is_ident("device_repr") {
            attribute.parse_nested_meta(|meta| {
                if !meta.path.is_ident("crate") {
                    return Err(meta.error("expected `crate = path`"));
                }
                cuda = meta.value()?.parse()?;
                Ok(())
            })?;
        } else if attribute.path().is_ident("repr") {
            attribute.parse_nested_meta(|meta| {
                if meta.path.is_ident("C") || meta.path.is_ident("transparent") {
                    stable_layout = true;
                    Ok(())
                } else if meta.path.is_ident("align") {
                    let content;
                    syn::parenthesized!(content in meta.input);
                    let _: syn::LitInt = content.parse()?;
                    Ok(())
                } else {
                    Err(meta.error("device representations require `repr(C)` or `repr(transparent)` without packing"))
                }
            })?;
        }
    }
    if !stable_layout {
        return Err(Error::new_spanned(
            &input.ident,
            "device representations require `#[repr(C)]` or `#[repr(transparent)]`",
        ));
    }
    let Data::Struct(data) = &input.data else {
        return Err(Error::new_spanned(
            &input.ident,
            "device representation derives support structs only; enums and unions require a manual unsafe implementation",
        ));
    };
    let (trait_name, parent): (Path, Path) = match representation {
        Representation::Transfer => (
            parse_quote!(#cuda::view::DeviceRepr),
            parse_quote!(::core::marker::Copy),
        ),
        Representation::Zeroable => (
            parse_quote!(#cuda::view::ZeroableDeviceRepr),
            parse_quote!(#cuda::view::DeviceRepr),
        ),
        Representation::NoUninit => (
            parse_quote!(#cuda::view::NoUninitDeviceRepr),
            parse_quote!(#cuda::view::DeviceRepr),
        ),
        Representation::AnyBitPattern => (
            parse_quote!(#cuda::view::AnyBitPatternDeviceRepr),
            parse_quote!(#cuda::view::ZeroableDeviceRepr),
        ),
    };
    let name = &input.ident;
    let field_types: Vec<_> = data.fields.iter().map(|field| &field.ty).collect();
    let mut generics = input.generics.clone();
    let predicates = &mut generics.make_where_clause().predicates;
    predicates.push(parse_quote!(Self: #parent + 'static));
    for field in &field_types {
        predicates.push(parse_quote!(#field: #trait_name));
    }
    let (impl_generics, type_generics, where_clause) = generics.split_for_impl();
    let body = if matches!(representation, Representation::Transfer) {
        let field_checks = field_types.iter().map(|ty| {
            quote_spanned! {ty.span()=> let () = <#ty as #cuda::view::DeviceRepr>::LAYOUT; }
        });
        Some(quote! {
            const LAYOUT: () = {
                #(#field_checks)*
                assert!(
                    ::core::mem::size_of::<Self>() == 0 #( + ::core::mem::size_of::<#field_types>())*,
                    "device representations cannot contain implicit padding; add explicit byte fields"
                );
            };
        })
    } else {
        None
    };
    let layout_check =
        if input.generics.params.is_empty() && matches!(representation, Representation::Transfer) {
            Some(quote! {
                const _: () = <#name as #cuda::view::DeviceRepr>::LAYOUT;
            })
        } else {
            None
        };
    Ok(quote! {
        unsafe impl #impl_generics #trait_name for #name #type_generics #where_clause {
            #body
        }
        #layout_check
    })
}
