use proc_macro2::{TokenStream, TokenTree};

use quote::{ToTokens, quote, quote_spanned};

use syn::{
    Error as SynError, Expr, Ident, Result as SynResult, Token,
    parse::{Parse, ParseStream, Parser},
    punctuated::Punctuated,
    spanned::Spanned,
};

struct TensorElements {
    elements: Vec<Expr>,
    shape: Vec<u32>,
}

impl TensorElements {
    fn new() -> Self {
        Self {
            elements: Vec::new(),
            shape: Vec::new(),
        }
    }
}

fn parse_recursive(expr: &Expr, data: &mut TensorElements, depth: usize) -> SynResult<()> {
    if let Expr::Array(array_expr) = expr {
        let current_len = u32::try_from(array_expr.elems.len())
            .map_err(|_| SynError::new_spanned(array_expr, "tensor dimension exceeds u32"))?;

        if depth >= data.shape.len() {
            data.shape.push(current_len);
        } else if data.shape[depth] == 0 && current_len > 0 {
            data.shape[depth] = current_len;
        } else if current_len != data.shape[depth] && data.shape[depth] != 0 {
            return Err(SynError::new(
                array_expr.span(),
                format!(
                    "jagged array: dimension {} has conflicting lengths (expected {}, found {})",
                    depth, data.shape[depth], current_len
                ),
            ));
        }

        if current_len == 0 {
            let next_depth = depth + 1;
            if next_depth >= data.shape.len() {
                data.shape.push(0);
            }
        }

        for elem in &array_expr.elems {
            parse_recursive(elem, data, depth + 1)?;
        }
    } else {
        data.elements.push(expr.clone());
    }
    Ok(())
}

struct TensorMacroInput {
    elements_expr: Expr,
    device_expr: Option<Expr>,
    scalar: bool,
}

impl Parse for TensorMacroInput {
    fn parse(input: ParseStream) -> SynResult<Self> {
        let elements_expr: Expr = input.parse()?;
        let mut scalar = true;
        let mut device_expr: Option<Expr> = None;

        while !input.is_empty() {
            input.parse::<Token![,]>()?;
            scalar = false;
            if input.is_empty() {
                break;
            }

            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            let value: Expr = input.parse()?;

            match key.to_string().as_str() {
                "device" => {
                    if device_expr.is_some() {
                        return Err(SynError::new(key.span(), "duplicate `device` argument"));
                    }

                    device_expr = Some(value);
                }
                _ => {
                    return Err(SynError::new(
                        key.span(),
                        format!("unknown argument `{key}`. supported arguments: `device`"),
                    ));
                }
            }
        }

        Ok(Self {
            elements_expr,
            device_expr,
            scalar,
        })
    }
}

enum MacroInputType {
    Full(Box<TensorMacroInput>),
    ElementsExpr(Box<Expr>),
    FlatList(Punctuated<Expr, Token![,]>),
}

fn parse_macro_input(input: TokenStream) -> SynResult<MacroInputType> {
    let parsed_input = if let Ok(inp) = syn::parse2::<TensorMacroInput>(input.clone()) {
        match inp.elements_expr {
            Expr::Array(_)
            | Expr::Tuple(_)
            | Expr::Call(_)
            | Expr::MethodCall(_)
            | Expr::Path(_)
            | Expr::Paren(_) => MacroInputType::Full(Box::new(inp)),
            _ if !inp
                .elements_expr
                .to_token_stream()
                .to_string()
                .contains(',') =>
            {
                MacroInputType::Full(Box::new(inp))
            }
            _ => match syn::parse2::<Expr>(input) {
                Ok(expr) => MacroInputType::ElementsExpr(Box::new(expr)),
                Err(error) => {
                    return Err(SynError::new(
                        error.span(),
                        format!(
                            "failed to parse tensor! input as a tensor expression. expected format like `tensor!([1,2])`, `tensor!(data, device=...)`, or `tensor![1, 2, 3]`. error: {error}"
                        ),
                    ));
                }
            },
        }
    } else if input
        .clone()
        .into_iter()
        .any(|tt| matches!(tt, TokenTree::Punct(p) if p.as_char() == ','))
    {
        match Punctuated::<Expr, Token![,]>::parse_terminated.parse2(input.clone()) {
            Ok(punctuated) if !punctuated.is_empty() => MacroInputType::FlatList(punctuated),
            _ => match syn::parse2::<Expr>(input) {
                Ok(expr) => MacroInputType::ElementsExpr(Box::new(expr)),
                Err(error) => {
                    return Err(SynError::new(
                        error.span(),
                        format!(
                            "failed to parse tensor! input after attempting flat list. expected single expression like `[1,2]`. error: {error}"
                        ),
                    ));
                }
            },
        }
    } else {
        match syn::parse2::<Expr>(input) {
            Ok(expr) => MacroInputType::ElementsExpr(Box::new(expr)),
            Err(error) => {
                return Err(SynError::new(
                    error.span(),
                    format!(
                        "failed to parse tensor! input. expected format like `tensor!([1,2])`, `tensor!(data, device=...)`, or `tensor![1, 2, 3]`. error: {error}"
                    ),
                ));
            }
        }
    };

    Ok(parsed_input)
}

pub fn expand(input: TokenStream) -> SynResult<TokenStream> {
    let (elements_expr, device_expr_opt, is_scalar) = match parse_macro_input(input)? {
        MacroInputType::Full(inp) => (inp.elements_expr, inp.device_expr, inp.scalar),
        MacroInputType::ElementsExpr(expr) => (*expr, None, false),
        MacroInputType::FlatList(punctuated) => {
            let list_span = punctuated.span();
            let array_tokens = quote_spanned!(list_span=> [ #punctuated ] );
            match syn::parse2::<Expr>(array_tokens) {
                Ok(expr_array) => (expr_array, None, false),
                Err(error) => {
                    return Err(SynError::new(
                        list_span,
                        format!("failed to re-parse generated array from flat list: {error}"),
                    ));
                }
            }
        }
    };

    let original_elements_expr = elements_expr.clone();

    let mut tensor_data = TensorElements::new();

    let is_empty_top_array = match &elements_expr {
        Expr::Array(a) => a.elems.is_empty(),
        _ => false,
    };

    if is_empty_top_array {
        tensor_data.shape = vec![0];
    } else {
        parse_recursive(&elements_expr, &mut tensor_data, 0)?;

        if tensor_data.elements.is_empty() && tensor_data.shape.is_empty() && !is_empty_top_array {
            if !elements_expr.to_token_stream().is_empty() {
                return Err(SynError::new_spanned(
                    elements_expr,
                    "parsed elements expression resulted in no data and no shape",
                ));
            }

            tensor_data.shape = vec![0];
        }
    }

    if !is_scalar && tensor_data.shape.is_empty() {
        tensor_data.shape = vec![u32::try_from(tensor_data.elements.len()).map_err(|_| {
            SynError::new_spanned(&original_elements_expr, "tensor element count exceeds u32")
        })?];
    }

    let element_tokens = &tensor_data.elements;
    let shape_tokens = &tensor_data.shape;

    let device_token_stream = device_expr_opt.map_or_else(
        || quote! { ::singe::Device::Cpu },
        |device| quote! { #device },
    );

    Ok(quote! {
        {
            let data_vec: Vec<_> = vec![#(#element_tokens),*];
            let shape = match ::singe::Shape::contiguous([#(#shape_tokens),*]) {
                Ok(shape) => shape,
                Err(error) => {
                    panic!(
                        "failed to create shape in tensor! macro. error: {:?}. input expression structure: {}",
                        error,
                        stringify!(#original_elements_expr)
                    );
                }
            };
            let target_device: ::singe::Device = #device_token_stream;

            fn create_tensor<T: ::singe::TensorDtype>(
                device: ::singe::Device,
                data: Vec<T>,
                shape: &::singe::Shape,
                expr_str: &str,
            ) -> ::singe::Tensor {
                match ::singe::Tensor::from_vec(device, data, shape) {
                    Ok(tensor) => tensor,
                    Err(error) => {
                        panic!(
                            "failed to create tensor in tensor! macro. error: {:?}. input expression structure: {}",
                            error,
                            expr_str
                        );
                    }
                }
            }

            create_tensor(target_device, data_vec, &shape, stringify!(#original_elements_expr))
        }
    })
}
