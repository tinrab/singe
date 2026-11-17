// Parsing of the `cuda_module!` input: the module declaration and the fields of its body.

use std::collections::HashMap;

use syn::{
    Error, Expr, Ident, LitStr, Path, Result, Token, Type, Visibility, braced, bracketed,
    parenthesized,
    parse::{Parse, ParseStream},
    parse_quote, token,
};

pub(crate) struct CudaModuleInput {
    pub(crate) visibility: Visibility,
    pub(crate) module: Ident,
    pub(crate) config: ModuleConfig,
}

impl Parse for CudaModuleInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let visibility = input.parse::<Visibility>()?;
        input.parse::<Token![mod]>()?;
        let module = input.parse::<Ident>()?;

        let content;
        braced!(content in input);

        Ok(Self {
            visibility,
            config: ModuleConfig::parse_body(&content, &module)?,
            module,
        })
    }
}

pub(crate) struct ModuleConfig {
    pub(crate) source: LitStr,
    pub(crate) exports: Vec<ExportSpec>,
    pub(crate) headers: Vec<HeaderSpec>,
    pub(crate) nvrtc_args: Vec<LitStr>,
    // The path through which the generated code names `singe_cuda`.
    pub(crate) krate: Path,
}

impl ModuleConfig {
    fn parse_body(input: ParseStream, module: &Ident) -> Result<Self> {
        let mut source = None;
        let mut exports = Vec::new();
        let mut headers = Vec::new();
        let mut nvrtc_args = Vec::new();
        let mut compile = false;
        let mut krate = None;

        while !input.is_empty() {
            if input.peek(Token![crate]) {
                let key = input.parse::<Token![crate]>()?;
                input.parse::<Token![:]>()?;
                if krate.is_some() {
                    return Err(Error::new(key.span, "duplicate `crate` field"));
                }

                krate = Some(input.parse::<Path>()?);
            } else {
                let key = input.parse::<Ident>()?;
                input.parse::<Token![:]>()?;

                match key.to_string().as_str() {
                    "source" => {
                        if source.is_some() {
                            return Err(Error::new(key.span(), "duplicate `source` field"));
                        }

                        source = Some(input.parse::<LitStr>()?);
                    }
                    "exports" => {
                        if !exports.is_empty() {
                            return Err(Error::new(key.span(), "duplicate `exports` field"));
                        }

                        exports = parse_export_specs(input)?;
                    }
                    "headers" => {
                        if !headers.is_empty() {
                            return Err(Error::new(key.span(), "duplicate `headers` field"));
                        }

                        headers = parse_header_specs(input)?;
                    }
                    "compile" => {
                        if compile {
                            return Err(Error::new(key.span(), "duplicate `compile` field"));
                        }

                        compile = true;
                        nvrtc_args = parse_compile_config(input)?;
                    }
                    _ => return Err(Error::new(key.span(), format!("unknown field `{key}`"))),
                }
            }

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        let Some(source) = source else {
            return Err(Error::new(
                module.span(),
                "`cuda_module!` requires a `source` field",
            ));
        };

        Ok(Self {
            source,
            exports,
            headers,
            nvrtc_args,
            krate: krate.unwrap_or_else(|| parse_quote!(::singe_cuda)),
        })
    }
}

pub(crate) struct ExportSpec {
    pub(crate) source_name: Ident,
    pub(crate) rust_name: Option<Ident>,
    pub(crate) parameters: HashMap<String, ParameterSpec>,
    pub(crate) requirements: Option<Expr>,
}

pub(crate) struct ParameterSpec {
    pub(crate) name: Ident,
    pub(crate) access: ParameterMode,
    pub(crate) rust_type: Option<Type>,
}

#[derive(Clone, Copy)]
pub(crate) enum ParameterMode {
    Value,
    Read,
    Write,
    ReadWrite,
}

pub(crate) struct HeaderSpec {
    pub(crate) include_name: LitStr,
    pub(crate) source: LitStr,
}

pub(crate) fn parse_export_specs(input: ParseStream) -> Result<Vec<ExportSpec>> {
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
                            return Err(Error::new(key.span(), "duplicate `requirements` field"));
                        }

                        requirements = Some(body.parse::<Expr>()?);
                    }
                    "parameters" => {
                        parse_parameter_specs(&body, &mut parameters)?;
                    }
                    _ => {
                        return Err(Error::new(
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
) -> Result<()> {
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
                return Err(Error::new(
                    mode.span(),
                    "expected `value`, `read`, `write` or `read_write`",
                ));
            }
        };
        let rust_type = if args.peek(token::Paren) {
            let ty;
            parenthesized!(ty in args);
            let parsed = ty.parse::<Type>()?;
            if !ty.is_empty() {
                return Err(ty.error("expected a single Rust type"));
            }
            Some(parsed)
        } else {
            None
        };
        if matches!(access, ParameterMode::Value) && rust_type.is_none() {
            return Err(Error::new(
                mode.span(),
                "`value` requires an explicit Rust type",
            ));
        }

        let key = name.to_string().trim_start_matches("r#").to_owned();
        if parameters.contains_key(&key) {
            return Err(Error::new(name.span(), "duplicate parameter mapping"));
        }

        parameters.insert(
            key,
            ParameterSpec {
                name,
                access,
                rust_type,
            },
        );
        if !args.is_empty() {
            args.parse::<Token![,]>()?;
        }
    }
    Ok(())
}

fn parse_header_specs(input: ParseStream) -> Result<Vec<HeaderSpec>> {
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

// Parses `compile: { nvrtc_args: [..] }`, returning the NVRTC arguments.
fn parse_compile_config(input: ParseStream) -> Result<Vec<LitStr>> {
    let content;
    braced!(content in input);

    let mut nvrtc_args = None;

    while !content.is_empty() {
        let key = content.parse::<Ident>()?;
        content.parse::<Token![:]>()?;

        match key.to_string().as_str() {
            "nvrtc_args" => {
                if nvrtc_args.is_some() {
                    return Err(Error::new(key.span(), "duplicate `nvrtc_args` field"));
                }

                nvrtc_args = Some(parse_string_list(&content)?);
            }
            _ => {
                return Err(Error::new(
                    key.span(),
                    format!("unknown compile field `{key}`"),
                ));
            }
        }

        if content.peek(Token![,]) {
            content.parse::<Token![,]>()?;
        }
    }

    Ok(nvrtc_args.unwrap_or_default())
}

fn parse_string_list(input: ParseStream) -> Result<Vec<LitStr>> {
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

#[cfg(test)]
mod tests {
    use quote::quote;
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
}
