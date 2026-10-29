mod common;
mod depth;
mod directive;
mod error;
mod function;
mod instruction;
mod module;
mod operand;

use nom::Err as NomErr;

pub use crate::parser::error::{ParseError, ParseErrorKind};

use crate::{
    ast::{Module, Span},
    parser::common::{Input, SyntaxErrorKind},
};

/// Parses a complete PTX module into an owned syntax tree.
///
/// The module must start with `.version` and `.target`. Parsing checks syntax only;
/// [`ParsedModule::compile`](crate::emulator::ParsedModule::compile) checks whether a kernel
/// can execute. Block nesting and expression depth are limited to 128.
///
/// # Errors
///
/// Returns [`ParseErrorKind::TrailingInput`] if non-whitespace text follows the module,
/// [`ParseErrorKind::DepthLimitExceeded`] for input nested too deeply, and
/// [`ParseErrorKind::UnexpectedToken`] or [`ParseErrorKind::UnexpectedEof`] for other
/// syntax errors.
pub fn parse_module(input: &str) -> Result<Module, ParseError> {
    match module::module(Input::new(input)) {
        Ok((remaining, module)) => {
            let remaining = remaining.fragment().trim_start();
            if remaining.is_empty() {
                Ok(module)
            } else {
                Err(ParseError {
                    kind: ParseErrorKind::TrailingInput,
                    span: Span::new(input.len() - remaining.len(), input.len()),
                })
            }
        }
        Err(error) => {
            let offset = match &error {
                NomErr::Error(inner) | NomErr::Failure(inner) => inner.input.location_offset(),
                NomErr::Incomplete(_) => input.len(),
            };
            Err(ParseError {
                kind: if matches!(&error, NomErr::Failure(inner) if inner.kind == SyntaxErrorKind::DepthLimitExceeded)
                {
                    ParseErrorKind::DepthLimitExceeded
                } else if offset == input.len() {
                    ParseErrorKind::UnexpectedEof
                } else {
                    ParseErrorKind::UnexpectedToken(input[offset..].chars().take(20).collect())
                },
                span: Span::new(
                    offset,
                    offset + input[offset..].chars().next().map_or(0, char::len_utf8),
                ),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{
        AddressSize, Attribute, DirectiveStatement, FunctionDirective, Immediate, Initializer,
        InitializerBinaryOp, InitializerCastType, InitializerUnaryOp, LocFunctionName, Operand,
        SectionDataWidth, SectionLine, SectionValue, Statement, TopLevelItem,
    };

    #[test]
    fn test_directive_keywords_require_token_boundaries() {
        for source in [
            ".version9.4 .target sm_80",
            ".version 9.4 .targetsm_80",
            ".version 9.4 .target sm_80 .address_size64",
            ".version 9.4 .target sm_80 .global .u32value;",
            ".version 9.4 .target sm_80 .entrykernel() {ret;}",
            ".version 9.4 .target sm_80 .entry k() .reqntid32 {ret;}",
            ".version 9.4 .target sm_80 .file1 \"kernel.cu\"",
        ] {
            assert!(parse_module(source).is_err(), "{source}");
        }
        for source in [
            ".version/**/9.4 .target/**/sm_80 .global.u32 value;",
            ".version 9.4 .target sm_80 .entry k() {.reg.u32%r; mov.u32 %r,1; ret;}",
        ] {
            assert!(parse_module(source).is_ok(), "{source}");
        }
    }

    #[test]
    fn test_identifier_prefix_is_one_character_and_the_rest_may_be_underscores() {
        for name in ["__", "_$", "$__", "%_", "%$"] {
            let source = format!(
                ".version 9.4 .target sm_80
                .entry k() {{ .reg .u32 {name}; mov.u32 {name},1; ret; }}"
            );
            assert!(parse_module(&source).is_ok(), "{name}");
        }
        for name in ["%%r", "_%r", "$%r"] {
            let source = format!(
                ".version 9.4 .target sm_80
                .entry k() {{ .reg .u32 {name}; ret; }}"
            );
            assert!(parse_module(&source).is_err(), "{name}");
        }
    }

    #[test]
    fn test_scientific_literals_accept_one_exponent_in_instructions_and_initializers() {
        for literal in ["1e2", "1E-2", "-1e+2", "+1e2", "1.e2", ".5e-2", "-0.5E+2"] {
            let source = format!(
                ".version 9.4 .target sm_80
                .global .f64 value = {literal};
                .entry k() {{ .reg .f64 d; mov.f64 d,{literal}; ret; }}"
            );
            assert!(parse_module(&source).is_ok(), "{literal}");
        }
    }

    #[test]
    fn test_scientific_literals_reject_repeated_exponents() {
        for literal in ["1e2e3", "1E2E3", "-1e+2E-3", "+1E-2e+3"] {
            for body in [
                format!(".global .f64 value = {literal};"),
                format!(".entry k() {{ .reg .f64 d; mov.f64 d,{literal}; ret; }}"),
            ] {
                let source = format!(".version 9.4 .target sm_80 {body}");
                assert!(parse_module(&source).is_err(), "{literal}: {body}");
            }
        }
    }

    #[test]
    fn test_parse_no_whitespace() {
        let src = ".version 9.4\n.target sm_90\n.address_size 64";
        let module = parse_module(src).unwrap();
        assert_eq!(module.version.major, 9);
        assert_eq!(module.version.minor, 4);
    }

    #[test]
    fn test_parse_minimal_module() {
        let src = r"
            .version 9.4
            .target sm_90
            .address_size 64
        ";

        let module = parse_module(src).unwrap();
        assert_eq!(module.version.major, 9);
        assert_eq!(module.version.minor, 4);
        assert_eq!(module.target.specifiers, vec!["sm_90"]);
        assert_eq!(module.address_size, AddressSize::Bits64);
        assert!(module.items.is_empty());
    }

    #[test]
    fn test_parse_multiple_targets() {
        let src = r"
            .version 9.4
            .target sm_90, texmode_unified
            .address_size 64
        ";

        let module = parse_module(src).unwrap();
        assert_eq!(module.target.specifiers, vec!["sm_90", "texmode_unified"]);
    }

    #[test]
    fn test_parse_32bit_address() {
        let src = r"
            .version 8.5
            .target sm_80
            .address_size 32
        ";

        let module = parse_module(src).unwrap();
        assert_eq!(module.address_size, AddressSize::Bits32);
    }

    #[test]
    fn test_parse_default_32bit_address_when_omitted() {
        let src = r"
            .version 9.4
            .target sm_90
        ";

        let module = parse_module(src).unwrap();
        assert_eq!(module.address_size, AddressSize::Bits32);
    }

    #[test]
    fn test_parse_with_comments() {
        let src = r"
            // PTX module
            .version 9.4
            .target sm_90  // target arch
            .address_size 64
            /* end of preamble */
        ";

        let module = parse_module(src).unwrap();
        assert_eq!(module.version.major, 9);
    }

    #[test]
    fn test_parse_empty_entry_function() {
        let src = r"
            .version 9.4
            .target sm_90
            .address_size 64

            .entry kernel() {
            }
        ";

        let module = parse_module(src).unwrap();
        assert_eq!(module.items.len(), 1);
        match &module.items[0] {
            TopLevelItem::Function(f) => {
                assert!(f.entry);
                assert_eq!(f.name, "kernel");
                assert!(f.params.is_empty());
                assert!(f.body.as_ref().unwrap().is_empty());
            }
            other => panic!("expected function, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_entry_with_instructions() {
        let src = r"
            .version 9.4
            .target sm_90
            .address_size 64

            .entry kernel() {
                add.s32 %r1, %r2, %r3;
                ret;
            }
        ";

        let module = parse_module(src).unwrap();
        let func = match &module.items[0] {
            TopLevelItem::Function(f) => f,
            other => panic!("expected function, got {other:?}"),
        };
        let body = func.body.as_ref().unwrap();
        assert_eq!(body.len(), 2);

        match &body[0] {
            Statement::Instruction(inst) => {
                assert_eq!(inst.head, "add.s32");
                assert_eq!(inst.operands.len(), 3);
            }
            other => panic!("expected instruction, got {other:?}"),
        }

        match &body[1] {
            Statement::Instruction(inst) => {
                assert_eq!(inst.head, "ret");
                assert!(inst.operands.is_empty());
            }
            other => panic!("expected instruction, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_predicated_instruction() {
        let src = r"
            .version 9.4
            .target sm_90
            .address_size 64

            .entry kernel() {
                @%p1 add.s32 %r1, %r2, %r3;
                @!%p2 bra label;
            }
        ";

        let module = parse_module(src).unwrap();
        let func = match &module.items[0] {
            TopLevelItem::Function(f) => f,
            other => panic!("expected function, got {other:?}"),
        };
        let body = func.body.as_ref().unwrap();

        match &body[0] {
            Statement::Instruction(inst) => {
                let guard = inst.guard.as_ref().unwrap();
                assert!(!guard.negated);
                assert_eq!(guard.register, "%p1");
            }
            other => panic!("expected instruction, got {other:?}"),
        }

        match &body[1] {
            Statement::Instruction(inst) => {
                let guard = inst.guard.as_ref().unwrap();
                assert!(guard.negated);
                assert_eq!(guard.register, "%p2");
                assert_eq!(inst.head, "bra");
                match &inst.operands[0] {
                    Operand::Symbol { name, .. } => assert_eq!(name, "label"),
                    other => panic!("expected label, got {other:?}"),
                }
            }
            other => panic!("expected instruction, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_function_with_params() {
        let src = r"
            .version 9.4
            .target sm_90
            .address_size 64

            .entry foo(.param .b32 N, .param .b8 buffer[64]) {
                ret;
            }
        ";

        let module = parse_module(src).unwrap();
        let func = match &module.items[0] {
            TopLevelItem::Function(f) => f,
            other => panic!("expected function, got {other:?}"),
        };
        assert_eq!(func.params.len(), 2);
        assert_eq!(func.params[0].name, "N");
        assert_eq!(func.params[1].name, "buffer");
        assert_eq!(func.params[1].array_bounds, vec![Some(64)]);
    }

    #[test]
    fn test_parse_func_without_return_params() {
        let src = r"
            .version 9.4
            .target sm_90
            .address_size 64

            .func foo(.param .b32 x) {
                ret;
            }
        ";

        let module = parse_module(src).unwrap();
        let func = match &module.items[0] {
            TopLevelItem::Function(func) => func,
            other => panic!("expected function, got {other:?}"),
        };

        assert!(func.return_params.is_empty());
        assert_eq!(func.params.len(), 1);
        assert_eq!(func.params[0].name, "x");
    }

    #[test]
    fn test_parse_function_with_return_params_and_directives() {
        let src = r"
            .version 9.4
            .target sm_90
            .address_size 64

            .func (.reg .b32 rval) foo(.reg .b32 x) .noreturn .abi_preserve 8 .abi_preserve_control 2;
        ";

        let module = parse_module(src).unwrap();
        let func = match &module.items[0] {
            TopLevelItem::Function(func) => func,
            other => panic!("expected function, got {other:?}"),
        };

        assert_eq!(func.return_params.len(), 1);
        assert_eq!(func.return_params[0].name, "rval");
        assert_eq!(func.params.len(), 1);
        assert_eq!(func.directives.len(), 3);
        assert!(matches!(
            func.directives[0],
            FunctionDirective::NoReturn { .. }
        ));
        assert!(matches!(
            func.directives[1],
            FunctionDirective::AbiPreserve { value: 8, .. }
        ));
        assert!(matches!(
            func.directives[2],
            FunctionDirective::AbiPreserveControl { value: 2, .. }
        ));
    }

    #[test]
    fn test_parse_label_and_branch() {
        let src = r"
            .version 9.4
            .target sm_90
            .address_size 64

            .entry kernel() {
            start:
                mov.b32 %r1, %r2;
                bra start;
            }
        ";

        let module = parse_module(src).unwrap();
        let func = match &module.items[0] {
            TopLevelItem::Function(f) => f,
            other => panic!("expected function, got {other:?}"),
        };
        let body = func.body.as_ref().unwrap();

        match &body[0] {
            Statement::Label { name, .. } => assert_eq!(name, "start"),
            other => panic!("expected label, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_variable_initializers_and_multidimensional_arrays() {
        let src = r"
            .version 9.4
            .target sm_90
            .address_size 64

            .visible .global .align 8 .u64 g_data[2] = {1, 2};
            .global .s32 offset[][2] = {{-1, 0}, {0, -1}};
            .const .u32 p = generic(g_data) + 8;
        ";

        let module = parse_module(src).unwrap();
        assert_eq!(module.items.len(), 3);

        let variable = match &module.items[0] {
            TopLevelItem::Variable(variable) => variable,
            other => panic!("expected variable, got {other:?}"),
        };
        assert_eq!(variable.array_bounds, vec![Some(2)]);
        match variable.initializer.as_ref().unwrap() {
            Initializer::List { values, .. } => assert_eq!(values.len(), 2),
            other => panic!("expected list initializer, got {other:?}"),
        }

        let variable = match &module.items[1] {
            TopLevelItem::Variable(variable) => variable,
            other => panic!("expected variable, got {other:?}"),
        };
        assert_eq!(variable.array_bounds, vec![None, Some(2)]);

        let variable = match &module.items[2] {
            TopLevelItem::Variable(variable) => variable,
            other => panic!("expected variable, got {other:?}"),
        };
        match variable.initializer.as_ref().unwrap() {
            Initializer::Binary {
                op: InitializerBinaryOp::Add,
                left,
                right,
                ..
            } => {
                assert!(
                    matches!(left.as_ref(), Initializer::Generic { expr: inner, .. } if matches!(inner.as_ref(), Initializer::Symbol { name: symbol, .. } if symbol == "g_data"))
                );
                assert!(matches!(
                    right.as_ref(),
                    Initializer::Integer { value: 8, .. }
                ));
            }
            other => panic!("expected generic add initializer, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_function_body_variables_blocks_and_directives() {
        let src = r#"
            .version 8.5
            .target sm_80
            .address_size 64

            .visible .func helper(.param .b32 helper_param)
            {
                .reg .b32 %r0;
                mov.u32 %r0, %r0;
                ret;
            }

            .visible .entry step64_kernel(
                .param .u64 param0,
                .param .u32 param1
            )
            .maxnreg 32
            .reqntid 256, 1, 1
            {
                .reg .pred %p1;
                .reg .b32 %r1;
                .loc 1 42 3
                mov.u32 %r1, %tid.x;
                .pragma "nounroll";
            $L_loop:
                entry_br: .branchtargets $L_loop;
                entry_call: .calltargets helper;
                entry_proto: .callprototype _ (.param .u32 param_placeholder) .abi_preserve 4 .abi_preserve_control 2;
                {
                    .reg .b32 %r3;
                    add.s32 %r3, %r1, -1;
                }
                ret;
            }
        "#;

        let module = parse_module(src).unwrap();
        assert_eq!(module.items.len(), 2);

        let entry = match &module.items[1] {
            TopLevelItem::Function(func) => func,
            other => panic!("expected function, got {other:?}"),
        };
        assert!(matches!(
            entry.directives[0],
            FunctionDirective::MaxNReg { value: 32, .. }
        ));
        assert!(
            matches!(entry.directives[1], FunctionDirective::ReqNTid { ref values, .. } if values == &vec![256, 1, 1])
        );

        let body = entry.body.as_ref().unwrap();
        assert!(matches!(body[0], Statement::Variable(_)));
        assert!(matches!(body[1], Statement::Variable(_)));
        assert!(matches!(
            body[2],
            Statement::Directive(DirectiveStatement::Loc(_))
        ));
        assert!(matches!(
            body[4],
            Statement::Directive(DirectiveStatement::Pragma { .. })
        ));
        assert!(matches!(&body[5], Statement::Label { name, .. } if name == "$L_loop"));
        assert!(matches!(&body[6], Statement::Label { name, .. } if name == "entry_br"));
        assert!(matches!(
            body[7],
            Statement::Directive(DirectiveStatement::BranchTargets { .. })
        ));
        assert!(matches!(&body[8], Statement::Label { name, .. } if name == "entry_call"));
        assert!(matches!(
            body[9],
            Statement::Directive(DirectiveStatement::CallTargets { .. })
        ));
        assert!(matches!(&body[10], Statement::Label { name, .. } if name == "entry_proto"));
        assert!(matches!(
            body[11],
            Statement::Directive(DirectiveStatement::CallPrototype { .. })
        ));
        assert!(matches!(body[12], Statement::Block { .. }));
    }

    #[test]
    fn test_parse_negative_address_offset() {
        let src = r"
            .version 9.4
            .target sm_90
            .address_size 64

            .entry kernel(.param .u64 ptr) {
                ld.global.u32 %r1, [%rd1 - 8];
                ret;
            }
        ";

        let module = parse_module(src).unwrap();
        let func = match &module.items[0] {
            TopLevelItem::Function(func) => func,
            other => panic!("expected function, got {other:?}"),
        };
        let body = func.body.as_ref().unwrap();

        match &body[0] {
            Statement::Instruction(inst) => match &inst.operands[1] {
                Operand::Address { base, offset, .. } => {
                    assert_eq!(base, "%rd1");
                    assert_eq!(*offset, -8);
                }
                other => panic!("expected register offset address, got {other:?}"),
            },
            other => panic!("expected instruction, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_binary_and_octal_literals() {
        let src = r"
            .version 9.4
            .target sm_90
            .address_size 64

            .entry kernel() {
                mov.u32 %r1, 0b1010;
                mov.u32 %r2, 017;
                ret;
            }
        ";

        let module = parse_module(src).unwrap();
        let func = match &module.items[0] {
            TopLevelItem::Function(func) => func,
            other => panic!("expected function, got {other:?}"),
        };
        let body = func.body.as_ref().unwrap();

        match &body[0] {
            Statement::Instruction(inst) => {
                assert!(matches!(inst.operands[1], Operand::Immediate { .. }));
            }
            other => panic!("expected instruction, got {other:?}"),
        }
        match &body[1] {
            Statement::Instruction(inst) => {
                assert!(matches!(inst.operands[1], Operand::Immediate { .. }));
            }
            other => panic!("expected instruction, got {other:?}"),
        }
    }

    #[test]
    fn test_reject_identifier_with_dot() {
        let src = r"
            .version 9.4
            .target sm_90.something
            .address_size 64
        ";

        assert!(parse_module(src).is_err());
    }

    #[test]
    fn test_parse_alias_and_section_and_file_metadata() {
        let src = r#"
            .version 9.4
            .target sm_90, debug
            .address_size 64
            .file 1 "kernel.cu", 1339013327, 64118;
            .visible .func foo();
            .visible .func bar();
            .alias bar, foo;
            .section .debug_info {
                .b32 11430
            }
        "#;

        let module = parse_module(src).unwrap();
        assert!(matches!(
            module.items[0],
            TopLevelItem::File {
                timestamp: Some(1_339_013_327),
                file_size: Some(64118),
                ..
            }
        ));
        assert!(matches!(module.items[3], TopLevelItem::Alias { .. }));
        assert!(
            matches!(module.items[4], TopLevelItem::Section(ref section) if section.name == ".debug_info")
        );
    }

    #[test]
    fn test_parse_file_without_semicolon() {
        let src = r#"
            .version 9.4
            .target sm_90
            .file 1 "kernel.cu"
        "#;

        let module = parse_module(src).unwrap();
        assert!(matches!(
            module.items[0],
            TopLevelItem::File {
                index: 1,
                timestamp: Some(0),
                file_size: Some(0),
                ..
            }
        ));
    }

    #[test]
    fn test_parse_entry_scoped_pragma_before_body() {
        let src = r#"
            .version 9.4
            .target sm_90
            .address_size 64

            .entry kernel() .pragma "nounroll"; {
                ret;
            }
        "#;

        let module = parse_module(src).unwrap();
        let func = match &module.items[0] {
            TopLevelItem::Function(func) => func,
            other => panic!("expected function, got {other:?}"),
        };

        assert!(matches!(
            func.directives[0],
            FunctionDirective::Pragma { ref value, .. } if value == "nounroll"
        ));
        assert!(func.body.is_some());
    }

    #[test]
    fn test_parse_leading_dot_float_initializer() {
        let src = r"
            .version 9.4
            .target sm_90
            .address_size 64

            .global .f32 blur_kernel[][3] = {{.05, .1, .05}, {.1, .4, .1}, {.05, .1, .05}};
        ";

        let module = parse_module(src).unwrap();
        let variable = match &module.items[0] {
            TopLevelItem::Variable(variable) => variable,
            other => panic!("expected variable, got {other:?}"),
        };

        assert_eq!(variable.array_bounds, vec![None, Some(3)]);
        assert!(matches!(
            variable.initializer,
            Some(Initializer::List { .. })
        ));
    }

    #[test]
    fn test_parse_initializer_expression_precedence_and_casts() {
        let src = r"
            .version 9.4
            .target sm_90
            .address_size 64

            .global .u64 expr0 = 1 + 2 * 3;
            .global .u64 expr1 = (1 + 2) * 3;
            .global .u64 expr2 = 1 ? 2 : 3;
            .global .u64 expr3 = (.u64) (1 + 2);
            .global .u64 expr4 = ~1 & 3;
        ";

        let module = parse_module(src).unwrap();

        let expr0 = match &module.items[0] {
            TopLevelItem::Variable(variable) => variable.initializer.as_ref().unwrap(),
            other => panic!("expected variable, got {other:?}"),
        };
        assert!(matches!(
            expr0,
            Initializer::Binary {
                op: InitializerBinaryOp::Add,
                ..
            }
        ));

        let expr1 = match &module.items[1] {
            TopLevelItem::Variable(variable) => variable.initializer.as_ref().unwrap(),
            other => panic!("expected variable, got {other:?}"),
        };
        assert!(matches!(
            expr1,
            Initializer::Binary {
                op: InitializerBinaryOp::Mul,
                ..
            }
        ));

        let expr2 = match &module.items[2] {
            TopLevelItem::Variable(variable) => variable.initializer.as_ref().unwrap(),
            other => panic!("expected variable, got {other:?}"),
        };
        assert!(matches!(expr2, Initializer::Conditional { .. }));

        let expr3 = match &module.items[3] {
            TopLevelItem::Variable(variable) => variable.initializer.as_ref().unwrap(),
            other => panic!("expected variable, got {other:?}"),
        };
        assert!(matches!(
            expr3,
            Initializer::Cast {
                ty: InitializerCastType::U64,
                ..
            }
        ));

        let expr4 = match &module.items[4] {
            TopLevelItem::Variable(variable) => variable.initializer.as_ref().unwrap(),
            other => panic!("expected variable, got {other:?}"),
        };
        assert!(matches!(
            expr4,
            Initializer::Binary {
                op: InitializerBinaryOp::BitAnd,
                left,
                ..
            } if matches!(left.as_ref(), Initializer::Unary { op: InitializerUnaryOp::BitwiseNot, .. })
        ));
    }

    #[test]
    fn test_parse_mask_and_generic_expression_initializers() {
        let src = r"
            .version 9.4
            .target sm_90
            .address_size 64

            .const .u32 foo = 42;
            .global .u32 ptr = generic(foo) + 8;
            .global .u8 addr[] = {0xff(foo + 4), 0xff(generic(foo) + 4), 0xff(1000 + 546)};
        ";

        let module = parse_module(src).unwrap();

        let ptr = match &module.items[1] {
            TopLevelItem::Variable(variable) => variable.initializer.as_ref().unwrap(),
            other => panic!("expected variable, got {other:?}"),
        };
        assert!(matches!(
            ptr,
            Initializer::Binary {
                op: InitializerBinaryOp::Add,
                left,
                right,
                ..
            } if matches!(left.as_ref(), Initializer::Generic { .. }) && matches!(right.as_ref(), Initializer::Integer { value: 8, .. })
        ));

        let addr = match &module.items[2] {
            TopLevelItem::Variable(variable) => variable.initializer.as_ref().unwrap(),
            other => panic!("expected variable, got {other:?}"),
        };
        assert!(
            matches!(addr, Initializer::List { values, .. } if matches!(&values[0], Initializer::Masked { .. }))
        );
    }

    #[test]
    fn test_parse_setp_dual_destination_and_negated_predicate_input() {
        let src = r"
            .version 9.4
            .target sm_90
            .address_size 64

            .entry kernel() {
                setp.lt.and.s32 %p1|%p2, %r1, %r2, !%p3;
                ret;
            }
        ";

        let module = parse_module(src).unwrap();
        let func = match &module.items[0] {
            TopLevelItem::Function(func) => func,
            other => panic!("expected function, got {other:?}"),
        };
        let body = func.body.as_ref().unwrap();

        match &body[0] {
            Statement::Instruction(inst) => {
                assert_eq!(inst.head, "setp.lt.and.s32");
                assert_eq!(inst.operands.len(), 4);
                assert!(
                    matches!(&inst.operands[0], Operand::Pair { first, second, .. }
                    if matches!(first.as_ref(), Operand::Symbol { name, .. } if name == "%p1")
                    && matches!(second.as_ref(), Operand::Symbol { name, .. } if name == "%p2"))
                );
                assert!(matches!(&inst.operands[3], Operand::Negated { operand, .. }
                    if matches!(operand.as_ref(), Operand::Symbol { name, .. } if name == "%p3")));
            }
            other => panic!("expected instruction, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_setp_with_bit_bucket_destination() {
        let src = r"
            .version 9.4
            .target sm_90
            .address_size 64

            .entry kernel() {
                setp.eq.s32 _|%p2, %r1, %r2;
                ret;
            }
        ";

        let module = parse_module(src).unwrap();
        let func = match &module.items[0] {
            TopLevelItem::Function(func) => func,
            other => panic!("expected function, got {other:?}"),
        };
        let body = func.body.as_ref().unwrap();

        match &body[0] {
            Statement::Instruction(inst) => {
                assert!(
                    matches!(&inst.operands[0], Operand::Pair { first, second, .. }
                    if matches!(first.as_ref(), Operand::BitBucket { .. })
                    && matches!(second.as_ref(), Operand::Symbol { name, .. } if name == "%p2"))
                );
            }
            other => panic!("expected instruction, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_variable_and_function_attributes() {
        let src = r"
            .version 9.4
            .target sm_90
            .address_size 64

            .global .attribute(.managed) .s32 g;
            .global .attribute(.unified(19, 95)) .f32 f;
            .func .attribute(.unified(0xAB, 0xCD)) bar() {
                ret;
            }
        ";

        let module = parse_module(src).unwrap();

        let g = match &module.items[0] {
            TopLevelItem::Variable(variable) => variable,
            other => panic!("expected variable, got {other:?}"),
        };
        assert!(matches!(
            g.attributes.as_slice(),
            [Attribute::Managed { .. }]
        ));

        let f = match &module.items[1] {
            TopLevelItem::Variable(variable) => variable,
            other => panic!("expected variable, got {other:?}"),
        };
        assert!(matches!(
            f.attributes.as_slice(),
            [Attribute::Unified {
                uuid1: 19,
                uuid2: 95,
                ..
            }]
        ));

        let bar = match &module.items[2] {
            TopLevelItem::Function(function) => function,
            other => panic!("expected function, got {other:?}"),
        };
        assert!(matches!(
            bar.attributes.as_slice(),
            [Attribute::Unified {
                uuid1: 0xAB,
                uuid2: 0xCD,
                ..
            }]
        ));
    }

    #[test]
    fn test_parse_structured_loc_directive() {
        let src = r"
            .version 9.4
            .target sm_90
            .address_size 64

            .entry kernel() {
                .loc 1 15 3, function_name .debug_str+16, inlined_at 1 10 5
                ret;
            }
        ";

        let module = parse_module(src).unwrap();
        let func = match &module.items[0] {
            TopLevelItem::Function(function) => function,
            other => panic!("expected function, got {other:?}"),
        };

        match &func.body.as_ref().unwrap()[0] {
            Statement::Directive(DirectiveStatement::Loc(loc)) => {
                assert_eq!(loc.file, 1);
                assert_eq!(loc.line, 15);
                assert_eq!(loc.column, 3);
                assert!(matches!(
                    loc.function_name,
                    Some(LocFunctionName::LabelOffset { ref label, offset, .. }) if label == ".debug_str" && offset == 16
                ));
                let inline = loc.inlined_at.as_ref().unwrap();
                assert_eq!(inline.file, 1);
                assert_eq!(inline.line, 10);
                assert_eq!(inline.column, 5);
            }
            other => panic!("expected loc directive, got {other:?}"),
        }
    }

    #[test]
    fn test_reject_negative_version_components() {
        let src = ".version -1.-2\n.target sm_90\n.address_size 64\n";
        assert!(parse_module(src).is_err());
    }

    #[test]
    fn test_reject_negative_file_indices() {
        let src = ".version 9.4\n.target sm_90\n.file -1 \"kernel.cu\"\n";
        assert!(parse_module(src).is_err());
    }

    #[test]
    fn test_preserve_unknown_instruction_modifiers() {
        let src = r"
            .version 9.4
            .target sm_90
            .address_size 64

            .entry kernel() {
                add.s32.fake %r1, %r2, %r3;
            }
        ";

        assert!(parse_module(src).is_ok());
    }

    #[test]
    fn test_parse_setp_with_compare_only_modifier() {
        let src = r"
            .version 9.4
            .target sm_90
            .address_size 64

            .entry kernel() {
                setp.ge.s32 %p1, %r1, %r2;
                ret;
            }
        ";

        let module = parse_module(src).unwrap();
        let function = match &module.items[0] {
            TopLevelItem::Function(function) => function,
            other => panic!("expected function, got {other:?}"),
        };
        let body = function.body.as_ref().unwrap();
        match &body[0] {
            Statement::Instruction(inst) => {
                assert_eq!(inst.head, "setp.ge.s32");
            }
            other => panic!("expected instruction, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_mul_with_mode_and_type_modifiers() {
        let src = r"
            .version 9.4
            .target sm_90
            .address_size 64

            .entry kernel() {
                mul.wide.s32 %rd4, %r1, 4;
                ret;
            }
        ";

        let module = parse_module(src).unwrap();
        let function = match &module.items[0] {
            TopLevelItem::Function(function) => function,
            other => panic!("expected function, got {other:?}"),
        };
        let body = function.body.as_ref().unwrap();
        match &body[0] {
            Statement::Instruction(inst) => {
                assert_eq!(inst.head, "mul.wide.s32");
            }
            other => panic!("expected instruction, got {other:?}"),
        }
    }

    #[test]
    fn test_keep_directive_spans_relative_to_module_source() {
        let src = r#"
            .version 9.4
            .target sm_90
            .address_size 64

            .file 1 "kernel.cu";
            .alias bar, foo;
            .section .debug_info {
                start:
                .b32 start+12
            }

            .entry kernel() {
                .pragma "nounroll";
                .loc 1 15 3, function_name .debug_str+16, inlined_at 1 10 5
                ret;
            }
        "#;

        let module = parse_module(src).unwrap();

        let file = match &module.items[0] {
            TopLevelItem::File { span, .. } => span,
            other => panic!("expected file directive, got {other:?}"),
        };
        assert_eq!(&src[file.start..file.end], ".file 1 \"kernel.cu\";");

        let alias = match &module.items[1] {
            TopLevelItem::Alias { span, .. } => span,
            other => panic!("expected alias directive, got {other:?}"),
        };
        assert_eq!(&src[alias.start..alias.end], ".alias bar, foo;");

        let section = match &module.items[2] {
            TopLevelItem::Section(section) => section,
            other => panic!("expected section directive, got {other:?}"),
        };
        assert_eq!(
            &src[section.span.start..section.span.end],
            ".section .debug_info {\n                start:\n                .b32 start+12\n            }"
        );
        match &section.lines[0] {
            SectionLine::Label { span, name } => {
                assert_eq!(name, "start");
                assert_eq!(&src[span.start..span.end], "start:");
            }
            other @ SectionLine::Data { .. } => panic!("expected section label, got {other:?}"),
        }
        match &section.lines[1] {
            SectionLine::Data { values, .. } => match &values[0] {
                SectionValue::LabelOffset {
                    span,
                    label,
                    offset,
                } => {
                    assert_eq!(label, "start");
                    assert_eq!(*offset, 12);
                    assert_eq!(&src[span.start..span.end], "start+12");
                }
                other => panic!("expected label offset, got {other:?}"),
            },
            other @ SectionLine::Label { .. } => panic!("expected section data, got {other:?}"),
        }

        let function = match &module.items[3] {
            TopLevelItem::Function(function) => function,
            other => panic!("expected function, got {other:?}"),
        };
        let body = function.body.as_ref().unwrap();
        match &body[0] {
            Statement::Directive(DirectiveStatement::Pragma { span, value }) => {
                assert_eq!(value, "nounroll");
                assert_eq!(&src[span.start..span.end], ".pragma \"nounroll\";");
            }
            other => panic!("expected pragma, got {other:?}"),
        }
        match &body[1] {
            Statement::Directive(DirectiveStatement::Loc(loc)) => {
                assert_eq!(
                    &src[loc.span.start..loc.span.end],
                    ".loc 1 15 3, function_name .debug_str+16, inlined_at 1 10 5"
                );
                match loc.function_name.as_ref().unwrap() {
                    LocFunctionName::LabelOffset {
                        span,
                        label,
                        offset,
                    } => {
                        assert_eq!(label, ".debug_str");
                        assert_eq!(*offset, 16);
                        assert_eq!(&src[span.start..span.end], ".debug_str+16");
                    }
                    other @ LocFunctionName::Label { .. } => {
                        panic!("expected function name label offset, got {other:?}")
                    }
                }
                let inline = loc.inlined_at.as_ref().unwrap();
                assert_eq!(&src[inline.span.start..inline.span.end], " 1 10 5");
            }
            other => panic!("expected loc directive, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_nvcc_emitted_ptx_control_flow_sequence() {
        let src = r"
//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: UNKNOWN
// Cuda compilation tools, release 13.2, V13.2.51
// Based on NVVM 7.0.1
//

.version 9.2
.target sm_75
.address_size 64

	// .globl	scale_add

.visible .entry scale_add(
	.param .u64 scale_add_param_0,
	.param .u64 scale_add_param_1,
	.param .f32 scale_add_param_2,
	.param .u32 scale_add_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<6>;
	.reg .b64 	%rd<8>;


	ld.param.u64 	%rd1, [scale_add_param_0];
	ld.param.u64 	%rd2, [scale_add_param_1];
	ld.param.f32 	%f1, [scale_add_param_2];
	ld.param.u32 	%r2, [scale_add_param_3];
	mov.u32 	%r3, %ctaid.x;
	mov.u32 	%r4, %ntid.x;
	mov.u32 	%r5, %tid.x;
	mad.lo.s32 	%r1, %r3, %r4, %r5;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd3, %rd1;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.f32 	%f2, [%rd5];
	fma.rn.f32 	%f3, %f2, %f1, 0f3F800000;
	cvta.to.global.u64 	%rd6, %rd2;
	add.s64 	%rd7, %rd6, %rd4;
	st.global.f32 	[%rd7], %f3;

$L__BB0_2:
	ret;

}
";

        let parsed = parse_module(src);
        assert!(parsed.is_ok(), "{parsed:?}");
    }

    #[test]
    fn test_parse_section_debug_lines() {
        let src = r"
            .version 9.4
            .target sm_90
            .address_size 64

            .section .debug_info {
                start:
                .b32 start+12
                .b64 start-start
                .b16 -5, -65535
                .b8 2, 0
            }
        ";

        let module = parse_module(src).unwrap();
        let section = match &module.items[0] {
            TopLevelItem::Section(section) => section,
            other => panic!("expected section, got {other:?}"),
        };

        assert_eq!(section.name, ".debug_info");
        assert!(matches!(&section.lines[0], SectionLine::Label { name, .. } if name == "start"));
        assert!(matches!(
            section.lines[1],
            SectionLine::Data {
                width: SectionDataWidth::B32,
                ref values,
                ..
            } if matches!(values[0], SectionValue::LabelOffset { ref label, offset, .. } if label == "start" && offset == 12)
        ));
        assert!(matches!(
            section.lines[2],
            SectionLine::Data {
                width: SectionDataWidth::B64,
                ref values,
                ..
            } if matches!(values[0], SectionValue::LabelDifference { ref left, ref right, .. } if left == "start" && right == "start")
        ));
        assert!(matches!(
            section.lines[3],
            SectionLine::Data {
                width: SectionDataWidth::B16,
                ref values,
                ..
            } if matches!(values[0], SectionValue::Integer { value: -5, .. })
                && matches!(values[1], SectionValue::Integer { value: -65535, .. })
        ));
    }

    #[test]
    fn test_track_spans_for_structural_nodes() {
        let src = ".version 9.4\n.target sm_90\n.address_size 64\n.entry kernel() {\nlabel0:\n  ret;\n}\n";

        let module = parse_module(src).unwrap();
        assert_eq!(module.span.start, 0);
        assert_eq!(module.span.end, src.len());
        assert_eq!(
            &src[module.version.span.start..module.version.span.end],
            ".version 9.4"
        );
        assert_eq!(
            &src[module.target.span.start..module.target.span.end],
            ".target sm_90"
        );

        let func = match &module.items[0] {
            TopLevelItem::Function(function) => function,
            other => panic!("expected function, got {other:?}"),
        };
        assert_eq!(
            &src[func.span.start..func.span.end],
            ".entry kernel() {\nlabel0:\n  ret;\n}"
        );

        let body = func.body.as_ref().unwrap();
        match &body[0] {
            Statement::Label { span, name } => {
                assert_eq!(name, "label0");
                assert_eq!(&src[span.start..span.end], "label0:");
            }
            other => panic!("expected label, got {other:?}"),
        }

        match &body[1] {
            Statement::Instruction(inst) => {
                assert_eq!(&src[inst.span.start..inst.span.end], "ret;");
            }
            other => panic!("expected instruction, got {other:?}"),
        }

        let start = func.span.start_position(src);
        let end = func.span.end_position(src);
        assert_eq!((start.line, start.column), (4, 1));
        assert_eq!((end.line, end.column), (7, 2));
    }

    #[test]
    fn test_track_spans_for_operand_leaves() {
        let src = ".version 9.4\n.target sm_90\n.address_size 64\n.entry kernel() {\n  @!%p0 add.s32 %r1, [%rd4 + 16], 0b1010;\n}\n";

        let module = parse_module(src).unwrap();
        let func = match &module.items[0] {
            TopLevelItem::Function(function) => function,
            other => panic!("expected function, got {other:?}"),
        };

        let inst = match &func.body.as_ref().unwrap()[0] {
            Statement::Instruction(inst) => inst,
            other => panic!("expected instruction, got {other:?}"),
        };

        let guard = inst.guard.as_ref().unwrap();
        assert_eq!(&src[guard.span.start..guard.span.end], "@!%p0");

        match &inst.operands[0] {
            Operand::Symbol { span, name } => {
                assert_eq!(name, "%r1");
                assert_eq!(&src[span.start..span.end], "%r1");
            }
            other => panic!("expected register, got {other:?}"),
        }

        match &inst.operands[1] {
            Operand::Address { span, base, offset } => {
                assert_eq!(&src[span.start..span.end], "[%rd4 + 16]");
                assert_eq!(base, "%rd4");
                assert_eq!(*offset, 16);
            }
            other => panic!("expected address operand, got {other:?}"),
        }

        match &inst.operands[2] {
            Operand::Immediate { span, value } => {
                assert_eq!(&src[span.start..span.end], "0b1010");
                assert_eq!(*value, Immediate::Integer(10));
            }
            other => panic!("expected immediate operand, got {other:?}"),
        }
    }

    #[test]
    fn test_track_spans_for_initializer_leaves() {
        let src = ".version 9.4\n.target sm_90\n.address_size 64\n.const .u32 foo = 42;\n.global .u64 expr = generic(foo) + 8;\n.global .u8 addr[] = {0xff(foo + 4)};\n";

        let module = parse_module(src).unwrap();

        let expr = match &module.items[1] {
            TopLevelItem::Variable(variable) => variable.initializer.as_ref().unwrap(),
            other => panic!("expected variable, got {other:?}"),
        };

        match expr {
            Initializer::Binary {
                span,
                op: InitializerBinaryOp::Add,
                left,
                right,
            } => {
                assert_eq!(&src[span.start..span.end], "generic(foo) + 8");
                match left.as_ref() {
                    Initializer::Generic { span, expr } => {
                        assert_eq!(&src[span.start..span.end], "generic(foo)");
                        match expr.as_ref() {
                            Initializer::Symbol { span, name } => {
                                assert_eq!(name, "foo");
                                assert_eq!(&src[span.start..span.end], "foo");
                            }
                            other => panic!("expected symbol initializer, got {other:?}"),
                        }
                    }
                    other => panic!("expected generic initializer, got {other:?}"),
                }
                match right.as_ref() {
                    Initializer::Integer { span, value } => {
                        assert_eq!(*value, 8);
                        assert_eq!(&src[span.start..span.end], "8");
                    }
                    other => panic!("expected integer initializer, got {other:?}"),
                }
            }
            other => panic!("expected binary initializer, got {other:?}"),
        }

        let addr = match &module.items[2] {
            TopLevelItem::Variable(variable) => variable.initializer.as_ref().unwrap(),
            other => panic!("expected variable, got {other:?}"),
        };

        match addr {
            Initializer::List { values, .. } => match &values[0] {
                Initializer::Masked { span, mask, value } => {
                    assert_eq!(*mask, 0xff);
                    assert_eq!(&src[span.start..span.end], "0xff(foo + 4)");
                    match value.as_ref() {
                        Initializer::Binary {
                            span,
                            op: InitializerBinaryOp::Add,
                            left,
                            right,
                        } => {
                            assert_eq!(&src[span.start..span.end], "foo + 4");
                            assert!(
                                matches!(left.as_ref(), Initializer::Symbol { name, .. } if name == "foo")
                            );
                            assert!(matches!(
                                right.as_ref(),
                                Initializer::Integer { value: 4, .. }
                            ));
                        }
                        other => panic!("expected binary initializer, got {other:?}"),
                    }
                }
                other => panic!("expected masked initializer, got {other:?}"),
            },
            other => panic!("expected list initializer, got {other:?}"),
        }
    }

    #[test]
    fn test_integer_literals_use_one_radix_grammar_in_every_context() {
        let src = ".version 9.4 .target sm_80
            .section .debug_info { .b32 010, 0x10, 0b10, 10, -010, 0xffffffffffffffff }
            .entry k() { .reg .u64 %a; ld.u64 %a,[%a+0x10]; ld.u64 %a,[%a-010]; ret; }";
        let module = parse_module(src).unwrap();
        let TopLevelItem::Section(section) = &module.items[0] else {
            panic!("expected section");
        };
        let SectionLine::Data { values, .. } = &section.lines[0] else {
            panic!("expected section data");
        };
        let values: Vec<_> = values
            .iter()
            .map(|value| match value {
                SectionValue::Integer { value, .. } => i128::from(*value),
                SectionValue::UnsignedInteger { value, .. } => i128::from(*value),
                other => panic!("expected integer, got {other:?}"),
            })
            .collect();
        assert_eq!(values, [8, 16, 2, 10, -8, i128::from(u64::MAX)]);
        let TopLevelItem::Function(function) = &module.items[1] else {
            panic!("expected function");
        };
        let offsets: Vec<_> = function
            .body
            .as_deref()
            .unwrap()
            .iter()
            .filter_map(|statement| match statement {
                Statement::Instruction(instruction) => match &instruction.operands[..] {
                    [_, Operand::Address { offset, .. }] => Some(*offset),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert_eq!(offsets, [16, -8]);
    }

    #[test]
    fn test_address_displacements_reject_repeated_signs() {
        for address in ["[a+-4]", "[a--4]", "[a+ +4]", "[a+0o4]"] {
            let src = format!(
                ".version 9.4 .target sm_80 .entry k() {{ .reg .u64 a; ld.u64 a,{address}; ret; }}"
            );
            assert!(parse_module(&src).is_err(), "{address}");
        }
    }

    #[test]
    fn test_unterminated_comments_and_strings_report_end_of_input() {
        for source in [
            ".version 9.4 .target sm_80 /* open",
            ".version 9.4 .target sm_80 .pragma \"open;",
        ] {
            let error = parse_module(source).unwrap_err();
            assert!(
                matches!(error.kind, ParseErrorKind::UnexpectedEof),
                "{source}: {error}"
            );
        }
    }

    #[test]
    fn test_string_literals_unescape_quotes_and_backslashes() {
        let module =
            parse_module(r#".version 9.4 .target sm_80 .file 1 "dir\\a \"b\".cu""#).unwrap();
        assert!(matches!(
            &module.items[0],
            TopLevelItem::File { name, .. } if name == r#"dir\a "b".cu"#
        ));
    }
}
