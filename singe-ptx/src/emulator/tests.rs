use std::{
    sync::{Arc, Barrier},
    thread,
};

use crate::{
    ast::{
        FloatLiteral, Immediate, Initializer, Operand, ScalarType, SourcePosition, StateSpace,
        Statement, TopLevelItem,
    },
    emulator::{
        CompileErrorKind, CompiledKernel, ParsedModule,
        ir::{self, Destination, FrameRole, Operation, Predicate, Source},
        layout::InitialBits,
    },
};

fn parse(items: &str) -> ParsedModule {
    ParsedModule::parse(&format!(
        ".version 9.3\n.target sm_80\n.address_size 64\n{items}"
    ))
    .unwrap()
}

fn compile(items: &str) -> CompiledKernel {
    parse(items).compile("kernel").unwrap()
}

fn body(instructions: &str) -> CompiledKernel {
    compile(&format!(".entry kernel() {{ {instructions} }}"))
}

fn reject(items: &str, expected: &str) {
    let parsed = parse(items);
    let error = parsed.compile("kernel").unwrap_err();
    assert!(error.to_string().contains(expected), "{error}");
    assert!(
        parsed
            .source()
            .get(error.span.start..error.span.end)
            .is_some()
    );
}

#[test]
fn test_owned_code_survives_source_and_parser_drop() {
    fn shared<T: Clone + Send + Sync + 'static>() {}
    shared::<ParsedModule>();
    shared::<CompiledKernel>();
    let compiled = {
        let text = String::from(".entry kernel() { ret; }");
        let parsed = parse(&text);
        parsed.compile("kernel").unwrap()
    };
    assert_eq!(compiled.name(), "kernel");
    assert!(compiled.source().source().contains(".entry kernel"));
    assert!(matches!(
        compiled.artifact.executable.functions[0].instructions[0].operation,
        Operation::Control(ir::ControlOperation::Return)
    ));
}

#[test]
fn test_decimal_constants_use_binary64_before_binary32() {
    let code = body(".reg .f32 %f; mov.f32 %f, 1.0000000596046448; ret;");
    assert!(matches!(
        code.artifact.executable.functions[0].instructions[0].operation,
        Operation::Data(ir::DataOperation::Move {
            source: Source::Immediate(0x3f80_0000),
            ..
        })
    ));
    assert_eq!(
        "1.0000000596046448".parse::<f32>().unwrap().to_bits(),
        0x3f80_0001
    );
}

#[test]
fn test_exact_literals_preserve_nan_payload_and_negative_zero() {
    let code = body(
        ".reg .f32 %f; .reg .f64 %d; mov.f32 %f, 0f7fc00123; mov.f64 %d, 0d8000000000000000; ret;",
    );
    assert!(matches!(
        code.artifact.executable.functions[0].instructions[0].operation,
        Operation::Data(ir::DataOperation::Move {
            source: Source::Immediate(0x7fc0_0123),
            ..
        })
    ));
    assert!(matches!(
        code.artifact.executable.functions[0].instructions[1].operation,
        Operation::Data(ir::DataOperation::Move {
            source: Source::Immediate(0x8000_0000_0000_0000),
            ..
        })
    ));
}

#[test]
fn test_syntax_preserves_literal_representation() {
    let parsed =
        parse(".entry kernel() { mov.f32 %f, 1.0000000596046448; mov.f32 %f, 0f7fc00123; }");
    let TopLevelItem::Function(function) = &parsed.syntax().items[0] else {
        panic!("function");
    };
    let Statement::Instruction(first) = &function.body.as_ref().unwrap()[0] else {
        panic!("instruction");
    };
    assert!(
        matches!(&first.operands[1], Operand::Immediate { value: Immediate::Float(FloatLiteral::Decimal(value)), .. } if value == "1.0000000596046448")
    );
}

#[test]
fn test_integer_syntax_preserves_unsigned_suffix_and_c_octal() {
    let parsed =
        parse(".global .u64 values[] = {42U, 017, 0x7fffffffffffffff, 18446744073709551615};");
    let TopLevelItem::Variable(variable) = &parsed.syntax().items[0] else {
        panic!("variable");
    };
    let Initializer::List { values, .. } = variable.initializer.as_ref().unwrap() else {
        panic!("list");
    };
    assert!(matches!(
        values[0],
        Initializer::UnsignedInteger { value: 42, .. }
    ));
    assert!(matches!(values[1], Initializer::Integer { value: 15, .. }));
    assert!(matches!(
        values[2],
        Initializer::Integer {
            value: i64::MAX,
            ..
        }
    ));
    assert!(matches!(
        values[3],
        Initializer::UnsignedInteger {
            value: u64::MAX,
            ..
        }
    ));
    assert!(
        ParsedModule::parse(".version 9.3 .target sm_80 .entry k() { mov.u32 %r, 0o17; }").is_err()
    );
}

#[test]
fn test_unknown_instructions_parse_and_reject_only_for_reachable_functions() {
    let parsed = parse(".entry other() { future.instruction %unknown; } .entry kernel() { ret; }");
    parsed.compile("kernel").unwrap();
    let error = parsed.compile("other").unwrap_err();
    assert!(matches!(error.kind, CompileErrorKind::Unsupported(_)));
    assert!(parsed.source()[error.span.start..error.span.end].contains("future.instruction"));
}

#[test]
fn test_untaken_branch_does_not_hide_unsupported_instructions() {
    reject(
        ".entry kernel() { bra done; future.instruction; done: ret; }",
        "future.instruction",
    );
}

#[test]
fn test_branch_indices_and_register_scopes_are_resolved() {
    let code = body(
        ".reg .u32 %r; mov.u32 %r, 1; { .reg .u32 %r; mov.u32 %r, 2; } bra done; mov.u32 %r, 3; done: ret;",
    );
    let function = &code.artifact.executable.functions[0];
    assert_eq!(
        function
            .registers
            .iter()
            .map(|register| register.ty)
            .collect::<Vec<_>>(),
        [ScalarType::U32, ScalarType::U32]
    );
    assert!(matches!(
        function.instructions[0].operation,
        Operation::Data(ir::DataOperation::Move {
            destination: Destination::Register(ir::RegisterId(0)),
            ..
        })
    ));
    assert!(matches!(
        function.instructions[1].operation,
        Operation::Data(ir::DataOperation::Move {
            destination: Destination::Register(ir::RegisterId(1)),
            ..
        })
    ));
    assert!(matches!(
        function.instructions[2].operation,
        Operation::Control(ir::ControlOperation::Branch {
            target: ir::ProgramCounter(4),
            ..
        })
    ));
}

#[test]
fn test_nested_symbols_do_not_escape_their_scope() {
    reject(
        ".entry kernel() { { .reg .u32 %r; } mov.u32 %r, 0; ret; }",
        "undefined symbol",
    );
    reject(
        ".entry kernel() { bra inner; { inner: ret; } }",
        "undefined symbol",
    );
}

#[test]
fn test_duplicate_symbols_and_missing_registers_fail() {
    reject(
        ".entry kernel() { .reg .u32 %r; .reg .u32 %r; ret; }",
        "duplicate symbol",
    );
    reject(
        ".entry kernel() { mov.u32 %missing, 0; ret; }",
        "undefined symbol",
    );
}

#[test]
fn test_only_reachable_fallthrough_is_an_error() {
    body(".reg .u32 %r; ret; mov.u32 %r, 0;");
    body("loop: bra loop;");
    reject(
        ".entry kernel() { .reg .pred %p; @%p ret; }",
        "fall through",
    );
    reject(".entry kernel() { }", "fall through");
}

#[test]
fn test_predicate_guards_and_paired_results_are_typed() {
    let code =
        body(".reg .pred %p<2>; .reg .u32 %r; @!%p0 setp.eq.and.u32 %p0|%p1, %r, 0, !%p1; ret;");
    let instruction = &code.artifact.executable.functions[0].instructions[0];
    assert!(matches!(
        instruction.guard,
        Some(Predicate::Register {
            register: ir::RegisterId(0),
            negated: true
        })
    ));
    assert!(matches!(
        instruction.operation,
        Operation::Data(ir::DataOperation::Compare {
            inverse: Some(Destination::Register(ir::RegisterId(1))),
            combine: Some(ir::Combine {
                op: ir::BooleanOp::And,
                predicate: Predicate::Register { negated: true, .. }
            }),
            ..
        })
    ));
}

#[test]
fn test_predicate_constants_keep_boolean_semantics() {
    let code = body(".reg .pred %p; .reg .u32 %r; mov.pred %p, 42; selp.u32 %r, 1, 0, !0; ret;");
    assert!(matches!(
        code.artifact.executable.functions[0].instructions[0].operation,
        Operation::Data(ir::DataOperation::Move {
            source: Source::Immediate(1),
            ..
        })
    ));
    assert!(matches!(
        code.artifact.executable.functions[0].instructions[1].operation,
        Operation::Data(ir::DataOperation::Select {
            predicate: Predicate::Constant(true),
            ..
        })
    ));
}

#[test]
fn test_relaxed_register_widths_apply_only_to_load_store_and_convert() {
    compile(
        ".global .u8 data; .entry kernel() { .reg .b64 %r; ld.global.u8 %r, [data]; st.global.u8 [data], %r; cvt.u16.u8 %r, %r; ret; }",
    );
}

#[test]
fn test_arithmetic_rejects_mismatched_register_widths_and_literal_categories() {
    for operation in [
        "add.u16 %r, %r, 1;",
        "mov.f32 %f, 1;",
        "mov.u32 %r, 1.0;",
        "add.u32 _, %r, 1;",
    ] {
        reject(
            &format!(".entry kernel() {{ .reg .u32 %r; .reg .f32 %f; {operation} ret; }}"),
            "",
        );
    }
}

#[test]
fn test_parameter_layout_separates_pointer_alignment_and_storage_alignment() {
    let code = compile(
        ".entry kernel(.param .u8 flag, .param .u64 .ptr .global .align 16 data, .param .align 8 .b8 aggregate[12]) { ret; }",
    );
    let parameters = code.parameters();
    assert_eq!(
        parameters
            .iter()
            .map(super::layout::ParameterLayout::byte_offset)
            .collect::<Vec<_>>(),
        [0, 8, 16]
    );
    assert_eq!(parameters[1].storage().alignment(), 8);
    assert_eq!(parameters[1].pointer().unwrap().alignment, Some(16));
    assert_eq!(parameters[2].storage().byte_len(), 12);
    assert!(parameters[2].is_aggregate());
}

#[test]
fn test_parameter_layout_rejects_invalid_alignment_and_bounds() {
    reject(
        ".entry kernel(.param .align 3 .b8 bytes[4]) { ret; }",
        "power of two",
    );
    reject(
        ".entry kernel(.param .b8 bytes[]) { ret; }",
        "positive bounds",
    );
}

#[test]
fn test_direct_calls_link_once_and_validate_parameter_storage() {
    let code = compile(
        ".func (.param .u32 result) helper(.param .u32 input) { .reg .b32 %r; ld.param.u32 %r, [input]; st.param.u32 [result], %r; ret; } .entry kernel() { .param .u32 arg; .param .u32 result; st.param.u32 [arg], 42; call (result), helper, (arg); call (result), helper, (arg); ret; }",
    );
    assert_eq!(code.function_count(), 2);
    assert_eq!(code.artifact.executable.functions[1].returns.len(), 1);
    assert!(matches!(
        code.artifact.executable.functions[0].instructions[1].operation,
        Operation::Control(ir::ControlOperation::Call {
            function: ir::FunctionId(1),
            ..
        })
    ));
    reject(
        ".func helper(.param .u64 a) { ret; } .entry kernel() { .param .u32 a; call helper, (a); ret; }",
        "storage does not match",
    );
}

#[test]
fn test_recursive_calls_resolve_without_recursing_the_compiler() {
    let code = compile(
        ".func recursive() { call recursive; ret; } .entry kernel() { call recursive; ret; }",
    );
    assert_eq!(code.function_count(), 2);
}

#[test]
fn test_noreturn_call_terminates_the_callers_control_flow() {
    // An unreachable ret is legal even when the function promises not to return.
    let code = compile(".func stop() .noreturn { trap; ret; } .entry kernel() { call stop; }");
    assert!(matches!(
        code.artifact.executable.functions[0].instructions[0].operation,
        Operation::Control(ir::ControlOperation::Call { noreturn: true, .. })
    ));
}

#[test]
fn test_formal_parameters_cannot_be_used_directly_as_call_arguments() {
    reject(
        ".func helper(.param .u32 value) { ret; } .entry kernel(.param .u32 value) { call helper, (value); ret; }",
        "storage does not match",
    );
    reject(
        ".func helper(.param .u32 value) { st.param.u32 [value], 0; ret; } .entry kernel() { .param .u32 arg; call helper, (arg); ret; }",
        "read-only",
    );
}

#[test]
fn test_maxntid_hint_does_not_replace_architectural_limits() {
    let code = compile(".entry kernel() .maxntid 4096 { ret; }");
    code.launch_requirements()
        .validate_block([1024, 1, 1])
        .unwrap();
    assert!(
        code.launch_requirements()
            .validate_block([2048, 1, 1])
            .is_err()
    );
}

#[test]
fn test_initializer_relocations_validate_storage_type_and_state_space() {
    compile(".global .u32 data; .global .u32 address = generic(data); .entry kernel() { ret; }");
    reject(
        ".global .u32 data; .global .b64 address = data; .entry kernel() { ret; }",
        "address initializers require",
    );
    reject(
        ".shared .u32 data; .global .u64 address = data; .entry kernel() { ret; }",
        "global or constant storage",
    );
    reject(
        ".global .v4 .u64 data; .entry kernel() { ret; }",
        "128 bits",
    );
}

#[test]
fn test_initializer_masks_validate_the_declared_version() {
    for (version, initializer, expected) in [
        ("7.0", "0xff(data)", "PTX 7.1"),
        ("7.1", "0xff(42)", "PTX 7.3"),
    ] {
        let parsed = ParsedModule::parse(&format!(".version {version} .target sm_80 .address_size 64 .global .u32 data; .global .u8 byte = {initializer}; .entry kernel() {{ ret; }}")).unwrap();
        assert!(
            parsed
                .compile("kernel")
                .unwrap_err()
                .to_string()
                .contains(expected)
        );
    }
}

#[test]
fn test_b128_declarations_require_ptx_83_in_all_scopes() {
    for declaration in [
        ".global .b128 g; .entry kernel() { ret; }",
        ".entry kernel(.param .b128 p) { ret; }",
        ".entry kernel() { .reg .b128 r; ret; }",
    ] {
        let parsed = ParsedModule::parse(&format!(
            ".version 7.0 .target sm_80 .address_size 64 {declaration}"
        ))
        .unwrap();
        assert!(
            parsed
                .compile("kernel")
                .unwrap_err()
                .to_string()
                .contains("PTX 8.3")
        );
        compile(declaration);
    }
}

#[test]
fn test_bitwise_constant_expressions_preserve_usual_integer_conversions() {
    let code = compile(".global .u32 negative = ((-1 & -1) < 0); .entry kernel() { ret; }");
    assert!(matches!(
        code.artifact.executable.module.variables[0].initial_values[0].value,
        InitialBits::Bits(1)
    ));
}

#[test]
fn test_constant_shift_counts_match_64_bit_constant_evaluation() {
    let code =
        compile(".global .u64 values[] = {1 << 64, (-1) >> 64, 1 << -1}; .entry kernel() { ret; }");
    let values: Vec<_> = code.artifact.executable.module.variables[0]
        .initial_values
        .iter()
        .map(|value| match value.value {
            InitialBits::Bits(bits) => u64::try_from(bits).unwrap(),
            InitialBits::Address(_) => panic!("bits"),
        })
        .collect();
    assert_eq!(values, [1, u64::MAX, 1 << 63]);
}

#[test]
fn test_logical_constant_operators_preserve_bitwise_precedence() {
    let code = compile(
        ".global .u32 values[] = {2 && 4, 0 || 2, 2 & 4 || 1, 1 | 2 && 0}; .entry kernel() { ret; }",
    );
    let values: Vec<_> = code.artifact.executable.module.variables[0]
        .initial_values
        .iter()
        .map(|value| match value.value {
            InitialBits::Bits(bits) => bits,
            InitialBits::Address(_) => panic!("bits"),
        })
        .collect();
    assert_eq!(values, [1, 1, 1, 0]);
}

#[test]
fn test_invalid_constant_subexpressions_are_diagnosed_before_selection() {
    reject(
        ".global .f64 x = 1.0 / 0.0; .entry kernel() { ret; }",
        "division by zero",
    );
    reject(
        ".global .u64 x = 0 && (1/0); .entry kernel() { ret; }",
        "division by zero",
    );
    reject(
        ".global .u64 x = 0 ? missing : 1; .entry kernel() { ret; }",
        "undefined symbol",
    );
}

#[test]
fn test_generic_memory_access_cannot_read_parameter_symbols() {
    reject(
        ".entry kernel(.param .u32 p) { .reg .u32 %r; ld.u32 %r, [p]; ret; }",
        "state space disagrees",
    );
}

#[test]
fn test_undefined_and_unsupported_callees_fail() {
    reject(
        ".extern .func external(); .entry kernel() { call external; ret; }",
        "unresolved external",
    );
    reject(
        ".func helper() { unsupported; ret; } .entry kernel() { call helper; ret; }",
        "unsupported",
    );
}

#[test]
fn test_maxntid_limits_the_product_not_individual_dimensions() {
    let code = compile(".entry kernel() .maxntid 256 { ret; }");
    assert_eq!(code.launch_requirements().maximum_threads(), Some(256));
    code.launch_requirements()
        .validate_block([16, 16, 1])
        .unwrap();
    assert!(
        code.launch_requirements()
            .validate_block([16, 32, 1])
            .is_err()
    );
    compile(".entry kernel() .maxntid 1,1,1024 { ret; }")
        .launch_requirements()
        .validate_block([1024, 1, 1])
        .unwrap();
}

#[test]
fn test_reqntid_requires_exact_dimensions_and_excludes_maxntid() {
    let code = compile(".entry kernel() .reqntid 16,16 { ret; }");
    code.launch_requirements()
        .validate_block([16, 16, 1])
        .unwrap();
    assert!(
        code.launch_requirements()
            .validate_block([256, 1, 1])
            .is_err()
    );
    reject(
        ".entry kernel() .reqntid 32 .maxntid 32 { ret; }",
        "cannot appear together",
    );
    reject(".entry kernel() .reqntid 0 { ret; }", "block dimensions");
}

#[test]
fn test_target_version_and_address_size_are_independent() {
    for header in [
        ".version 10.0 .target sm_80 .address_size 64",
        ".version 9.5 .target sm_80 .address_size 64",
        ".version 9.3 .target sm_90 .address_size 64",
    ] {
        let parsed = ParsedModule::parse(&format!("{header} .entry kernel() {{ ret; }}")).unwrap();
        assert!(matches!(
            parsed.compile("kernel").unwrap_err().kind,
            CompileErrorKind::Unsupported(_)
        ));
    }
    for size in [32, 64] {
        ParsedModule::parse(&format!(
            ".version 9.4 .target sm_80 .address_size {size} .entry kernel() {{ ret; }}"
        ))
        .unwrap()
        .compile("kernel")
        .unwrap();
    }
    let old = ParsedModule::parse(
        ".version 7.0 .target sm_80 .address_size 64 .entry kernel() { bar.cta.sync 0; ret; }",
    )
    .unwrap();
    assert!(
        old.compile("kernel")
            .unwrap_err()
            .to_string()
            .contains("PTX 7.8")
    );
}

#[test]
fn test_module_layout_preserves_sparse_rows_and_forward_relocations() {
    let code = compile(
        ".global .u32 rows[2][3] = {{1}, {2,3}}; .global .u64 pointer = generic(target)+4; .global .u32 target[8]; .entry kernel() { ret; }",
    );
    let variables = &code.artifact.executable.module.variables;
    assert_eq!(variables[0].layout.byte_len(), 24);
    assert_eq!(
        variables[0]
            .initial_values
            .iter()
            .map(|v| v.offset)
            .collect::<Vec<_>>(),
        [0, 12, 16]
    );
    assert!(
        matches!(variables[1].initial_values[0].value, InitialBits::Address(ref relocation) if relocation.target == ir::GlobalId(2) && relocation.offset == 4 && relocation.generic)
    );
    assert!(variables[2].initial_values.is_empty());
}

#[test]
fn test_constant_expression_evaluation_uses_ptx_signedness() {
    let code = compile(
        ".global .u64 values[] = {(-8 >> 1U), ((-8 % 3) < 0), ((~0) > 0), ((1 ? -1 : 1U) > 0), (0x10 / -2)}; .entry kernel() { ret; }",
    );
    let values: Vec<_> = code.artifact.executable.module.variables[0]
        .initial_values
        .iter()
        .map(|v| match v.value {
            InitialBits::Bits(bits) => u64::try_from(bits).unwrap(),
            InitialBits::Address(_) => panic!("bits"),
        })
        .collect();
    assert_eq!(values, [(-4i64) as u64, 0, 1, 1, (-8i64) as u64]);
}

#[test]
fn test_masks_extract_bytes_and_preserve_relocation_masks() {
    let code = compile(
        ".global .u64 address; .global .u8 bytes[] = {0xff00(131187), 0xff0000(generic(address)+8)}; .entry kernel() { ret; }",
    );
    let values = &code.artifact.executable.module.variables[1].initial_values;
    assert!(matches!(values[0].value, InitialBits::Bits(0)));
    assert!(
        matches!(values[1].value, InitialBits::Address(ref relocation) if relocation.mask == Some(0x00ff_0000) && relocation.offset == 8)
    );
    reject(
        ".global .u8 bad = 0x0f(42); .entry kernel() { ret; }",
        "one byte",
    );
}

#[test]
fn test_binary64_constant_expressions_and_comparisons_are_software_evaluated() {
    let code = compile(
        ".global .f32 value = 1.0000000596046448 + 0.0; .global .u32 comparison = (1.0 < 2.0); .entry kernel() { ret; }",
    );
    assert!(matches!(
        code.artifact.executable.module.variables[0].initial_values[0].value,
        InitialBits::Bits(0x3f80_0000)
    ));
    assert!(matches!(
        code.artifact.executable.module.variables[1].initial_values[0].value,
        InitialBits::Bits(1)
    ));
    reject(
        ".global .f32 value = 0f3f800000 + 0.0; .entry kernel() { ret; }",
        "floating-point expressions",
    );
}

#[test]
fn test_layout_overflow_is_rejected_before_allocation() {
    reject(
        ".global .u64 huge[4294967295][4294967295]; .entry kernel() { ret; }",
        "layout exceeds",
    );
    reject(
        ".global .u32 values[1] = {1,2}; .entry kernel() { ret; }",
        "too many",
    );
    reject(
        ".shared .u32 value = 1; .entry kernel() { ret; }",
        "shared variables cannot",
    );
}

#[test]
fn test_local_shared_and_parameter_templates_keep_distinct_ownership() {
    let code = compile(
        ".entry kernel(.param .u64 input) { .local .u32 local[3]; .shared .align 16 .b8 shared[64]; .extern .shared .align 16 .b8 dynamic[]; ret; }",
    );
    let objects = &code.artifact.executable.functions[0].objects;
    assert_eq!(objects[0].role, FrameRole::KernelInput);
    assert_eq!(objects[1].space, StateSpace::Local);
    assert_eq!(objects[1].layout.byte_len(), 12);
    assert_eq!(objects[2].space, StateSpace::Shared);
    assert!(objects[3].dynamic);
    assert_eq!(objects[3].layout.byte_len(), 0);
}

#[test]
fn test_illegal_state_space_writes_and_foreign_space_symbols_fail() {
    reject(
        ".const .u32 c; .entry kernel() { st.u32 [c], 0; ret; }",
        "read-only",
    );
    reject(
        ".entry kernel(.param .u32 p) { st.param.u32 [p], 0; ret; }",
        "read-only",
    );
    reject(
        ".global .u32 g; .entry kernel() { .reg .u32 %r; ld.shared.u32 %r, [g]; ret; }",
        "state space disagrees",
    );
}

#[test]
fn test_source_positions_do_not_split_utf8_codepoints() {
    assert_eq!(
        SourcePosition::from_offset("a\néx", 3),
        SourcePosition { line: 2, column: 1 }
    );
    let parsed = parse("// λ\n.entry kernel() { unsupported; }");
    let error = parsed.compile("kernel").unwrap_err();
    assert_eq!(error.span.start_position(parsed.source()).line, 5);
}

#[test]
fn test_compilation_cache_shares_successful_artifacts_without_owning_the_source() {
    let source =
        ParsedModule::parse(".version 9.4 .target sm_80 .address_size 64 .entry k() {ret;}")
            .unwrap();
    let weak_source = Arc::downgrade(&source.inner);
    let first = source.compile("k").unwrap();
    let second = source.compile("k").unwrap();
    assert!(Arc::ptr_eq(&first.artifact, &second.artifact));
    assert!(source.compile("missing").is_err());
    assert_eq!(source.inner.compiled.lock().unwrap().len(), 1);
    let weak_artifact = Arc::downgrade(&first.artifact);
    drop((first, second, source));
    assert!(weak_source.upgrade().is_none());
    assert!(weak_artifact.upgrade().is_none());
}

#[test]
fn test_concurrent_compilation_returns_one_cached_artifact() {
    let source =
        ParsedModule::parse(".version 9.4 .target sm_80 .address_size 64 .entry k() {ret;}")
            .unwrap();
    let barrier = Barrier::new(8);
    let kernels = thread::scope(|scope| {
        let workers: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    barrier.wait();
                    source.compile("k").unwrap()
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert!(
        kernels
            .iter()
            .all(|kernel| Arc::ptr_eq(&kernels[0].artifact, &kernel.artifact))
    );
}
