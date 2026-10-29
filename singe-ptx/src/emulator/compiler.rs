use std::collections::{BTreeMap, BTreeSet};

use crate::{
    ast::{
        self, DirectiveStatement, FunctionDirective, Immediate, Operand, ScalarType, Span,
        StateSpace, Statement, TopLevelItem, VariableDecl,
    },
    emulator::{
        constants,
        error::{CompileError, CompileErrorKind, Result},
        ir::{
            self, AddressBase, Axis, Destination, FrameId, FrameObject, FrameRole, FunctionId,
            GlobalId, LaneMask, MemoryRef, Predicate, ProgramCounter, RegisterId, Source,
            SpecialRegister,
        },
        layout::{self, ModuleLayout, ParameterLayout, StorageLayout},
        lowering,
        target::{self, MAX_FUNCTION_REGISTERS, PtxVersion},
    },
};

/// Block dimensions a kernel accepts, from its `.reqntid` and `.maxntid` directives
/// and the `sm_80` limits.
#[derive(Debug, Clone)]
pub struct LaunchRequirements {
    required: Option<[u32; 3]>,
    maximum: Option<u64>,
    span: Span,
}

impl LaunchRequirements {
    /// Returns the block dimensions required by `.reqntid`, if present.
    #[must_use]
    pub fn required_block(&self) -> Option<[u32; 3]> {
        self.required
    }

    /// Returns the maximum threads per CTA allowed by `.maxntid`, if present.
    ///
    /// The `sm_80` limit of 1024 threads still applies when this is larger.
    #[must_use]
    pub fn maximum_threads(&self) -> Option<u64> {
        self.maximum
    }

    /// Checks that the kernel can launch with CTAs of `dimensions` threads.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is zero or exceeds the `sm_80` limits, or if
    /// `dimensions` violates `.reqntid` or `.maxntid`.
    pub fn validate_block(&self, dimensions: [u32; 3]) -> Result<()> {
        target::validate_block(dimensions, self.span)?;
        if self.required.is_some_and(|required| required != dimensions)
            || self.maximum.is_some_and(|maximum| {
                dimensions.into_iter().map(u64::from).product::<u64>() > maximum
            })
        {
            return Err(CompileError::new(
                self.span,
                CompileErrorKind::InvalidLaunch(
                    "block dimensions violate entry-point directives".into(),
                ),
            ));
        }
        Ok(())
    }
}

pub fn compile(module: &ast::Module, entry: &str) -> Result<(ir::Executable, LaunchRequirements)> {
    let layout = ModuleLayout::create(module)?;
    let mut definitions: BTreeMap<&str, &ast::Function> = BTreeMap::new();
    for item in &module.items {
        if let TopLevelItem::Function(function) = item {
            if layout.symbols.contains_key(&function.name) {
                return Err(CompileError::new(
                    function.span,
                    CompileErrorKind::DuplicateSymbol(function.name.clone()),
                ));
            }
            if let Some(previous) = definitions.get(function.name.as_str()) {
                if (previous.body.is_some() && function.body.is_some())
                    || !same_signature(previous, function)
                {
                    return Err(CompileError::new(
                        function.span,
                        CompileErrorKind::DuplicateSymbol(function.name.clone()),
                    ));
                }
                if previous.body.is_some() {
                    continue;
                }
            }
            definitions.insert(function.name.as_str(), function);
        }
    }
    let entry_function = definitions
        .get(entry)
        .copied()
        .filter(|function| function.entry && function.body.is_some())
        .ok_or_else(|| {
            CompileError::new(
                module.span,
                CompileErrorKind::InvalidEntry(entry.to_owned()),
            )
        })?;
    let launch = requirements(entry_function)?;
    let mut linker = Linker {
        definitions,
        ids: BTreeMap::new(),
        pending: Vec::new(),
    };
    linker
        .ids
        .insert(entry_function.name.as_str(), FunctionId(0));
    linker.pending.push(entry_function);
    let mut functions = Vec::new();
    while functions.len() < linker.pending.len() {
        let function = linker.pending[functions.len()];
        requirements(function)?;
        let compiled =
            FunctionCompiler::create(function, &layout, &mut linker, PtxVersion::of(module))?
                .compile()?;
        functions.push(compiled);
    }
    Ok((
        ir::Executable {
            entry: FunctionId(0),
            functions,
            module: layout,
        },
        launch,
    ))
}

fn same_signature(a: &ast::Function, b: &ast::Function) -> bool {
    fn equal(a: &[ast::Parameter], b: &[ast::Parameter]) -> bool {
        a.len() == b.len()
            && a.iter().zip(b).all(|(a, b)| {
                a.ty == b.ty
                    && a.state_space == b.state_space
                    && a.alignment == b.alignment
                    && a.array_bounds == b.array_bounds
                    && a.pointer == b.pointer
            })
    }
    a.entry == b.entry && equal(&a.params, &b.params) && equal(&a.return_params, &b.return_params)
}

fn noreturn(function: &ast::Function) -> bool {
    function
        .directives
        .iter()
        .any(|directive| matches!(directive, FunctionDirective::NoReturn { .. }))
}

/// Pads the components of a `.reqntid` or `.maxntid` directive to three dimensions.
fn thread_dimensions(function: &ast::Function, values: &[u32], span: Span) -> Result<[u32; 3]> {
    if !function.entry || values.is_empty() || values.len() > 3 {
        return Err(CompileError::new(
            span,
            CompileErrorKind::InvalidLaunch(
                "thread dimensions require an entry point and one to three components".into(),
            ),
        ));
    }
    let mut dimensions = [1; 3];
    dimensions[..values.len()].copy_from_slice(values);
    Ok(dimensions)
}

fn requirements(function: &ast::Function) -> Result<LaunchRequirements> {
    let mut result = LaunchRequirements {
        required: None,
        maximum: None,
        span: function.span,
    };
    let mut seen = BTreeSet::new();
    if !function.attributes.is_empty() {
        return Err(CompileError::unsupported(
            function.span,
            "function resource attributes",
        ));
    }
    if function.entry && !function.return_params.is_empty() {
        return Err(CompileError::declaration(
            function.span,
            "entry points cannot return parameters",
        ));
    }
    for directive in &function.directives {
        let (name, span) = match directive {
            FunctionDirective::ReqNTid { span, values }
            | FunctionDirective::MaxNTid { span, values } => {
                let dimensions = thread_dimensions(function, values, *span)?;
                if matches!(directive, FunctionDirective::ReqNTid { .. }) {
                    target::validate_block(dimensions, *span)?;
                    result.required = Some(dimensions);
                    ("reqntid", *span)
                } else {
                    let count = dimensions
                        .into_iter()
                        .map(u64::from)
                        .try_fold(1u64, u64::checked_mul)
                        .filter(|count| *count > 0)
                        .ok_or_else(|| {
                            CompileError::declaration(
                                *span,
                                "maxntid requires a positive total representable as u64",
                            )
                        })?;
                    result.maximum = Some(count);
                    ("maxntid", *span)
                }
            }
            FunctionDirective::MaxNReg { span, value }
            | FunctionDirective::MinNCtaPerSm { span, value } => {
                if *value == 0 || !function.entry {
                    return Err(CompileError::declaration(
                        *span,
                        "resource hints require positive values on an entry point",
                    ));
                }
                if matches!(directive, FunctionDirective::MaxNReg { .. }) {
                    ("maxnreg", *span)
                } else {
                    ("minnctapersm", *span)
                }
            }
            FunctionDirective::NoReturn { span } => {
                if function.entry || !function.return_params.is_empty() {
                    return Err(CompileError::declaration(
                        *span,
                        "noreturn requires a device function without return parameters",
                    ));
                }
                ("noreturn", *span)
            }
            FunctionDirective::Pragma { span, value } => {
                layout::validate_pragma(value, *span)?;
                continue;
            }
            FunctionDirective::AbiPreserve { span, .. }
            | FunctionDirective::AbiPreserveControl { span, .. } => {
                return Err(CompileError::unsupported(
                    *span,
                    "ABI preservation directives",
                ));
            }
        };
        if !seen.insert(name) {
            return Err(CompileError::declaration(
                span,
                format!("duplicate .{name} directive"),
            ));
        }
    }
    if result.required.is_some() && result.maximum.is_some() {
        return Err(CompileError::declaration(
            function.span,
            "reqntid and maxntid cannot appear together",
        ));
    }
    Ok(result)
}

/// Where a frame object or module variable lives and how it is laid out.
#[derive(Debug, Clone, Copy)]
pub struct ObjectProperties {
    pub space: StateSpace,
    pub layout: StorageLayout,
    pub role: FrameRole,
}

struct Linker<'a> {
    definitions: BTreeMap<&'a str, &'a ast::Function>,
    ids: BTreeMap<&'a str, FunctionId>,
    pending: Vec<&'a ast::Function>,
}

#[derive(Debug, Clone, Copy)]
enum Symbol {
    Register(RegisterId),
    Memory(MemoryRef),
    Label(ProgramCounter),
}

struct Scope {
    parent: Option<usize>,
    symbols: BTreeMap<String, Symbol>,
}

pub struct FunctionCompiler<'a, 'b> {
    function: &'a ast::Function,
    module: &'b ModuleLayout,
    linker: &'b mut Linker<'a>,
    scopes: Vec<Scope>,
    registers: Vec<ir::RegisterLayout>,
    objects: Vec<FrameObject>,
    raw: Vec<(&'a ast::Instruction, usize)>,
    parameters: Vec<ParameterLayout>,
    returns: Vec<ParameterLayout>,
    version: PtxVersion,
    scope: usize,
}

impl<'a, 'b> FunctionCompiler<'a, 'b> {
    fn create(
        function: &'a ast::Function,
        module: &'b ModuleLayout,
        linker: &'b mut Linker<'a>,
        version: PtxVersion,
    ) -> Result<Self> {
        for parameter in function.params.iter().chain(&function.return_params) {
            layout::validate_type_version(parameter.ty, version, parameter.span)?;
        }
        let parameters = layout::parameters(&function.params, module.address_size)?;
        let returns = layout::parameters(&function.return_params, module.address_size)?;
        let mut result = Self {
            function,
            module,
            linker,
            scopes: vec![Scope {
                parent: None,
                symbols: BTreeMap::new(),
            }],
            registers: Vec::new(),
            objects: Vec::new(),
            raw: Vec::new(),
            parameters,
            returns,
            version,
            scope: 0,
        };
        let input_role = if function.entry {
            FrameRole::KernelInput
        } else {
            FrameRole::Input
        };
        for (role, parameters) in [
            (input_role, result.parameters.clone()),
            (FrameRole::Return, result.returns.clone()),
        ] {
            for parameter in parameters {
                let object = result.add_object(StateSpace::Param, parameter.layout, role, false);
                result.insert(0, &parameter.name, Symbol::Memory(object), function.span)?;
            }
        }
        let body = function.body.as_deref().ok_or_else(|| {
            CompileError::unsupported(function.span, "undefined external function")
        })?;
        result.collect(body, 0)?;
        Ok(result)
    }

    fn insert(&mut self, scope: usize, name: &str, symbol: Symbol, span: Span) -> Result<()> {
        if self.scopes[scope]
            .symbols
            .insert(name.to_owned(), symbol)
            .is_some()
        {
            return Err(CompileError::new(
                span,
                CompileErrorKind::DuplicateSymbol(name.to_owned()),
            ));
        }
        Ok(())
    }

    fn add_object(
        &mut self,
        space: StateSpace,
        layout: StorageLayout,
        role: FrameRole,
        dynamic: bool,
    ) -> MemoryRef {
        let id = FrameId(self.objects.len());
        self.objects.push(FrameObject {
            space,
            layout,
            role,
            dynamic,
        });
        MemoryRef::Frame(id)
    }

    fn collect(&mut self, statements: &'a [Statement], scope: usize) -> Result<()> {
        for statement in statements {
            match statement {
                Statement::Instruction(instruction) => self.raw.push((instruction, scope)),
                Statement::Label { span, name } => self.insert(
                    scope,
                    name,
                    Symbol::Label(ProgramCounter(self.raw.len())),
                    *span,
                )?,
                Statement::Block { statements, .. } => {
                    let child = self.scopes.len();
                    self.scopes.push(Scope {
                        parent: Some(scope),
                        symbols: BTreeMap::new(),
                    });
                    self.collect(statements, child)?;
                }
                Statement::Variable(variable) => {
                    layout::validate_type_version(variable.ty, self.version, variable.span)?;
                    if variable.state_space == Some(StateSpace::Reg) {
                        self.declare_registers(variable, scope)?;
                    } else {
                        self.declare_storage(variable, scope)?;
                    }
                }
                Statement::Directive(directive) => match directive {
                    DirectiveStatement::Loc(_) => {}
                    DirectiveStatement::Pragma { span, value } => {
                        layout::validate_pragma(value, *span)?;
                    }
                    DirectiveStatement::BranchTargets { span, .. }
                    | DirectiveStatement::CallTargets { span, .. }
                    | DirectiveStatement::CallPrototype { span, .. } => {
                        return Err(CompileError::unsupported(
                            *span,
                            "indirect branch or call directives",
                        ));
                    }
                },
            }
        }
        Ok(())
    }

    /// Declares a `.reg` variable, or each register of a `<n>` register range.
    fn declare_registers(&mut self, variable: &'a VariableDecl, scope: usize) -> Result<()> {
        if variable.initializer.is_some()
            || variable.alignment.is_some()
            || variable.vector.is_some()
            || !variable.array_bounds.is_empty()
            || variable.linkage.is_some()
            || !variable.attributes.is_empty()
        {
            return Err(CompileError::declaration(
                variable.span,
                "invalid register declaration",
            ));
        }
        if matches!(
            variable.ty,
            ScalarType::B8
                | ScalarType::U8
                | ScalarType::S8
                | ScalarType::Tf32
                | ScalarType::Bf16
                | ScalarType::TexRef
                | ScalarType::SamplerRef
                | ScalarType::SurfRef
        ) {
            return Err(CompileError::unsupported(
                variable.span,
                format!("register type .{}", variable.ty.suffix()),
            ));
        }
        let count = variable.count.unwrap_or(1);
        if count == 0
            || (count as usize)
                .checked_add(self.registers.len())
                .is_none_or(|total| total > MAX_FUNCTION_REGISTERS)
        {
            return Err(CompileError::declaration(
                variable.span,
                format!(
                    "register declarations exceed the frontend limit of {MAX_FUNCTION_REGISTERS} slots per function"
                ),
            ));
        }
        for index in 0..count {
            let name = if variable.count.is_some() {
                format!("{}{index}", variable.name)
            } else {
                variable.name.clone()
            };
            let register = RegisterId(self.registers.len());
            self.registers.push(ir::RegisterLayout {
                name: name.clone(),
                ty: variable.ty,
                span: variable.span,
            });
            self.insert(scope, &name, Symbol::Register(register), variable.span)?;
        }
        Ok(())
    }

    /// Declares a function-scope `.param`, `.local`, or `.shared` variable.
    fn declare_storage(&mut self, variable: &'a VariableDecl, scope: usize) -> Result<()> {
        let space = variable
            .state_space
            .filter(|space| {
                matches!(
                    space,
                    StateSpace::Param | StateSpace::Local | StateSpace::Shared
                )
            })
            .ok_or_else(|| {
                CompileError::unsupported(variable.span, "function variable state space")
            })?;
        let dynamic = layout::is_dynamic_shared(variable);
        if variable.initializer.is_some()
            || variable.count.is_some()
            || !variable.attributes.is_empty()
            || (variable.linkage.is_some() && !dynamic)
        {
            return Err(CompileError::declaration(
                variable.span,
                "invalid function storage declaration",
            ));
        }
        let layout = layout::variable_layout(variable)?;
        let object = self.add_object(space, layout, FrameRole::Variable, dynamic);
        self.insert(scope, &variable.name, Symbol::Memory(object), variable.span)?;
        Ok(())
    }

    fn compile(mut self) -> Result<ir::Function> {
        let mut instructions = Vec::with_capacity(self.raw.len());
        for index in 0..self.raw.len() {
            let (instruction, scope) = self.raw[index];
            self.scope = scope;
            let guard = instruction
                .guard
                .as_ref()
                .map(|guard| self.named_predicate(&guard.register, guard.negated, guard.span))
                .transpose()?;
            let operation = lowering::lower(&mut self, instruction)?;
            instructions.push(ir::Instruction {
                span: instruction.span,
                guard,
                operation,
            });
        }
        validate_control_flow(&instructions, self.function.span)?;
        Ok(ir::Function {
            name: self.function.name.clone(),
            noreturn: noreturn(self.function),
            registers: self.registers,
            objects: self.objects,
            parameters: self.parameters,
            returns: self.returns,
            instructions,
        })
    }

    fn find_symbol(&self, name: &str) -> Option<Symbol> {
        let mut scope = Some(self.scope);
        while let Some(index) = scope {
            if let Some(symbol) = self.scopes[index].symbols.get(name) {
                return Some(*symbol);
            }
            scope = self.scopes[index].parent;
        }
        self.module
            .symbols
            .get(name)
            .map(|id| Symbol::Memory(MemoryRef::Global(*id)))
    }

    fn symbol(&self, name: &str, span: Span) -> Result<Symbol> {
        self.find_symbol(name).ok_or_else(|| {
            CompileError::new(span, CompileErrorKind::UndefinedSymbol(name.to_owned()))
        })
    }

    fn register(
        &self,
        name: &str,
        expected: ScalarType,
        span: Span,
        width: Width,
    ) -> Result<RegisterId> {
        let Symbol::Register(id) = self.symbol(name, span)? else {
            return Err(CompileError::new(
                span,
                CompileErrorKind::TypeMismatch(format!("`{name}` is not a register")),
            ));
        };
        let declared = self.registers[id.0].ty;
        if !compatible(declared, expected, width) {
            return Err(CompileError::new(
                span,
                CompileErrorKind::TypeMismatch(format!(
                    "`{name}` is .{}, expected .{}",
                    declared.suffix(),
                    expected.suffix()
                )),
            ));
        }
        Ok(id)
    }

    fn named_predicate(&self, name: &str, negated: bool, span: Span) -> Result<Predicate> {
        Ok(Predicate::Register {
            register: self.register(name, ScalarType::Pred, span, Width::Exact)?,
            negated,
        })
    }

    pub(crate) fn predicate(&self, operand: &Operand) -> Result<Predicate> {
        let (operand, negated) = match operand {
            Operand::Negated { operand, .. } => (operand.as_ref(), true),
            operand => (operand, false),
        };
        match operand {
            Operand::Symbol { name, span } => self.named_predicate(name, negated, *span),
            Operand::Immediate {
                value: Immediate::Integer(value),
                ..
            } => Ok(Predicate::Constant((*value != 0) != negated)),
            Operand::Immediate {
                value: Immediate::UnsignedInteger(value),
                ..
            } => Ok(Predicate::Constant((*value != 0) != negated)),
            _ => Err(CompileError::instruction(
                operand.span(),
                "expected a predicate register",
            )),
        }
    }

    pub(crate) fn destination(&self, operand: &Operand, ty: ScalarType) -> Result<Destination> {
        self.destination_with(operand, ty, Width::Exact)
    }

    /// Resolves a destination that may be a register wider than `ty`, as `ld` and `cvt` allow.
    pub(crate) fn wide_destination(
        &self,
        operand: &Operand,
        ty: ScalarType,
    ) -> Result<Destination> {
        self.destination_with(operand, ty, Width::AtLeast)
    }

    fn destination_with(
        &self,
        operand: &Operand,
        ty: ScalarType,
        width: Width,
    ) -> Result<Destination> {
        match operand {
            Operand::Symbol { name, span } => Ok(Destination::Register(
                self.register(name, ty, *span, width)?,
            )),
            Operand::BitBucket { .. } => Ok(Destination::Discard),
            _ => Err(CompileError::instruction(
                operand.span(),
                "expected a destination register or bit bucket",
            )),
        }
    }

    fn validate_float_register(
        &self,
        id: RegisterId,
        format: ir::FloatFormat,
        span: Span,
    ) -> Result<()> {
        use ScalarType::{B16, B32, F16, F16x2};
        use ir::FloatFormat;
        let declared = self.registers[id.0].ty;
        let valid = match format {
            FloatFormat::F16 => matches!(declared, F16 | B16),
            FloatFormat::Bf16 => declared == B16,
            FloatFormat::F16x2 => matches!(declared, F16x2 | B32),
            FloatFormat::Bf16x2 => declared == B32,
            FloatFormat::F32 | FloatFormat::F64 => true,
        };
        if valid {
            Ok(())
        } else {
            Err(CompileError::instruction(
                span,
                "register type does not match the floating format",
            ))
        }
    }

    pub(crate) fn float_source(
        &self,
        operand: &Operand,
        format: ir::FloatFormat,
    ) -> Result<Source> {
        let source = self.source(operand, format.storage())?;
        if format.half() {
            let Source::Register(id) = source else {
                return Err(CompileError::instruction(
                    operand.span(),
                    "half arithmetic requires register operands",
                ));
            };
            self.validate_float_register(id, format, operand.span())?;
        }
        Ok(source)
    }

    pub(crate) fn float_destination(
        &self,
        operand: &Operand,
        format: ir::FloatFormat,
    ) -> Result<Destination> {
        let destination = self.destination(operand, format.storage())?;
        if format.half() {
            let Destination::Register(id) = destination else {
                return Err(CompileError::instruction(
                    operand.span(),
                    "half arithmetic requires a destination register",
                ));
            };
            self.validate_float_register(id, format, operand.span())?;
        }
        Ok(destination)
    }

    pub(crate) fn source(&self, operand: &Operand, ty: ScalarType) -> Result<Source> {
        self.source_with(operand, ty, Width::Exact)
    }

    /// Resolves a source that may be a register wider than `ty`, as `st` allows.
    pub(crate) fn wide_source(&self, operand: &Operand, ty: ScalarType) -> Result<Source> {
        self.source_with(operand, ty, Width::AtLeast)
    }

    pub(crate) fn reject_special_registers(&self, operand: &Operand) -> Result<()> {
        match operand {
            Operand::Symbol { name, span }
                if special_register(name).is_some() && self.find_symbol(name).is_none() =>
            {
                Err(CompileError::instruction(
                    *span,
                    "special registers require mov or cvt",
                ))
            }
            Operand::Vector { elements, .. } | Operand::Tuple { elements, .. } => elements
                .iter()
                .try_for_each(|element| self.reject_special_registers(element)),
            Operand::Pair { first, second, .. } => {
                self.reject_special_registers(first)?;
                self.reject_special_registers(second)
            }
            Operand::Negated { operand, .. } => self.reject_special_registers(operand),
            _ => Ok(()),
        }
    }

    fn source_with(&self, operand: &Operand, ty: ScalarType, width: Width) -> Result<Source> {
        match operand {
            Operand::Symbol { name, span } => {
                if let Some(special) = special_register(name)
                    && self.find_symbol(name).is_none()
                {
                    if !compatible(special.ty(), ty, Width::Exact) {
                        return Err(CompileError::instruction(
                            *span,
                            "special register source type does not match its declared or legacy width",
                        ));
                    }
                    return Ok(Source::Special(special));
                }
                if name == "WARP_SZ" {
                    let bits = constants::Constant::Signed(32).bits(ty, *span)?;
                    return Ok(Source::Immediate(if ty == ScalarType::Pred {
                        1
                    } else {
                        bits
                    }));
                }
                Ok(Source::Register(self.register(name, ty, *span, width)?))
            }
            Operand::Immediate { span, value } => {
                let bits = match value {
                    Immediate::Integer(value) => {
                        constants::Constant::Signed(*value).bits(ty, *span)?
                    }
                    Immediate::UnsignedInteger(value) => {
                        constants::Constant::Unsigned(*value).bits(ty, *span)?
                    }
                    Immediate::Float(value) => {
                        constants::float_constant(value, *span)?.bits(ty, *span)?
                    }
                };
                if ty == ScalarType::Pred {
                    return Ok(Source::Immediate(u128::from(bits != 0)));
                }
                let width = layout::scalar_bytes(ty, *span)? * 8;
                Ok(Source::Immediate(if width == 128 {
                    bits
                } else {
                    bits & ((1u128 << width) - 1)
                }))
            }
            _ => Err(CompileError::instruction(
                operand.span(),
                "expected a scalar source",
            )),
        }
    }

    pub(crate) fn move_source(&self, operand: &Operand, ty: ScalarType) -> Result<Source> {
        if let Operand::Symbol { name, .. } = operand
            && let Some(special) = special_register(name)
            && special.legacy_type(ty)
            && self.find_symbol(name).is_none()
        {
            return Ok(Source::Special(special));
        }
        if let Operand::Symbol { name, span } = operand
            && name != "WARP_SZ"
            && let Some(Symbol::Memory(object)) = self.find_symbol(name)
        {
            if !matches!(
                ty,
                ScalarType::U32 | ScalarType::U64 | ScalarType::B32 | ScalarType::B64
            ) {
                return Err(CompileError::instruction(
                    *span,
                    "address values require a 32-bit or 64-bit integer destination",
                ));
            }
            return Ok(Source::Address { object, offset: 0 });
        }
        self.source(operand, ty)
    }

    pub(crate) fn conversion_source(&self, operand: &Operand, ty: ScalarType) -> Result<Source> {
        if let Operand::Symbol { name, .. } = operand
            && let Some(special) = special_register(name)
            && special.legacy_type(ty)
            && self.find_symbol(name).is_none()
        {
            return Ok(Source::Special(special));
        }
        self.source_with(operand, ty, Width::AtLeast)
    }

    pub(crate) fn object(&self, operand: &Operand) -> Result<MemoryRef> {
        match operand {
            Operand::Symbol { name, span } => match self.symbol(name, *span)? {
                Symbol::Memory(object) => Ok(object),
                _ => Err(CompileError::instruction(
                    *span,
                    "expected an addressable parameter",
                )),
            },
            _ => Err(CompileError::instruction(
                operand.span(),
                "expected a memory symbol",
            )),
        }
    }

    pub(crate) fn object_properties(&self, object: MemoryRef) -> ObjectProperties {
        match object {
            MemoryRef::Frame(id) => {
                let object = &self.objects[id.0];
                ObjectProperties {
                    space: object.space,
                    layout: object.layout,
                    role: object.role,
                }
            }
            MemoryRef::Global(GlobalId(index)) => {
                let object = &self.module.variables[index];
                ObjectProperties {
                    space: object.space,
                    layout: object.layout,
                    role: FrameRole::Variable,
                }
            }
        }
    }

    pub(crate) fn address(
        &self,
        operand: &Operand,
        space: Option<StateSpace>,
        write: bool,
    ) -> Result<ir::Address> {
        let Operand::Address { span, base, offset } = operand else {
            return Err(CompileError::instruction(
                operand.span(),
                "expected a bracketed memory address",
            ));
        };
        let base = match self.symbol(base, *span)? {
            Symbol::Register(id) => {
                let ty = self.registers[id.0].ty;
                let valid = match space {
                    Some(StateSpace::Param) => false,
                    Some(StateSpace::Shared | StateSpace::Local) => matches!(
                        ty,
                        ScalarType::B32
                            | ScalarType::U32
                            | ScalarType::S32
                            | ScalarType::B64
                            | ScalarType::U64
                            | ScalarType::S64
                    ),
                    _ => match self.module.address_size {
                        ast::AddressSize::Bits32 => {
                            matches!(ty, ScalarType::B32 | ScalarType::U32 | ScalarType::S32)
                        }
                        ast::AddressSize::Bits64 => {
                            matches!(ty, ScalarType::B64 | ScalarType::U64 | ScalarType::S64)
                        }
                    },
                };
                if !valid {
                    return Err(CompileError::instruction(
                        *span,
                        "invalid address register width or type",
                    ));
                }
                AddressBase::Register(id)
            }
            Symbol::Memory(object) => {
                let ObjectProperties {
                    space: actual,
                    role,
                    ..
                } = self.object_properties(object);
                if space.is_some_and(|space| space != actual)
                    || (space.is_none() && actual == StateSpace::Param)
                {
                    return Err(CompileError::instruction(
                        *span,
                        "address state space disagrees with the instruction",
                    ));
                }
                if write
                    && (actual == StateSpace::Const
                        || matches!(role, FrameRole::KernelInput | FrameRole::Input))
                {
                    return Err(CompileError::instruction(
                        *span,
                        "cannot write read-only storage",
                    ));
                }
                AddressBase::Object(object)
            }
            Symbol::Label(_) => {
                return Err(CompileError::instruction(
                    *span,
                    "label is not a memory address",
                ));
            }
        };
        Ok(ir::Address {
            base,
            offset: *offset,
        })
    }

    pub(crate) fn branch_target(&self, operand: &Operand) -> Result<ProgramCounter> {
        let Operand::Symbol { name, span } = operand else {
            return Err(CompileError::instruction(
                operand.span(),
                "branch requires a direct label",
            ));
        };
        match self.symbol(name, *span)? {
            Symbol::Label(pc) if pc.0 < self.raw.len() => Ok(pc),
            _ => Err(CompileError::instruction(*span, "invalid branch target")),
        }
    }

    pub(crate) fn call(
        &mut self,
        operand: &Operand,
        arguments: &[Operand],
        returns: &[Operand],
    ) -> Result<ir::Operation> {
        let Operand::Symbol { name, span } = operand else {
            return Err(CompileError::unsupported(
                operand.span(),
                "indirect function call",
            ));
        };
        let function = self
            .linker
            .definitions
            .get(name.as_str())
            .copied()
            .ok_or_else(|| {
                CompileError::new(*span, CompileErrorKind::UndefinedSymbol(name.clone()))
            })?;
        if function.entry || function.body.is_none() {
            return Err(CompileError::unsupported(
                *span,
                "call to an entry point or unresolved external function",
            ));
        }
        let arguments = self.call_arguments(
            arguments,
            &layout::parameters(&function.params, self.module.address_size)?,
            *span,
        )?;
        let returns = self.call_arguments(
            returns,
            &layout::parameters(&function.return_params, self.module.address_size)?,
            *span,
        )?;
        let id = if let Some(id) = self.linker.ids.get(name.as_str()) {
            *id
        } else {
            let id = FunctionId(self.linker.pending.len());
            self.linker.ids.insert(function.name.as_str(), id);
            self.linker.pending.push(function);
            id
        };
        Ok(ir::Operation::Control(ir::ControlOperation::Call {
            function: id,
            noreturn: noreturn(function),
            arguments,
            returns,
        }))
    }

    fn call_arguments(
        &self,
        operands: &[Operand],
        parameters: &[ParameterLayout],
        span: Span,
    ) -> Result<Vec<MemoryRef>> {
        if operands.len() != parameters.len() {
            return Err(CompileError::instruction(
                span,
                "call argument count does not match its declaration",
            ));
        }
        operands
            .iter()
            .zip(parameters)
            .map(|(operand, parameter)| {
                let object = self.object(operand)?;
                let ObjectProperties {
                    space,
                    layout,
                    role,
                } = self.object_properties(object);
                if space != StateSpace::Param
                    || layout.byte_len() != parameter.layout.byte_len()
                    || layout.alignment() < parameter.layout.alignment()
                    || role != FrameRole::Variable
                {
                    return Err(CompileError::instruction(
                        operand.span(),
                        "call parameter storage does not match its declaration",
                    ));
                }
                Ok(object)
            })
            .collect()
    }

    pub(crate) fn require_version(&self, minimum: PtxVersion, span: Span) -> Result<()> {
        if self.version < minimum {
            return Err(CompileError::unsupported(
                span,
                format!("instruction form requires PTX {minimum}"),
            ));
        }
        Ok(())
    }
}

/// How a register's declared width must relate to the operand type an instruction expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Width {
    /// The register has exactly the expected width.
    Exact,
    /// The register may be wider, as for `ld`, `st`, and `cvt` operands.
    AtLeast,
}

/// Returns whether a register declared as `declared` can serve as an operand of type
/// `expected`. Bit types match any same-width type.
pub fn compatible(declared: ScalarType, expected: ScalarType, width: Width) -> bool {
    use ScalarType::{Bf16, Pred, SamplerRef, SurfRef, TexRef, Tf32};
    if declared == Pred || expected == Pred {
        return declared == expected;
    }
    let unsupported = |ty| matches!(ty, Bf16 | Tf32 | TexRef | SamplerRef | SurfRef);
    if unsupported(declared) || unsupported(expected) {
        return false;
    }
    let (actual, required) = (declared.bit_width(), expected.bit_width());
    let categories = declared.is_bits()
        || expected.is_bits()
        || declared == expected
        || (!declared.is_float() && !expected.is_float());
    categories
        && (actual == required
            || (width == Width::AtLeast
                && actual > required
                && (!expected.is_float() || declared.is_bits())))
}

fn validate_control_flow(instructions: &[ir::Instruction], span: Span) -> Result<()> {
    let mut visited = vec![false; instructions.len()];
    let mut pending = vec![0];
    while let Some(pc) = pending.pop() {
        let Some(instruction) = instructions.get(pc) else {
            return Err(CompileError::declaration(
                span,
                "function may fall through its end",
            ));
        };
        if visited[pc] {
            continue;
        }
        visited[pc] = true;
        if let ir::Operation::Control(ir::ControlOperation::Branch { target }) =
            instruction.operation
        {
            pending.push(target.0);
        }
        if instruction.guard.is_some()
            || !matches!(
                instruction.operation,
                ir::Operation::Control(
                    ir::ControlOperation::Branch { .. }
                        | ir::ControlOperation::Return
                        | ir::ControlOperation::Exit
                        | ir::ControlOperation::Trap
                        | ir::ControlOperation::Call { noreturn: true, .. }
                )
            )
        {
            pending.push(pc + 1);
        }
    }
    Ok(())
}

fn special_register(name: &str) -> Option<SpecialRegister> {
    match name {
        "%laneid" => Some(SpecialRegister::LaneId),
        "%warpid" => Some(SpecialRegister::WarpId),
        "%nwarpid" => Some(SpecialRegister::WarpCount),
        "%smid" => Some(SpecialRegister::SmId),
        "%nsmid" => Some(SpecialRegister::SmCount),
        "%gridid" => Some(SpecialRegister::GridId),
        "%lanemask_eq" => Some(SpecialRegister::LaneMask(LaneMask::Equal)),
        "%lanemask_le" => Some(SpecialRegister::LaneMask(LaneMask::LessEqual)),
        "%lanemask_lt" => Some(SpecialRegister::LaneMask(LaneMask::Less)),
        "%lanemask_ge" => Some(SpecialRegister::LaneMask(LaneMask::GreaterEqual)),
        "%lanemask_gt" => Some(SpecialRegister::LaneMask(LaneMask::Greater)),
        _ => {
            let (prefix, component) = name.split_once('.')?;
            let axis = match component {
                "x" => Axis::X,
                "y" => Axis::Y,
                "z" => Axis::Z,
                _ => return None,
            };
            Some(match prefix {
                "%tid" => SpecialRegister::ThreadId(axis),
                "%ctaid" => SpecialRegister::BlockId(axis),
                "%ntid" => SpecialRegister::BlockDimensions(axis),
                "%nctaid" => SpecialRegister::GridDimensions(axis),
                _ => return None,
            })
        }
    }
}
