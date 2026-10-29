use std::collections::{BTreeMap, BTreeSet};

use crate::{
    ast::{
        AddressSize, Initializer, LinkingDirective, Module, Parameter, PointerAnnotation,
        ScalarType, Span, StateSpace, TopLevelItem, VariableDecl, VectorSize,
    },
    emulator::{
        constants::{self, Constant, Relocation},
        error::{CompileError, CompileErrorKind, Result},
        ir::GlobalId,
        target::{self, PtxVersion},
    },
};

/// Size and alignment of a variable or parameter, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageLayout {
    size: u64,
    alignment: u32,
}

impl StorageLayout {
    /// Returns the size in bytes.
    #[must_use]
    pub fn byte_len(self) -> u64 {
        self.size
    }

    /// Returns the alignment in bytes.
    #[must_use]
    pub fn alignment(self) -> u32 {
        self.alignment
    }

    pub(crate) fn create(size: u64, alignment: u32, span: Span) -> Result<Self> {
        if !alignment.is_power_of_two() {
            return Err(CompileError::declaration(
                span,
                "alignment must be a nonzero power of two",
            ));
        }
        Ok(Self { size, alignment })
    }
}

/// Layout of one kernel or function parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterLayout {
    pub(crate) name: String,
    pub(crate) ty: ScalarType,
    pub(crate) layout: StorageLayout,
    pub(crate) offset: u64,
    pub(crate) pointer: Option<PointerAnnotation>,
    pub(crate) aggregate: bool,
}

impl ParameterLayout {
    /// Returns the parameter name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the declared element type.
    #[must_use]
    pub fn data_type(&self) -> ScalarType {
        self.ty
    }

    /// Returns the parameter's size and alignment.
    #[must_use]
    pub fn storage(&self) -> StorageLayout {
        self.layout
    }

    /// Returns the parameter's byte offset in the parameter block.
    #[must_use]
    pub fn byte_offset(&self) -> u64 {
        self.offset
    }

    /// Returns the pointee state space and alignment from a `.ptr` qualifier, if present.
    #[must_use]
    pub fn pointer(&self) -> Option<PointerAnnotation> {
        self.pointer
    }

    /// Returns whether the parameter is an array, passed with [`KernelArg::aggregate`].
    ///
    /// [`KernelArg::aggregate`]: crate::emulator::KernelArg::aggregate
    #[must_use]
    pub fn is_aggregate(&self) -> bool {
        self.aggregate
    }
}

/// Layout of one module variable, computed without allocating it.
#[derive(Debug, Clone)]
pub struct ModuleVariable {
    pub(crate) name: String,
    pub(crate) ty: ScalarType,
    pub(crate) space: StateSpace,
    pub(crate) layout: StorageLayout,
    pub(crate) dynamic: bool,
    pub(crate) initial_values: Vec<InitialValue>,
}

impl ModuleVariable {
    /// Returns the variable name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the declared element type.
    #[must_use]
    pub fn data_type(&self) -> ScalarType {
        self.ty
    }

    /// Returns the state space: global, constant, or shared.
    #[must_use]
    pub fn state_space(&self) -> StateSpace {
        self.space
    }

    /// Returns the variable's size and alignment.
    #[must_use]
    pub fn storage(&self) -> StorageLayout {
        self.layout
    }

    /// Returns whether this is an `.extern .shared` array sized at launch by
    /// [`LaunchConfig::create`](crate::emulator::LaunchConfig::create).
    #[must_use]
    pub fn is_dynamic(&self) -> bool {
        self.dynamic
    }
}

/// One write performed when a module variable is initialized.
#[derive(Debug, Clone)]
pub struct InitialValue {
    /// Byte offset within the variable.
    pub offset: u64,
    /// Number of bytes written by this initializer.
    pub width: u32,
    /// Literal bits or symbolic address to write.
    pub value: InitialBits,
}

/// Literal bits or an address relocation used to initialize module storage.
#[derive(Debug, Clone)]
pub enum InitialBits {
    /// Literal bits written into the initialized storage.
    Bits(u128),
    /// Symbolic address resolved when the module is loaded.
    Address(Relocation),
}

/// Layouts of a module's variables.
#[derive(Debug, Clone)]
pub struct ModuleLayout {
    pub(crate) address_size: AddressSize,
    pub(crate) variables: Vec<ModuleVariable>,
    pub(crate) symbols: BTreeMap<String, GlobalId>,
}

impl ModuleLayout {
    /// Returns the variable layouts in declaration order.
    #[must_use]
    pub fn variables(&self) -> &[ModuleVariable] {
        &self.variables
    }

    /// Returns the pointer width declared by `.address_size`.
    #[must_use]
    pub fn address_size(&self) -> AddressSize {
        self.address_size
    }

    pub(crate) fn create(module: &Module) -> Result<Self> {
        target::validate_module(module)?;
        let mut variables = Vec::new();
        let mut symbols = BTreeMap::new();
        for item in &module.items {
            match item {
                TopLevelItem::Variable(variable) => {
                    let layout = module_variable(variable, PtxVersion::of(module))?;
                    let id = GlobalId(variables.len());
                    if symbols.insert(variable.name.clone(), id).is_some() {
                        return Err(CompileError::new(
                            variable.span,
                            CompileErrorKind::DuplicateSymbol(variable.name.clone()),
                        ));
                    }
                    variables.push(layout);
                }
                TopLevelItem::Pragma { span, value } => validate_pragma(value, *span)?,
                TopLevelItem::Alias { span, .. } => {
                    return Err(CompileError::unsupported(*span, "module aliases"));
                }
                TopLevelItem::Section(section) if !section.name.starts_with(".debug_") => {
                    return Err(CompileError::unsupported(
                        section.span,
                        format!("section {}", section.name),
                    ));
                }
                TopLevelItem::Function(_)
                | TopLevelItem::File { .. }
                | TopLevelItem::Section(_) => {}
            }
        }
        for item in &module.items {
            if let TopLevelItem::Variable(variable) = item
                && let Some(initializer) = &variable.initializer
            {
                constants::validate_version(initializer, PtxVersion::of(module))?;
                if variable.state_space == Some(StateSpace::Shared) {
                    return Err(CompileError::declaration(
                        variable.span,
                        "shared variables cannot have initializers",
                    ));
                }
                let id = symbols[&variable.name];
                let dimensions = variable_dimensions(variable)?;
                let mut initial_values = Vec::new();
                initialize(
                    initializer,
                    &dimensions,
                    0,
                    variable.ty,
                    &symbols,
                    &mut initial_values,
                    variable.span,
                )?;
                for value in &initial_values {
                    if let InitialBits::Address(address) = &value.value
                        && !matches!(
                            variables[address.target.0].space,
                            StateSpace::Global | StateSpace::Const
                        )
                    {
                        return Err(CompileError::declaration(
                            variable.span,
                            "initializer addresses must refer to global or constant storage",
                        ));
                    }
                }
                variables[id.0].initial_values = initial_values;
            }
        }
        Ok(Self {
            address_size: module.address_size,
            variables,
            symbols,
        })
    }
}

pub fn validate_pragma(value: &str, span: Span) -> Result<()> {
    match value {
        "nounroll" | "used_bytes_mask" => Ok(()),
        _ => Err(CompileError::unsupported(span, format!("pragma {value}"))),
    }
}

/// Returns the storage size of a type that the emulator can place in memory.
pub fn scalar_bytes(ty: ScalarType, span: Span) -> Result<u32> {
    use ScalarType::{Bf16, SamplerRef, SurfRef, TexRef, Tf32};
    if matches!(ty, Bf16 | Tf32 | TexRef | SamplerRef | SurfRef) {
        return Err(CompileError::unsupported(
            span,
            format!("declaration type .{}", ty.suffix()),
        ));
    }
    ty.byte_len()
        .ok_or_else(|| CompileError::declaration(span, "predicates cannot occupy memory"))
}

pub fn validate_type_version(ty: ScalarType, version: PtxVersion, span: Span) -> Result<()> {
    if ty == ScalarType::B128 && version < PtxVersion::new(8, 3) {
        return Err(CompileError::unsupported(
            span,
            "b128 declarations require PTX 8.3",
        ));
    }
    Ok(())
}

pub fn align_up(value: u64, alignment: u32, span: Span) -> Result<u64> {
    let mask = u64::from(alignment) - 1;
    value
        .checked_add(mask)
        .map(|value| value & !mask)
        .ok_or_else(|| CompileError::new(span, CompileErrorKind::LayoutOverflow))
}

pub fn parameters(
    parameters: &[Parameter],
    address_size: AddressSize,
) -> Result<Vec<ParameterLayout>> {
    let mut result = Vec::new();
    let mut offset = 0;
    let mut names = BTreeSet::new();
    for parameter in parameters {
        if parameter.state_space != Some(StateSpace::Param) {
            return Err(CompileError::unsupported(
                parameter.span,
                "non-parameter-space function arguments",
            ));
        }
        if !names.insert(&parameter.name) {
            return Err(CompileError::new(
                parameter.span,
                CompileErrorKind::DuplicateSymbol(parameter.name.clone()),
            ));
        }
        let element = scalar_bytes(parameter.ty, parameter.span)?;
        let mut size = u64::from(element);
        for bound in &parameter.array_bounds {
            let bound = bound.filter(|bound| *bound > 0).ok_or_else(|| {
                CompileError::declaration(
                    parameter.span,
                    "parameter arrays require positive bounds",
                )
            })?;
            size = size.checked_mul(u64::from(bound)).ok_or_else(|| {
                CompileError::new(parameter.span, CompileErrorKind::LayoutOverflow)
            })?;
        }
        let alignment = parameter.alignment.unwrap_or(element);
        let layout = StorageLayout::create(size, alignment, parameter.span)?;
        if alignment < element {
            return Err(CompileError::declaration(
                parameter.span,
                "parameter alignment is smaller than its element alignment",
            ));
        }
        if let Some(pointer) = parameter.pointer {
            let pointer_type = match address_size {
                AddressSize::Bits32 => ScalarType::U32,
                AddressSize::Bits64 => ScalarType::U64,
            };
            if parameter.ty != pointer_type || !parameter.array_bounds.is_empty() {
                return Err(CompileError::declaration(
                    parameter.span,
                    "pointer parameters must be scalar unsigned values of the module address size",
                ));
            }
            if pointer.state_space.is_some_and(|space| {
                !matches!(
                    space,
                    StateSpace::Global | StateSpace::Const | StateSpace::Shared | StateSpace::Local
                )
            }) {
                return Err(CompileError::declaration(
                    parameter.span,
                    "invalid pointer state space",
                ));
            }
            if let Some(alignment) = pointer.alignment {
                StorageLayout::create(0, alignment, parameter.span)?;
            }
        }
        offset = align_up(offset, alignment, parameter.span)?;
        result.push(ParameterLayout {
            name: parameter.name.clone(),
            ty: parameter.ty,
            layout,
            offset,
            pointer: parameter.pointer,
            aggregate: !parameter.array_bounds.is_empty(),
        });
        offset = offset
            .checked_add(size)
            .ok_or_else(|| CompileError::new(parameter.span, CompileErrorKind::LayoutOverflow))?;
    }
    Ok(result)
}

pub fn is_dynamic_shared(variable: &VariableDecl) -> bool {
    variable.state_space == Some(StateSpace::Shared)
        && variable.linkage == Some(LinkingDirective::Extern)
        && variable.array_bounds == [None]
        && variable.initializer.is_none()
}

fn variable_dimensions(variable: &VariableDecl) -> Result<Vec<u64>> {
    let mut dimensions = Vec::new();
    for (axis, bound) in variable.array_bounds.iter().enumerate() {
        let count = match bound {
            Some(bound) if *bound > 0 => u64::from(*bound),
            None if axis == 0 => match &variable.initializer {
                Some(Initializer::List { values, .. }) if !values.is_empty() => values.len() as u64,
                _ => {
                    return Err(CompileError::declaration(
                        variable.span,
                        "an unsized array needs an initializer or external shared storage",
                    ));
                }
            },
            _ => {
                return Err(CompileError::declaration(
                    variable.span,
                    "array bounds must be positive and only the first bound may be inferred",
                ));
            }
        };
        dimensions.push(count);
    }
    if let Some(vector) = variable.vector {
        dimensions.push(match vector {
            VectorSize::V2 => 2,
            VectorSize::V4 => 4,
        });
    }
    Ok(dimensions)
}

pub fn variable_layout(variable: &VariableDecl) -> Result<StorageLayout> {
    if variable.initializer.is_some() && matches!(variable.ty, ScalarType::F16 | ScalarType::F16x2)
    {
        return Err(CompileError::declaration(
            variable.span,
            "half storage cannot have an initializer; initialize through bit storage or host transfers",
        ));
    }
    let element = scalar_bytes(variable.ty, variable.span)?;
    let lanes = match variable.vector {
        None => 1,
        Some(VectorSize::V2) => 2,
        Some(VectorSize::V4) => 4,
    };
    let natural = element
        .checked_mul(lanes)
        .ok_or_else(|| CompileError::new(variable.span, CompileErrorKind::LayoutOverflow))?;
    if variable.vector.is_some() && natural > 16 {
        return Err(CompileError::declaration(
            variable.span,
            "vector declarations cannot exceed 128 bits",
        ));
    }
    let alignment = variable.alignment.unwrap_or(natural);
    if alignment < natural {
        return Err(CompileError::declaration(
            variable.span,
            "alignment is smaller than the element alignment",
        ));
    }
    let size = if is_dynamic_shared(variable) {
        0
    } else {
        variable_dimensions(variable)?
            .into_iter()
            .try_fold(u64::from(element), u64::checked_mul)
            .ok_or_else(|| CompileError::new(variable.span, CompileErrorKind::LayoutOverflow))?
    };
    StorageLayout::create(size, alignment, variable.span)
}

fn initialize(
    initializer: &Initializer,
    dimensions: &[u64],
    offset: u64,
    ty: ScalarType,
    symbols: &BTreeMap<String, GlobalId>,
    output: &mut Vec<InitialValue>,
    span: Span,
) -> Result<()> {
    if let Some((&bound, tail)) = dimensions.split_first() {
        let Initializer::List { values, span } = initializer else {
            return Err(CompileError::declaration(
                span,
                "array initializer must be a list",
            ));
        };
        if values.len() as u64 > bound {
            return Err(CompileError::declaration(
                *span,
                "too many initializer elements",
            ));
        }
        let stride = tail
            .iter()
            .try_fold(u64::from(scalar_bytes(ty, *span)?), |size, count| {
                size.checked_mul(*count)
            })
            .ok_or_else(|| CompileError::new(*span, CompileErrorKind::LayoutOverflow))?;
        for (index, value) in values.iter().enumerate() {
            let offset = (index as u64)
                .checked_mul(stride)
                .and_then(|delta| offset.checked_add(delta))
                .ok_or_else(|| CompileError::new(*span, CompileErrorKind::LayoutOverflow))?;
            initialize(value, tail, offset, ty, symbols, output, *span)?;
        }
    } else {
        let width = scalar_bytes(ty, span)?;
        let value = match constants::evaluate(initializer, symbols)? {
            Constant::Address(address) => {
                if !(matches!(ty, ScalarType::U32 | ScalarType::U64)
                    || ty == ScalarType::U8 && address.mask.is_some())
                {
                    return Err(CompileError::declaration(
                        span,
                        "address initializers require u32 or u64 storage, or a masked u8 value",
                    ));
                }
                InitialBits::Address(address)
            }
            value => InitialBits::Bits(value.bits(ty, span)?),
        };
        output.push(InitialValue {
            offset,
            width,
            value,
        });
    }
    Ok(())
}

/// Validates one module-scope variable declaration and computes its layout.
fn module_variable(variable: &VariableDecl, version: PtxVersion) -> Result<ModuleVariable> {
    let Some(space @ (StateSpace::Global | StateSpace::Const | StateSpace::Shared)) =
        variable.state_space
    else {
        return Err(CompileError::declaration(
            variable.span,
            "module storage must be global, constant, or shared",
        ));
    };
    let dynamic = is_dynamic_shared(variable);
    validate_type_version(variable.ty, version, variable.span)?;
    if variable
        .linkage
        .is_some_and(|linkage| !matches!(linkage, LinkingDirective::Visible) && !dynamic)
    {
        return Err(CompileError::unsupported(
            variable.span,
            "external, weak, or common module storage",
        ));
    }
    if variable.count.is_some() || !variable.attributes.is_empty() {
        return Err(CompileError::unsupported(
            variable.span,
            "module register ranges or resource attributes",
        ));
    }
    let layout = variable_layout(variable)?;
    Ok(ModuleVariable {
        name: variable.name.clone(),
        ty: variable.ty,
        space,
        layout,
        dynamic,
        initial_values: Vec::new(),
    })
}
