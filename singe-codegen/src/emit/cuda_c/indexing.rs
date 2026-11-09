use crate::{
    emit::cuda_c::{Writer, spelling::store_expression},
    ir::{Access, AxisId, AxisKind, ElementType, Size, SizeId, Store},
};

impl Writer<'_> {
    /// Returns the statement that writes `store`'s value.
    pub fn write(&self, store: &Store) -> String {
        let element = self.kernel.operands[store.access.operand.0].element;
        let address = format!(
            "o{}[{}]",
            store.access.operand.0,
            self.offset(&store.access)
        );
        let value = format!("v{}", store.value.0);
        match (store.combiner, element) {
            (None, _) => format!("{address} = {};", store_expression(element, &value)),
            // CUDA adds 64-bit integers only as unsigned ones, whose two's complement sums agree.
            (Some(_), ElementType::I64) => format!(
                "atomicAdd(reinterpret_cast<unsigned long long*>(&{address}), static_cast<unsigned long long>({value}));"
            ),
            // CUDA has no atomic sums of 8- and 16-bit elements, so they swap the 32-bit word that holds them.
            (Some(_), ElementType::I8 | ElementType::U8 | ElementType::I16 | ElementType::U16) => {
                format!(
                    "singe_atomic_narrow(&{address}, {}U, singe_add_integer{{static_cast<unsigned int>({value})}});",
                    element.size_bytes()
                )
            }
            (Some(_), ElementType::F16) => {
                format!("singe_atomic_narrow(&{address}, 2U, singe_add_half{{{value}}});")
            }
            (Some(_), ElementType::Bf16) => {
                format!("singe_atomic_narrow(&{address}, 2U, singe_add_bfloat{{{value}}});")
            }
            (Some(_), ElementType::F8E4M3) => {
                format!(
                    "singe_atomic_narrow(&{address}, 1U, singe_add_fp8<4, 3, true>{{{value}}});"
                )
            }
            (Some(_), ElementType::F8E5M2) => {
                format!(
                    "singe_atomic_narrow(&{address}, 1U, singe_add_fp8<5, 2, false>{{{value}}});"
                )
            }
            (Some(_), ElementType::Bool) => format!(
                "singe_atomic_narrow(&{address}, 1U, singe_or_bool{{static_cast<unsigned int>({value})}});"
            ),
            (Some(_), _) => format!("atomicAdd(&{address}, {value});"),
        }
    }

    /// Returns the element offset of `access` as an unsigned 64-bit expression.
    pub fn offset(&self, access: &Access) -> String {
        self.offset_at(access, &[])
    }

    /// Returns the element offset of `access`, naming the index of each axis in `renamed` by its expression.
    pub fn offset_at(&self, access: &Access, renamed: &[(AxisId, &str)]) -> String {
        let ix = self.index;
        let index = |axis: AxisId| {
            renamed
                .iter()
                .find(|(renamed, _)| *renamed == axis)
                .map_or_else(|| format!("a{}", axis.0), |(_, name)| (*name).to_owned())
        };
        let mut terms = Vec::new();
        if let Some(base) = access.base {
            terms.push(self.size(base));
        }

        for &(axis, stride) in &access.strides {
            terms.push(match self.kernel.sizes[stride.0] {
                Size::Static(0) => continue,
                Size::Static(1) => index(axis),
                _ => format!("{} * {}", index(axis), self.size(stride)),
            });
        }

        for &(node, stride) in &access.indirect {
            let value = format!("static_cast<{ix}>(v{})", node.0);
            terms.push(match self.kernel.sizes[stride.0] {
                Size::Static(0) => continue,
                Size::Static(1) => value,
                _ => format!("{value} * {}", self.size(stride)),
            });
        }
        if terms.is_empty() {
            format!("0{}", self.suffix)
        } else {
            terms.join(" + ")
        }
    }

    /// Defines the index of each of `axes` from the flat index `flat`, the last axis varying fastest.
    pub fn decompose(&mut self, flat: &str, axes: &[AxisId]) {
        let named = axes
            .iter()
            .map(|axis| {
                (
                    format!("a{}", axis.0),
                    self.size(self.kernel.axes[axis.0].extent),
                )
            })
            .collect::<Vec<_>>();
        self.decompose_named(flat, &named);
    }

    /// Defines each named index from the flat index `flat` and the extents beside the names, the last varying fastest.
    pub fn decompose_named(&mut self, flat: &str, indices: &[(String, String)]) {
        let ix = self.index;
        let mut divisor: Vec<&str> = Vec::new();
        for (position, (name, extent)) in indices.iter().enumerate().rev() {
            let quotient = if divisor.is_empty() {
                flat.to_owned()
            } else {
                format!("({flat} / ({}))", divisor.join(" * "))
            };
            // The outermost index needs no modulo, since the flat index is in range.
            let index = if position == 0 {
                quotient
            } else {
                format!("({quotient} % {extent})")
            };
            self.line(&format!("const {ix} {name} = {index};"));
            divisor.push(extent);
        }
    }

    /// Returns the product of the extents of `axes`, `1ULL` for none.
    pub fn product(&self, axes: &[AxisId]) -> String {
        if axes.is_empty() {
            return format!("1{}", self.suffix);
        }
        axes.iter()
            .map(|axis| self.size(self.kernel.axes[axis.0].extent))
            .collect::<Vec<_>>()
            .join(" * ")
    }

    pub fn size(&self, size: SizeId) -> String {
        match self.kernel.sizes[size.0] {
            Size::Static(value) => format!("{value}{}", self.suffix),
            Size::Argument => format!("s{}", size.0),
        }
    }

    pub fn axes_of_kind(&self, kind: AxisKind) -> Vec<AxisId> {
        self.kernel
            .axes
            .iter()
            .enumerate()
            .filter(|(_, axis)| axis.kind == kind)
            .map(|(index, _)| AxisId(index))
            .collect()
    }
}
