use crate::{data_type::DataType, layout::Layout};

// Facts about a kernel its producer recorded, each optional; documentation states these and nothing else.
// Entries follow the ABI: input `i` describes `KernelArgument::Input(i)`, scalar `i` the `i`th value argument, and size `i` the `i`th `KernelArgument::Size`; an entry past the end of a list is unknown.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KernelDescription {
    // The operations the kernel computes, in order.
    pub operations: Vec<String>,
    pub inputs: Vec<BufferDescription>,
    pub outputs: Vec<BufferDescription>,
    pub scalars: Vec<String>,
    pub sizes: Vec<String>,
}

// The logical tensor a buffer argument holds; the launch pointer addresses its first element, so the layout's storage offset does not apply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferDescription {
    pub name: Option<String>,
    // The operation whose result the buffer holds.
    pub producer: Option<String>,
    pub data_type: DataType,
    pub layout: Layout,
}
