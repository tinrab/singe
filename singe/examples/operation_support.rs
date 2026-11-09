//! Prints the registry's dtype and execution-mode matrix without initializing CUDA.

use singe::backend::registry::operation_support_table;

fn main() {
    print!("{}", operation_support_table());
}
