// The generated module: the source description `compile_cached` reads, a signature type per kernel, and the `Module` wrapper with its launch methods.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Ident, Visibility};

use super::{
    KernelMetadata, ModuleMetadata,
    input::ModuleConfig,
    name::{ENTRY_POINTS, entry_point_name, prepare_name, signature_name},
};

pub(crate) fn generate_module(
    visibility: &Visibility,
    module: &Ident,
    config: &ModuleConfig,
    metadata: &ModuleMetadata,
) -> TokenStream {
    let krate = &config.krate;
    let source = &config.source;
    let cache_key = &metadata.cache_key;
    let headers = config.headers.iter().map(|header| {
        let include_name = &header.include_name;
        let source = &header.source;
        quote! {
            Header {
                include_name: #include_name,
                source: #source,
            }
        }
    });
    let nvrtc_args = &config.nvrtc_args;
    let kernel_names = metadata.kernels.iter().map(|kernel| &kernel.source_name);
    let signatures = metadata.kernels.iter().map(generate_signature);
    let methods = metadata.kernels.iter().map(generate_methods);

    quote! {
        #visibility mod #module {
            use #krate as _singe_cuda;

            use ::std::sync::Arc;

            use _singe_cuda::{
                context::Context,
                error::Result,
                graph::{ExecutableGraph, Graph, GraphNode},
                kernel::launch::{KernelSignature, LaunchRequirements, ParameterAccess, ParameterLayout, PreparedLaunch},
                module::{KernelParameters, LaunchConfig, LaunchGeometry, Module as CudaModule},
                nvrtc::{CompiledModule, Header, ModuleSource, compile_cached},
                stream::{Stream, StreamCaptureScope},
                view::{DeviceOutput, DeviceOutputView, DeviceRange, DeviceSlice, DeviceSliceMut, DeviceView, DeviceViewMut},
            };

            const MODULE_SOURCE: ModuleSource = ModuleSource {
                key: #cache_key,
                source: #source,
                headers: &[#(#headers),*],
                options: &[#(#nvrtc_args),*],
                kernels: &[#(#kernel_names),*],
            };

            #(#signatures)*

            /// The compiled module, shared by every `Module` created in the same context.
            #[derive(Debug, Clone)]
            pub struct Module {
                compiled: Arc<CompiledModule>,
            }

            impl Module {
                /// The CUDA source this module compiles.
                pub const SOURCE: &'static str = #source;

                /// Compiles the source for the device of `context`, or reuses the module a live `Module` already compiled there.
                ///
                /// # Errors
                ///
                /// Returns an error if NVRTC cannot compile the source for the device or CUDA cannot load the result.
                pub fn create(context: &Arc<Context>) -> Result<Self> {
                    Ok(Self {
                        compiled: compile_cached(context, &MODULE_SOURCE)?,
                    })
                }

                /// Returns the loaded CUDA module.
                #[must_use]
                pub fn module(&self) -> &CudaModule {
                    self.compiled.module()
                }

                #(#methods)*
            }
        }
    }
}

fn generate_signature(kernel: &KernelMetadata) -> TokenStream {
    let signature = signature_name(&kernel.rust_name);
    let raw_types = kernel.params.iter().map(|p| &p.raw_type);
    let names = kernel
        .params
        .iter()
        .map(|p| &p.rust_name)
        .collect::<Vec<_>>();
    let layouts = kernel.params.iter().map(|p| &p.layout);
    let view_types = kernel.params.iter().map(|p| &p.view_type);
    let bindings = kernel.params.iter().map(|p| &p.view_binding);
    let raw_values = kernel.params.iter().map(|p| &p.raw_value);
    let checks = kernel
        .params
        .iter()
        .filter_map(|p| p.context_check.as_ref());
    let requirements = &kernel.requirements;
    quote! {
        /// ABI and launch requirements extracted from this module's CUDA declaration.
        #[derive(Debug)]
        pub struct #signature;

        // SAFETY: argument order and representations come from the CUDA declaration and explicit type mappings.
        // Semantic access annotations are obligations of the unsafe launch caller.
        unsafe impl KernelSignature for #signature {
            type Arguments<'a> = (#(#view_types,)*);
            type RawArguments = (#(#raw_types,)*);
            const PARAMETERS: &'static [ParameterLayout] = &[#(#layouts),*];
            const REQUIREMENTS: LaunchRequirements = #requirements;
            fn raw_arguments(arguments: Self::Arguments<'_>, context: &Arc<Context>) -> Result<Self::RawArguments> {
                let (#(#bindings,)*) = arguments;
                #(#checks)*
                Ok((#(#raw_values,)*))
            }
            fn encode<R>(arguments: Self::RawArguments, f: impl FnOnce(KernelParameters<'_>) -> R) -> R {
                let (#(#names,)*) = arguments;
                let mut params = KernelParameters::new();
                #(params.arg(&#names);)*
                f(params)
            }
        }
    }
}

fn generate_methods(kernel: &KernelMetadata) -> TokenStream {
    let signature = signature_name(&kernel.rust_name);
    let prepare = prepare_name(&kernel.rust_name);
    let source_name = &kernel.source_name;
    let parameter_description = format!(
        "CUDA parameters in declaration order: {}.",
        kernel
            .params
            .iter()
            .map(|p| p.source_name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let arguments = kernel.params.iter().map(|p| {
        let name = &p.rust_name;
        let ty = &p.argument_type;
        quote! { #name: #ty }
    });
    let arguments = quote! { #(#arguments,)* };
    let values = kernel.params.iter().map(|p| &p.argument_value);
    let values = quote! { (#(#values,)*) };
    let methods = ENTRY_POINTS.map(|suffix| {
        let method = entry_point_name(&kernel.rust_name, suffix);
        let rust_name = &kernel.rust_name;
        let (description, leading, trailing, setup, dispatch, output, code_lifetime) = match suffix {
            "_async" => (
                format!("Launches `{rust_name}` on `stream`."),
                quote! {},
                quote! { config: &LaunchConfig, stream: &Stream, },
                quote! { let prepared = self.#prepare(config.clone())?; },
                quote! { prepared.launch_async(arguments, stream) },
                quote! { () },
                "Keep this module or another code owner alive until execution completes.",
            ),
            "_record" => (
                format!("Records `{rust_name}` in the active capture and retains its code."),
                quote! {},
                quote! { config: &LaunchConfig, scope: &StreamCaptureScope<'_>, },
                quote! { let prepared = self.#prepare(config.clone())?; },
                quote! { prepared.record(arguments, scope) },
                quote! { () },
                "This operation retains its code owner; keep the graph or executable alive until queued executions complete.",
            ),
            "_node" => (
                format!("Adds a `{rust_name}` kernel node and retains its code."),
                quote! { graph: &mut Graph, dependencies: &[GraphNode], config: &LaunchConfig, },
                quote! {},
                quote! { let prepared = self.#prepare(config.clone())?; },
                quote! { prepared.add_to_graph(arguments, graph, dependencies) },
                quote! { GraphNode },
                "This operation retains its code owner; keep the graph or executable alive until queued executions complete.",
            ),
            _ => (
                format!("Updates a `{rust_name}` executable node and retains its code."),
                quote! { executable: &mut ExecutableGraph, node: &GraphNode, geometry: &LaunchGeometry, },
                quote! {},
                quote! { let function = self.compiled.function(#source_name)?; },
                quote! { function.set_typed_graph_node_params::<#signature>(executable, node, geometry, arguments) },
                quote! { () },
                "This operation retains its code owner; keep the graph or executable alive until queued executions complete.\nExisting executable launch attributes must satisfy this kernel's requirements.",
            ),
        };
        quote! {
            #[doc = #description]
            #[doc = #parameter_description]
            ///
            /// # Safety
            ///
            /// Values and views must satisfy the CUDA kernel's bounds and semantic requirements, and raw pointers its initialization and aliasing requirements.
            /// Keep storage alive and preserve exclusive writable access until execution completes, including every graph replay.
            #[doc = #code_lifetime]
            /// Writable output views do not certify initialization coverage or completion.
            ///
            /// # Errors
            ///
            /// Returns an error if a view belongs to another context, the launch configuration violates the kernel's requirements, or CUDA rejects the work.
            pub unsafe fn #method(&self, #leading #arguments #trailing) -> Result<#output> {
                #setup
                let arguments = #values;
                // SAFETY: `prepared` or `function` comes from this module's compiled declaration of the kernel, and the caller guarantees the argument bounds, exclusive access, storage lifetimes, and attributes listed under `# Safety`.
                unsafe { #dispatch }
            }
        }
    });
    quote! {
        /// Prepares this kernel's signature and execution configuration in the loaded context.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA cannot find the kernel or the configuration violates its requirements.
        pub fn #prepare(&self, config: LaunchConfig) -> Result<PreparedLaunch<#signature>> {
            let function = self.compiled.function(#source_name)?;
            // SAFETY: this function and signature originate from the same compiled CUDA declaration.
            unsafe { PreparedLaunch::create(&function, config) }
        }

        #(#methods)*
    }
}
