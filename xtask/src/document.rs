use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::ErrorKind,
    iter,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use syn::{
    Attribute, Field, File, ForeignItemFn, ImplItem, ImplItemConst, ImplItemFn, ItemConst,
    ItemEnum, ItemImpl, ItemStatic, ItemStruct, ItemType, ItemUnion, Variant, parse_file,
    visit_mut::{self, VisitMut},
};

use crate::utility::workspace_root;

/// Injects scraped API comments into the selected generated binding files.
///
/// An empty `paths` slice selects every entry in the binding documentation catalog.
/// Paths are relative to the workspace root.
/// With `comments_only`, existing deprecation attributes remain unchanged.
///
/// # Errors
///
/// Returns an error for an unknown target, unreadable input, invalid JSON or Rust,
/// or an output file that cannot be written.
pub fn document(paths: &[PathBuf], comments_only: bool) -> Result<()> {
    let root = workspace_root()?;
    let targets = document_targets(&root);
    if paths.is_empty() {
        for target in &targets {
            document_binding_file(target, comments_only)?;
        }
    } else {
        for path in paths {
            let path = root.join(path);
            let target = targets
                .iter()
                .find(|target| target.rust_path == path)
                .with_context(|| format!("no documentation target for {}", path.display()))?;
            document_binding_file(target, comments_only)?;
        }
    }

    Ok(())
}

struct DocumentTargetSpec {
    json_paths: &'static [&'static str],
    rust_path: &'static str,
    scope_path: Option<&'static str>,
    scope_module: Option<&'static str>,
}

const DOCUMENT_TARGETS: &[DocumentTargetSpec] = &[
    DocumentTargetSpec {
        json_paths: &["xtask/docs/cublas-13.4.2-api.json"],
        rust_path: "singe-cublas-sys/src/sys_130800.rs",
        scope_path: Some("singe-cublas-sys/src/lib.rs"),
        scope_module: None,
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/cublas-13.4.2-api.json"],
        rust_path: "singe-cublas-sys/src/sys_lt_130800.rs",
        scope_path: Some("singe-cublas-sys/src/lib.rs"),
        scope_module: Some("lt_bindings"),
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/cublas-13.4.2-api.json"],
        rust_path: "singe-cublas-sys/src/sys_xt_130800.rs",
        scope_path: Some("singe-cublas-sys/src/lib.rs"),
        scope_module: Some("xt_bindings"),
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/cutensor-2.8.1-api.json"],
        rust_path: "singe-cutensor-sys/src/sys_20800.rs",
        scope_path: Some("singe-cutensor-sys/src/lib.rs"),
        scope_module: None,
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/cutensor-2.8.1-api.json"],
        rust_path: "singe-cutensor-sys/src/mg_20800.rs",
        scope_path: Some("singe-cutensor-sys/src/lib.rs"),
        scope_module: None,
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/cutensor-2.8.1-api.json"],
        rust_path: "singe-cutensor-sys/src/mp_20800.rs",
        scope_path: Some("singe-cutensor-sys/src/lib.rs"),
        scope_module: None,
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/cudnn-9.27.0-api.json"],
        rust_path: "singe-cudnn-sys/src/sys_92700.rs",
        scope_path: Some("singe-cudnn-sys/src/lib.rs"),
        scope_module: None,
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/cusolver-13.4.2-api.json"],
        rust_path: "singe-cusolver-sys/src/sys_12304.rs",
        scope_path: Some("singe-cusolver-sys/src/lib.rs"),
        scope_module: None,
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/cudss-0.8.0-api.json"],
        rust_path: "singe-cudss-sys/src/sys_800.rs",
        scope_path: Some("singe-cudss-sys/src/lib.rs"),
        scope_module: None,
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/cusparse-13.4.2-api.json"],
        rust_path: "singe-cusparse-sys/src/sys_12806.rs",
        scope_path: Some("singe-cusparse-sys/src/lib.rs"),
        scope_module: None,
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/cufft-13.4.2-api.json"],
        rust_path: "singe-cufft-sys/src/sys_12400.rs",
        scope_path: Some("singe-cufft-sys/src/lib.rs"),
        scope_module: None,
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/cufft-13.4.2-api.json"],
        rust_path: "singe-cufft-sys/src/device_12400.rs",
        scope_path: Some("singe-cufft-sys/src/lib.rs"),
        scope_module: Some("device"),
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/cufft-13.4.2-api.json"],
        rust_path: "singe-cufft-sys/src/xt_12400.rs",
        scope_path: Some("singe-cufft-sys/src/lib.rs"),
        scope_module: Some("xt"),
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/curand-13.4.2-api.json"],
        rust_path: "singe-curand-sys/src/sys_10404.rs",
        scope_path: Some("singe-curand-sys/src/lib.rs"),
        scope_module: None,
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/cuda-driver-13.4.2-api.json"],
        rust_path: "singe-cuda-sys/src/driver_sys_13040.rs",
        scope_path: Some("singe-cuda-sys/src/lib.rs"),
        scope_module: Some("driver"),
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/cuda-driver-13.4.2-api.json"],
        rust_path: "singe-cuda-sys/src/profiler_sys_13040.rs",
        scope_path: Some("singe-cuda-sys/src/lib.rs"),
        scope_module: Some("profiler"),
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/cuda-runtime-13.4.2-api.json"],
        rust_path: "singe-cuda-sys/src/driver_types_sys_13040.rs",
        scope_path: Some("singe-cuda-sys/src/lib.rs"),
        scope_module: Some("runtime"),
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/cuda-runtime-13.4.2-api.json"],
        rust_path: "singe-cuda-sys/src/library_types_sys_13040.rs",
        scope_path: Some("singe-cuda-sys/src/lib.rs"),
        scope_module: Some("library_types"),
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/cuda-runtime-13.4.2-api.json"],
        rust_path: "singe-cuda-sys/src/runtime_sys_13040.rs",
        scope_path: Some("singe-cuda-sys/src/lib.rs"),
        scope_module: Some("runtime"),
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/cufile-1.19.1-api.json"],
        rust_path: "singe-cufile-sys/src/sys_1190.rs",
        scope_path: Some("singe-cufile-sys/src/lib.rs"),
        scope_module: None,
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/cupti-13.4.2-api.json"],
        rust_path: "singe-cupti-sys/src/cupti_sys_130401.rs",
        scope_path: Some("singe-cupti-sys/src/lib.rs"),
        scope_module: None,
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/nvrtc-13.4.2-api.json"],
        rust_path: "singe-cuda-sys/src/nvrtc_sys_13040.rs",
        scope_path: Some("singe-cuda-sys/src/lib.rs"),
        scope_module: Some("nvrtc"),
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/libnvvm-13.4.2-api.json"],
        rust_path: "singe-cuda-sys/src/nvvm_sys_13040.rs",
        scope_path: Some("singe-cuda-sys/src/lib.rs"),
        scope_module: Some("nvvm"),
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/nvtx-3.6.0-api.json"],
        rust_path: "singe-cuda-sys/src/nvtx_sys_3.rs",
        scope_path: Some("singe-cuda-sys/src/lib.rs"),
        scope_module: Some("nvtx"),
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/nvml-615-api.json"],
        rust_path: "singe-nvml-sys/src/nvml_sys_13.rs",
        scope_path: Some("singe-nvml-sys/src/lib.rs"),
        scope_module: None,
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/nccl-2.32.3-api.json"],
        rust_path: "singe-nccl-sys/src/sys_23203.rs",
        scope_path: Some("singe-nccl-sys/src/lib.rs"),
        scope_module: None,
    },
    DocumentTargetSpec {
        json_paths: &["xtask/docs/npp-13.4.2-api.json"],
        rust_path: "singe-npp-sys/src/sys_13200.rs",
        scope_path: Some("singe-npp-sys/src/lib.rs"),
        scope_module: None,
    },
];

fn document_targets(root: &Path) -> Vec<DocumentTarget> {
    DOCUMENT_TARGETS
        .iter()
        .map(|spec| DocumentTarget {
            json_paths: spec.json_paths.iter().map(|path| root.join(path)).collect(),
            rust_path: root.join(spec.rust_path),
            scope_path: spec.scope_path.map(|path| root.join(path)),
            scope_module: spec.scope_module,
        })
        .collect()
}

struct DocumentTarget {
    json_paths: Vec<PathBuf>,
    rust_path: PathBuf,
    scope_path: Option<PathBuf>,
    scope_module: Option<&'static str>,
}

fn document_binding_file(target: &DocumentTarget, comments_only: bool) -> Result<()> {
    ensure_rust_path_exists(&target.rust_path)?;

    let source = fs::read_to_string(&target.rust_path)
        .with_context(|| format!("failed to read {}", target.rust_path.display()))?;
    let mut docs = load_docs(&target.json_paths)?;
    let preamble = generated_preamble(&source);
    let mut file = parse_file(&source)
        .with_context(|| format!("failed to parse {}", target.rust_path.display()))?;
    let symbols = load_symbols(
        &file,
        &target.rust_path,
        target.scope_path.as_deref(),
        target.scope_module,
    )?;
    docs.document_opaque_tags(&file, &symbols);

    BindingDocumenter {
        docs: &docs,
        symbols: &symbols,
        comments_only,
    }
    .visit_file_mut(&mut file);

    let rendered = render_file(&file, preamble);
    write_if_changed(&target.rust_path, &rendered)?;

    let root = workspace_root()?;
    println!("updated {}", rel(&root, &target.rust_path).display());

    Ok(())
}

fn ensure_rust_path_exists(path: &Path) -> Result<()> {
    match fs::metadata(path) {
        Ok(metadata) if metadata.is_file() => Ok(()),
        Ok(_) => bail!("rust_path is not a file: {}", path.display()),
        Err(error) if error.kind() == ErrorKind::NotFound => {
            bail!("rust_path file does not exist: {}", path.display())
        }
        Err(error) => Err(error).with_context(|| format!("failed to inspect {}", path.display())),
    }
}

fn load_symbols(
    file: &File,
    rust_path: &Path,
    scope_path: Option<&Path>,
    scope_module: Option<&str>,
) -> Result<SymbolIndex> {
    let mut symbols = collect_symbols_from_items(&file.items);

    let Some(scope_path) = scope_path else {
        return Ok(symbols);
    };

    let scope_file = parse_rust_file(scope_path)?;
    symbols.extend(collect_symbols_from_prior_includes(
        scope_file.items.as_slice(),
        scope_path,
        rust_path,
    )?);

    let scope_items = if let Some(module_name) = scope_module {
        scope_file
            .items
            .iter()
            .find_map(|item| match item {
                syn::Item::Mod(module) if module.ident == module_name => {
                    module.content.as_ref().map(|(_, items)| items.as_slice())
                }
                _ => None,
            })
            .with_context(|| {
                format!(
                    "failed to find module {module_name} in {}",
                    scope_path.display()
                )
            })?
    } else {
        scope_file.items.as_slice()
    };

    symbols.extend(collect_symbols_from_items(scope_items));
    symbols.extend(collect_symbols_from_prior_includes(
        scope_items,
        scope_path,
        rust_path,
    )?);
    Ok(symbols)
}

fn collect_symbols_from_prior_includes(
    items: &[syn::Item],
    scope_path: &Path,
    rust_path: &Path,
) -> Result<SymbolIndex> {
    let mut symbols = SymbolIndex::default();
    let Some(scope_dir) = scope_path.parent() else {
        return Ok(symbols);
    };
    let target_path = rust_path
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", rust_path.display()))?;

    for item in items {
        let Some(include_path) = include_macro_path(item, scope_dir)? else {
            continue;
        };
        let include_path = include_path
            .canonicalize()
            .with_context(|| format!("failed to canonicalize {}", include_path.display()))?;
        if include_path == target_path {
            break;
        }

        let include_file = parse_rust_file(&include_path)?;
        symbols.extend(collect_symbols_from_items(&include_file.items));
    }

    Ok(symbols)
}

fn include_macro_path(item: &syn::Item, scope_dir: &Path) -> Result<Option<PathBuf>> {
    let syn::Item::Macro(item) = item else {
        return Ok(None);
    };
    if !item.mac.path.is_ident("include") {
        return Ok(None);
    }

    let path = syn::parse2::<syn::LitStr>(item.mac.tokens.clone())
        .with_context(|| "failed to parse include! path")?;
    Ok(Some(scope_dir.join(path.value())))
}

fn collect_symbols_from_items(items: &[syn::Item]) -> SymbolIndex {
    let mut symbols = SymbolIndex::default();

    for item in items {
        collect_symbols_from_item(item, &mut symbols);
    }

    symbols
}

fn collect_symbols_from_item(item: &syn::Item, symbols: &mut SymbolIndex) {
    match item {
        syn::Item::Const(item) => {
            symbols.insert_name(item.ident.to_string());
        }
        syn::Item::Enum(item) => {
            let enum_name = item.ident.to_string();
            symbols.insert_name(enum_name.clone());
            for variant in &item.variants {
                symbols.insert_variant(variant.ident.to_string(), enum_name.clone());
            }
        }
        syn::Item::Fn(item) => {
            symbols.insert_name(item.sig.ident.to_string());
        }
        syn::Item::ForeignMod(item) => {
            for foreign_item in &item.items {
                match foreign_item {
                    syn::ForeignItem::Fn(function) => {
                        symbols.insert_name(function.sig.ident.to_string());
                    }
                    syn::ForeignItem::Static(item) => {
                        symbols.insert_name(item.ident.to_string());
                    }
                    _ => {}
                }
            }
        }
        syn::Item::Static(item) => {
            symbols.insert_name(item.ident.to_string());
        }
        syn::Item::Struct(item) => {
            symbols.insert_name(item.ident.to_string());
            if item.fields.iter().any(|field| {
                field
                    .ident
                    .as_ref()
                    .is_some_and(|name| name.to_string().starts_with("_bitfield_"))
            }) {
                symbols.bitfield_records.insert(item.ident.to_string());
            }
        }
        syn::Item::Trait(item) => {
            symbols.insert_name(item.ident.to_string());
        }
        syn::Item::Type(item) => {
            symbols.insert_name(item.ident.to_string());
            if let syn::Type::Path(path) = item.ty.as_ref()
                && let Some(target) = path.path.get_ident()
            {
                symbols.insert_alias(item.ident.to_string(), target.to_string());
            }
        }
        syn::Item::Union(item) => {
            symbols.insert_name(item.ident.to_string());
        }
        syn::Item::Use(item) => collect_symbols_from_use_tree(&item.tree, symbols),
        _ => {}
    }
}

fn collect_symbols_from_use_tree(tree: &syn::UseTree, symbols: &mut SymbolIndex) {
    match tree {
        syn::UseTree::Name(name) => {
            symbols.insert_name(name.ident.to_string());
        }
        syn::UseTree::Rename(rename) => {
            symbols.insert_alias(rename.rename.to_string(), rename.ident.to_string());
        }
        syn::UseTree::Group(group) => {
            for item in &group.items {
                collect_symbols_from_use_tree(item, symbols);
            }
        }
        syn::UseTree::Path(path) => {
            collect_symbols_from_use_tree(&path.tree, symbols);
        }
        syn::UseTree::Glob(_) => {}
    }
}

fn load_docs(paths: &[PathBuf]) -> Result<DocDatabase> {
    let mut entries = BTreeMap::new();
    let mut seen = BTreeSet::new();
    let mut sources = Vec::new();
    if let Some(directory) = paths.first().and_then(|path| path.parent()) {
        // CUDA types also appear in bindings for libraries built on the toolkit.
        for name in [
            "cuda-driver-13.4.2-api.json",
            "cuda-runtime-13.4.2-api.json",
        ] {
            sources.push(directory.join(name));
        }
    }
    sources.extend_from_slice(paths);
    if let Some(directory) = paths.first().and_then(|path| path.parent()) {
        sources.push(directory.join("binding-representations.json"));
    }

    for source in sources {
        let sidecars: Vec<_> = ["-supplement.json", "-headers.json"]
            .into_iter()
            .filter_map(|suffix| {
                let name = source.file_name()?.to_str()?;
                Some(source.with_file_name(name.replace("-api.json", suffix)))
            })
            .filter(|path| path.is_file())
            .collect();
        for (index, path) in iter::once(source).chain(sidecars).enumerate() {
            if !seen.insert(path.clone()) {
                continue;
            }
            let contents = fs::read_to_string(&path)
                .with_context(|| format!("failed to read {}", path.display()))?;
            let path_entries: BTreeMap<String, RawDocEntry> = serde_json::from_str(&contents)
                .with_context(|| format!("failed to parse {}", path.display()))?;
            for (name, mut entry) in path_entries {
                entry.supplemental = index != 0
                    || path
                        .file_name()
                        .is_some_and(|name| name == "binding-representations.json");
                if index == 0
                    && path
                        .file_name()
                        .is_none_or(|name| name != "binding-representations.json")
                {
                    if let Some(previous) = entries.remove(&name) {
                        entry.fill_missing(previous);
                    }
                    entries.insert(name, entry);
                } else if let Some(existing) = entries.get_mut(&name) {
                    existing.fill_missing(entry);
                } else {
                    entries.insert(name, entry);
                }
            }
        }
    }

    Ok(DocDatabase::from_entries(entries))
}

fn parse_rust_file(path: &Path) -> Result<File> {
    let source =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    parse_file(&source).with_context(|| format!("failed to parse {}", path.display()))
}

fn generated_preamble(source: &str) -> Option<&str> {
    if !source.starts_with("/*") {
        return None;
    }

    let end = source.find("*/")? + 2;
    Some(source[..end].trim_end())
}

fn render_file(file: &File, preamble: Option<&str>) -> String {
    let body = prettyplease::unparse(file);
    match preamble {
        Some(preamble) => format!("{preamble}\n\n{body}"),
        None => body,
    }
}

fn write_if_changed(path: &Path, contents: &str) -> Result<()> {
    match fs::read_to_string(path) {
        Ok(existing) if existing == contents => return Ok(()),
        Ok(_) => {}
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| format!("failed to read {}", path.display()));
        }
    }

    fs::write(path, contents).with_context(|| format!("failed to write {}", path.display()))
}

fn rel<'a>(root: &'a Path, path: &'a Path) -> &'a Path {
    path.strip_prefix(root).unwrap_or(path)
}

#[derive(Debug, Default, Deserialize)]
struct RawDocEntry {
    #[serde(skip)]
    supplemental: bool,
    #[serde(default)]
    doc: String,
    #[serde(default)]
    deprecated: bool,
    #[serde(default)]
    parameters: BTreeMap<String, RawParameterDocEntry>,
    #[serde(default)]
    return_value: Option<RawReturnValueDoc>,
    #[serde(default)]
    members: BTreeMap<String, RawMemberDoc>,
    #[serde(default)]
    fields: BTreeMap<String, RawMemberDoc>,
    #[serde(default)]
    children: BTreeMap<String, RawMemberDoc>,
    #[serde(default)]
    member_ranges: Vec<RawMemberRangeDoc>,
}

impl RawDocEntry {
    fn fill_missing(&mut self, other: Self) {
        if self.doc.trim().is_empty() {
            self.doc = other.doc;
        }
        for (name, value) in other.parameters {
            let existing = self
                .parameters
                .entry(name)
                .or_insert_with(|| RawParameterDocEntry { doc: String::new() });
            if existing.doc.trim().is_empty() {
                existing.doc = value.doc;
            }
        }
        if self.return_value.is_none() {
            self.return_value = other.return_value;
        }
        for (target, additional) in [
            (&mut self.members, other.members),
            (&mut self.fields, other.fields),
            (&mut self.children, other.children),
        ] {
            for (name, value) in additional {
                if let Some(existing) = target.get_mut(&name) {
                    existing.fill_missing(value);
                } else {
                    target.insert(name, value);
                }
            }
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawMemberDoc {
    Entry(RawMemberDocEntry),
    Text(String),
}

impl RawMemberDoc {
    fn fill_missing(&mut self, other: Self) {
        let additional = RawMemberDocEntry::from(other);
        let doc = match self {
            Self::Entry(entry) => &mut entry.doc,
            Self::Text(text) => text,
        };
        if doc.trim().is_empty() {
            *doc = additional.doc;
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawMemberDocEntry {
    #[serde(default)]
    doc: String,
    #[serde(default)]
    deprecated: bool,
}

impl From<RawMemberDoc> for RawMemberDocEntry {
    fn from(value: RawMemberDoc) -> Self {
        match value {
            RawMemberDoc::Entry(entry) => entry,
            RawMemberDoc::Text(doc) => Self {
                doc,
                deprecated: false,
            },
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawParameterDocEntry {
    #[serde(default)]
    doc: String,
}

#[derive(Debug, Deserialize)]
struct RawReturnValueDoc {
    #[serde(default)]
    kind: String,
    #[serde(default, rename = "type")]
    type_name: String,
    #[serde(default)]
    variants: BTreeMap<String, RawReturnVariantDocEntry>,
}

#[derive(Debug, Deserialize)]
struct RawReturnVariantDocEntry {
    #[serde(default)]
    doc: String,
}

#[derive(Debug, Deserialize)]
struct RawMemberRangeDoc {
    name: String,
    prefix: String,
    suffix: String,
    start: u32,
    end: u32,
}

#[derive(Debug)]
struct DocDatabase {
    entries: BTreeMap<String, DocEntry>,
    placeholder_entries: Vec<PlaceholderDocEntry>,
}

#[derive(Debug)]
struct DocEntry {
    supplemental: bool,
    item_doc: Option<String>,
    deprecated: bool,
    parameters: BTreeMap<String, String>,
    return_value: Option<ReturnValueDoc>,
    exact_members: BTreeMap<String, String>,
    exact_member_deprecated: BTreeMap<String, bool>,
    ranged_members: Vec<MemberRangeDoc>,
}

#[derive(Debug)]
struct ReturnValueDoc {
    kind: String,
    type_name: String,
    variants: Vec<(String, String)>,
}

#[derive(Debug)]
struct PlaceholderDocEntry {
    template_name: String,
    entry: DocEntry,
}

#[derive(Debug)]
struct MemberRangeDoc {
    prefix: String,
    suffix: String,
    start: u32,
    end: u32,
    doc: String,
    deprecated: bool,
}

#[derive(Debug, Default)]
struct SymbolIndex {
    names: BTreeSet<String>,
    aliases: BTreeMap<String, String>,
    variant_owners: BTreeMap<String, BTreeSet<String>>,
    bitfield_records: BTreeSet<String>,
}

impl SymbolIndex {
    fn insert_name(&mut self, name: String) {
        self.names.insert(name);
    }

    fn insert_alias(&mut self, alias: String, target: String) {
        self.names.insert(alias.clone());
        self.aliases.insert(alias, target);
    }

    fn insert_variant(&mut self, variant: String, owner: String) {
        self.names.insert(variant.clone());
        self.variant_owners
            .entry(variant)
            .or_default()
            .insert(owner);
    }

    fn contains_name(&self, name: &str) -> bool {
        self.names.contains(name)
    }

    fn contains_alias(&self, name: &str) -> bool {
        self.aliases.contains_key(name)
    }

    fn aliases_for_target(&self, target: &str) -> Vec<&str> {
        self.aliases
            .iter()
            .filter(|(_, t)| *t == target)
            .map(|(a, _)| a.as_str())
            .collect()
    }

    fn resolve_variant_path(&self, variant: &str) -> Option<String> {
        let owners = self.variant_owners.get(variant)?;
        if owners.len() != 1 {
            return None;
        }

        owners
            .iter()
            .next()
            .map(|owner| format!("{owner}::{variant}"))
    }

    fn contains_variant_path(&self, owner: &str, variant: &str) -> bool {
        let owner = self.aliases.get(owner).map_or(owner, String::as_str);
        self.variant_owners
            .get(variant)
            .is_some_and(|owners| owners.contains(owner))
    }

    fn extend(&mut self, other: Self) {
        self.names.extend(other.names);
        self.aliases.extend(other.aliases);
        self.bitfield_records.extend(other.bitfield_records);
        for (variant, owners) in other.variant_owners {
            self.variant_owners
                .entry(variant)
                .or_default()
                .extend(owners);
        }
    }
}

impl DocDatabase {
    fn document_opaque_tags(&mut self, file: &File, symbols: &SymbolIndex) {
        for item in &file.items {
            let syn::Item::Type(alias) = item else {
                continue;
            };
            let syn::Type::Ptr(pointer) = alias.ty.as_ref() else {
                continue;
            };
            let syn::Type::Path(target) = pointer.elem.as_ref() else {
                continue;
            };
            let Some(target) = target.path.get_ident() else {
                continue;
            };
            let opaque = file.items.iter().any(|item| {
                matches!(item, syn::Item::Struct(item) if item.ident == *target
                    && item.fields.len() == 1
                    && item.fields.iter().all(|field| field.ident.as_ref().is_some_and(|name| name == "_unused")))
            });
            if opaque && self.find_item_doc(&target.to_string(), symbols).is_none() {
                self.entries.insert(
                    target.to_string(),
                    DocEntry::from_raw(RawDocEntry {
                        doc: format!("Incomplete native type used by `{}` handles. The native API owns its layout.", alias.ident),
                        ..RawDocEntry::default()
                    }),
                );
            }
        }
    }

    fn from_entries(entries: BTreeMap<String, RawDocEntry>) -> Self {
        let mut exact_entries = BTreeMap::new();
        let mut placeholder_entries = Vec::new();

        for (name, entry) in entries {
            let entry = DocEntry::from_raw(entry);
            if name.contains("<t>") || name.contains("<t1><t2>") {
                placeholder_entries.push(PlaceholderDocEntry {
                    template_name: name,
                    entry,
                });
            } else {
                exact_entries.insert(name, entry);
            }
        }

        Self {
            entries: exact_entries,
            placeholder_entries,
        }
    }

    fn find_item_doc(&self, name: &str, symbols: &SymbolIndex) -> Option<String> {
        for candidate in binding_name_candidates(name) {
            if let Some(doc) = self.item_doc(candidate.as_str()) {
                return Some(doc);
            }

            if let Some((entry, arguments)) = self.find_placeholder_entry(&candidate)
                && let Some(doc) = &entry.item_doc
            {
                return Some(expand_type_argument_doc(doc, &arguments, symbols));
            }
        }

        for alias in symbols.aliases_for_target(name) {
            if let Some(doc) = self.item_doc(alias) {
                return Some(doc);
            }
        }

        let mut target = name;
        let mut seen = BTreeSet::new();
        while let Some(next) = symbols.aliases.get(target)
            && seen.insert(next)
        {
            for candidate in binding_name_candidates(next) {
                if let Some(doc) = self.item_doc(&candidate) {
                    return Some(doc);
                }
            }
            target = next;
        }

        None
    }

    fn find_placeholder_entry(&self, name: &str) -> Option<(&DocEntry, BTreeMap<String, String>)> {
        self.placeholder_entries.iter().find_map(|placeholder| {
            let arguments = match_type_arguments(&placeholder.template_name, name)?;
            Some((&placeholder.entry, arguments))
        })
    }

    fn item_doc(&self, name: &str) -> Option<String> {
        self.entries
            .get(name)
            .and_then(|entry| entry.item_doc.clone())
    }

    fn find_member_doc(&self, parent: &str, member: &str, symbols: &SymbolIndex) -> Option<&str> {
        self.find_entry(parent, symbols)
            .and_then(|entry| {
                entry.find_member_doc(member).or_else(|| {
                    member
                        .strip_suffix('_')
                        .and_then(|name| entry.find_member_doc(name))
                })
            })
            .or_else(|| self.entries.get(member)?.item_doc.as_deref())
    }

    fn find_entry(&self, name: &str, symbols: &SymbolIndex) -> Option<&DocEntry> {
        for candidate in binding_name_candidates(name) {
            if let Some(entry) = self.entries.get(candidate.as_str()) {
                return Some(entry);
            }
        }
        for alias in symbols.aliases_for_target(name) {
            if let Some(entry) = self.entries.get(alias) {
                return Some(entry);
            }
        }
        None
    }
}

impl DocEntry {
    fn from_raw(entry: RawDocEntry) -> Self {
        let RawDocEntry {
            supplemental,
            doc,
            deprecated,
            parameters,
            return_value,
            members,
            fields,
            children,
            member_ranges,
        } = entry;
        let mut exact_members = BTreeMap::new();
        let mut exact_member_deprecated = BTreeMap::new();
        let mut ranged_members = Vec::new();

        for (name, member) in children.into_iter().chain(members).chain(fields) {
            let member = RawMemberDocEntry::from(member);
            let member_doc = member.doc.trim().to_string();
            if member_doc.is_empty() {
                if member.deprecated {
                    exact_member_deprecated.insert(name, true);
                }
                continue;
            }

            exact_members.insert(name.clone(), member_doc.clone());
            exact_member_deprecated.insert(name, member.deprecated);
        }

        for range in member_ranges {
            let Some(doc) = exact_members.get(&range.name).cloned() else {
                continue;
            };
            ranged_members.push(MemberRangeDoc {
                prefix: range.prefix,
                suffix: range.suffix,
                start: range.start,
                end: range.end,
                doc,
                deprecated: exact_member_deprecated
                    .get(&range.name)
                    .copied()
                    .unwrap_or(false),
            });
        }

        Self {
            supplemental,
            item_doc: build_item_doc(&doc),
            deprecated,
            parameters: parameters
                .into_iter()
                .filter_map(|(name, entry)| {
                    let doc = entry.doc.trim();
                    (!doc.is_empty()).then(|| (name, doc.to_string()))
                })
                .collect(),
            return_value: build_return_value_doc(return_value),
            exact_members,
            exact_member_deprecated,
            ranged_members,
        }
    }

    fn find_member_doc(&self, member: &str) -> Option<&str> {
        if let Some(doc) = self.exact_members.get(member) {
            return Some(doc);
        }

        self.ranged_members
            .iter()
            .find(|range| range.matches(member))
            .map(|range| range.doc.as_str())
    }

    fn find_member_deprecated(&self, member: &str) -> bool {
        if let Some(deprecated) = self.exact_member_deprecated.get(member) {
            return *deprecated;
        }

        self.ranged_members
            .iter()
            .find(|range| range.matches(member))
            .is_some_and(|range| range.deprecated)
    }
}

impl MemberRangeDoc {
    fn matches(&self, member: &str) -> bool {
        let Some(member) = ParsedNumericName::parse(member) else {
            return false;
        };

        member.prefix == self.prefix
            && member.suffix == self.suffix
            && (self.start..=self.end).contains(&member.number)
    }
}

struct ParsedNumericName {
    prefix: String,
    number: u32,
    suffix: String,
}

impl ParsedNumericName {
    fn parse(name: &str) -> Option<Self> {
        let start = name.find(|ch: char| ch.is_ascii_digit())?;
        let end = name[start..]
            .find(|ch: char| !ch.is_ascii_digit())
            .map_or(name.len(), |offset| start + offset);

        Some(Self {
            prefix: name[..start].to_string(),
            number: name[start..end].parse().ok()?,
            suffix: name[end..].to_string(),
        })
    }
}

fn binding_name_candidates(name: &str) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut pending = vec![name.to_string()];
    let mut candidates = Vec::new();

    while let Some(candidate) = pending.pop() {
        if !seen.insert(candidate.clone()) {
            continue;
        }

        if let Some(stripped) = candidate.strip_suffix("_st") {
            pending.push(format!("{stripped}_t"));
        }

        if let Some(stripped) = candidate.strip_suffix("_enum") {
            pending.push(stripped.to_string());
            pending.push(format!("{stripped}_t"));
        }

        if let Some((base, version)) = candidate.rsplit_once("_v")
            && !version.is_empty()
            && version.chars().all(|digit| digit.is_ascii_digit())
        {
            pending.push(base.to_string());
        }
        if candidate.starts_with("__nvJitLink") {
            let mut parts = candidate.rsplitn(3, '_');
            if let (Some(minor), Some(major), Some(base)) =
                (parts.next(), parts.next(), parts.next())
                && !major.is_empty()
                && !minor.is_empty()
                && major
                    .chars()
                    .chain(minor.chars())
                    .all(|digit| digit.is_ascii_digit())
            {
                pending.push(base.trim_start_matches('_').to_owned());
            }
        }
        for suffix in ["_v2_64", "_64", "_ptsz", "_ptds"] {
            if let Some(stripped) = candidate.strip_suffix(suffix) {
                pending.push(stripped.to_string());
            }
        }
        // The native dot/ger reference pages group conjugated and
        // unconjugated complex entry points under the same <t> template.
        if let Some(operation) = candidate.strip_prefix("cublas")
            && let Some(operation) = operation.strip_prefix(['C', 'Z'])
            && matches!(operation, "dotu" | "dotc" | "geru" | "gerc")
        {
            pending.push(candidate[..candidate.len() - 1].to_owned());
        }
        // cuSOLVER reference pages group complex unitary/Hermitian routines
        // under the corresponding orthogonal/symmetric routine template.
        if let Some(operation) = candidate.strip_prefix("cusolverDn")
            && let Some(grouped) =
                operation
                    .strip_prefix(['C', 'Z'])
                    .and_then(|operation| match operation {
                        "ungqr" => Some("orgqr"),
                        "unmqr" => Some("ormqr"),
                        "ungbr" => Some("orgbr"),
                        "hetrd" => Some("sytrd"),
                        "ungtr" => Some("orgtr"),
                        "unmtr" => Some("ormtr"),
                        "heevd" => Some("syevd"),
                        "heevdx" => Some("syevdx"),
                        "hegvd" => Some("sygvd"),
                        "hegvdx" => Some("sygvdx"),
                        "heevj" => Some("syevj"),
                        "heevjBatched" => Some("syevjBatched"),
                        "hegvj" => Some("sygvj"),
                        _ => None,
                    })
        {
            pending.push(format!("cusolverDn{}{grouped}", &operation[..1]));
        }

        candidates.push(candidate);
    }

    candidates
}

fn match_type_arguments(template: &str, concrete: &str) -> Option<BTreeMap<String, String>> {
    if let Some((prefix, suffix)) = template.split_once("<t>") {
        let middle = concrete.strip_prefix(prefix)?.strip_suffix(suffix)?;
        return (!middle.is_empty()).then(|| BTreeMap::from([("t".to_owned(), middle.to_owned())]));
    }
    let (prefix, suffix) = template.split_once("<t1><t2>")?;
    let middle = concrete.strip_prefix(prefix)?.strip_suffix(suffix)?;
    let mut characters = middle.chars();
    let (Some(input), Some(internal), None) =
        (characters.next(), characters.next(), characters.next())
    else {
        return None;
    };
    (input.is_ascii_uppercase() && internal.is_ascii_uppercase()).then(|| {
        BTreeMap::from([
            ("t1".to_owned(), input.to_string()),
            ("t2".to_owned(), internal.to_string()),
        ])
    })
}

fn expand_type_placeholder(text: &str, type_token: &str) -> String {
    text.replace("<t>", type_token)
}

fn expand_type_argument_doc(
    doc: &str,
    arguments: &BTreeMap<String, String>,
    symbols: &SymbolIndex,
) -> String {
    let mut expanded = doc.to_owned();
    for (name, value) in arguments {
        for name in [name.to_owned(), name.to_uppercase()] {
            for placeholder in [
                format!("&lt;{name}&gt;"),
                format!("&lt;{name}>"),
                format!("\\<{name}\\>"),
                format!("\\<{name}>"),
                format!("<{name}>"),
            ] {
                expanded = expanded.replace(&placeholder, value);
            }
        }
    }
    rewrite_internal_markdown_links(&expanded, arguments.get("t").map(String::as_str), symbols)
}

fn build_item_doc(doc: &str) -> Option<String> {
    let doc = doc.trim();
    (!doc.is_empty()).then(|| doc.to_string())
}

fn build_return_value_doc(return_value: Option<RawReturnValueDoc>) -> Option<ReturnValueDoc> {
    let return_value = return_value?;

    let variants = return_value
        .variants
        .into_iter()
        .filter_map(|(name, entry)| {
            let doc = entry.doc.trim();
            (!doc.is_empty()).then(|| (name, doc.to_string()))
        })
        .collect::<Vec<_>>();

    (!variants.is_empty()).then_some(ReturnValueDoc {
        kind: return_value.kind,
        type_name: return_value.type_name,
        variants,
    })
}

struct BindingDocumenter<'a> {
    docs: &'a DocDatabase,
    symbols: &'a SymbolIndex,
    comments_only: bool,
}

impl BindingDocumenter<'_> {
    fn bitfield_accessor_doc(&self, owner: &str, function: &ImplItemFn) -> Option<String> {
        if !self.symbols.bitfield_records.contains(owner) {
            return None;
        }
        let name = function.sig.ident.to_string();
        if name.starts_with("new_bitfield_") {
            return Some("Constructs packed native bitfield storage from the supplied field values. Only the bits assigned to each field are stored; remaining bits are zero.".to_owned());
        }
        let raw = name.ends_with("_raw");
        let field = name.strip_suffix("_raw").unwrap_or(&name);
        let setter = field.starts_with("set_");
        let field = field.strip_prefix("set_").unwrap_or(field);
        let native = self.docs.find_member_doc(owner, field, self.symbols)?;
        let mut doc = if setter {
            format!(
                "Writes the `{field}` bitfield. Only the low bits that fit in the field are stored.\n\n{native}"
            )
        } else if raw {
            format!("Reads the `{field}` bitfield through a raw pointer.\n\n{native}")
        } else {
            return None;
        };
        if raw && matches!(function.sig.safety, syn::Safety::Unsafe(_)) {
            doc.push_str("\n\n# Safety\n\n`this` must address a live native record whose bitfield storage is initialized. ");
            doc.push_str(if setter {
                "The bytes containing the field must be writable and exclusively accessible during the call."
            } else {
                "The bytes containing the field must remain readable and must not be modified concurrently during the call."
            });
        }
        Some(doc)
    }

    fn update_deprecated_attr(&self, attrs: &mut Vec<Attribute>, deprecated: bool) {
        if self.comments_only {
            return;
        }
        attrs.retain(|attr| !attr.path().is_ident("deprecated"));
        if deprecated {
            attrs.insert(0, syn::parse_quote!(#[deprecated]));
        }
    }

    fn update_attrs(&self, attrs: &mut Vec<Attribute>, doc: Option<String>) {
        let Some(doc) = doc else {
            return;
        };

        attrs.retain(|attr| {
            !(attr.path().is_ident("doc") && matches!(&attr.meta, syn::Meta::NameValue(_)))
        });

        for line in normalize_doc_text(&doc, self.symbols).lines().rev() {
            let line = line.trim_end();
            let line = if line.trim().is_empty() {
                String::new()
            } else {
                format!(" {line}")
            };
            attrs.insert(0, syn::parse_quote!(#[doc = #line]));
        }
    }

    fn document_item(&self, ident: &syn::Ident, attrs: &mut Vec<Attribute>) {
        let deprecated = self
            .docs
            .find_entry(&ident.to_string(), self.symbols)
            .is_some_and(|entry| entry.deprecated);
        self.update_deprecated_attr(attrs, deprecated);
        self.update_attrs(
            attrs,
            self.docs.find_item_doc(&ident.to_string(), self.symbols),
        );
    }

    fn document_field(&self, parent: &str, index: usize, field: &mut Field) {
        let name = field
            .ident
            .as_ref()
            .map_or_else(|| index.to_string(), ToString::to_string);

        let deprecated = self
            .docs
            .find_entry(parent, self.symbols)
            .is_some_and(|entry| entry.find_member_deprecated(&name));
        self.update_deprecated_attr(&mut field.attrs, deprecated);
        let has_docs = field.attrs.iter().any(|attr| attr.path().is_ident("doc"));
        self.update_attrs(
            &mut field.attrs,
            self.docs
                .find_member_doc(parent, &name, self.symbols)
                .map(ToOwned::to_owned)
                .or_else(|| {
                    if name.starts_with("_bitfield_align_") {
                        Some("Zero-length marker used by bindgen to preserve native bitfield alignment.".to_owned())
                    } else if has_docs {
                        None
                    } else if name.starts_with("__bindgen_padding_") {
                        Some(
                            "Padding inserted by bindgen to preserve the native ABI layout."
                                .to_owned(),
                        )
                    } else if name == "reserved"
                        || name.starts_with("reserved") && matches!(&field.ty, syn::Type::Array(_))
                    {
                        Some("Reserved storage in the native ABI.".to_owned())
                    } else if name == "_internal_padding" {
                        Some("Internal padding in the native ABI layout.".to_owned())
                    } else if name == "_oversize" {
                        Some("Storage that fixes the native union's ABI size.".to_owned())
                    } else if name == "bindgen_union_field" {
                        Some("Backing storage shared by the members of this bindgen-emulated C union.".to_owned())
                    } else if name.starts_with("_bitfield_") {
                        Some("Packed storage for native C bitfields. Use the generated accessors to read or update individual fields.".to_owned())
                    } else if name.starts_with("__bindgen_anon_")
                        && let syn::Type::Path(path) = &field.ty
                        && let Some(target) = path.path.get_ident()
                    {
                        Some(format!(
                            "Anonymous native member. See `{target}` for its fields."
                        ))
                    } else {
                        None
                    }
                }),
        );
    }

    fn document_foreign_fn(&self, item: &mut ForeignItemFn) {
        let name = item.sig.ident.to_string();
        let deprecated = self
            .docs
            .find_entry(&name, self.symbols)
            .is_some_and(|entry| entry.deprecated);
        let doc = self
            .docs
            .find_entry(&name, self.symbols)
            .filter(|entry| !entry.supplemental)
            .and_then(|entry| self.build_foreign_fn_doc(entry, item))
            .or_else(|| {
                binding_name_candidates(&name)
                    .into_iter()
                    .find_map(|candidate| {
                        let (entry, arguments) = self.docs.find_placeholder_entry(&candidate)?;
                        let doc = self.build_foreign_fn_doc(entry, item)?;
                        Some(expand_type_argument_doc(&doc, &arguments, self.symbols))
                    })
            })
            .or_else(|| {
                self.docs
                    .find_entry(&name, self.symbols)
                    .and_then(|entry| self.build_foreign_fn_doc(entry, item))
            })
            .or_else(|| self.docs.find_item_doc(&name, self.symbols));
        self.update_deprecated_attr(&mut item.attrs, deprecated);
        self.update_attrs(&mut item.attrs, doc);
    }

    fn build_foreign_fn_doc(&self, entry: &DocEntry, item: &ForeignItemFn) -> Option<String> {
        let mut sections = Vec::new();

        if let Some(doc) = &entry.item_doc {
            sections.push(doc.clone());
        }

        let parameter_lines = item
            .sig
            .inputs
            .iter()
            .filter_map(|arg| match arg {
                syn::FnArg::Typed(arg) => match arg.pat.as_ref() {
                    syn::Pat::Ident(ident) => entry
                        .parameters
                        .get(&ident.ident.to_string())
                        .or_else(|| {
                            ident
                                .ident
                                .to_string()
                                .strip_suffix('_')
                                .and_then(|name| entry.parameters.get(name))
                        })
                        .map(|doc| format!("- `{}`: {}", ident.ident, doc)),
                    _ => None,
                },
                syn::FnArg::Receiver(_) => None,
            })
            .collect::<Vec<_>>();
        if !parameter_lines.is_empty() {
            sections.push(format!("# Parameters\n\n{}", parameter_lines.join("\n")));
        }

        let return_value_lines =
            build_return_value_lines(entry.return_value.as_ref(), self.symbols);
        if !return_value_lines.is_empty() {
            sections.push(format!(
                "# Return value\n\n{}",
                return_value_lines.join("\n")
            ));
        }

        (!sections.is_empty()).then(|| sections.join("\n\n"))
    }
}

fn build_return_value_lines(
    return_value: Option<&ReturnValueDoc>,
    symbols: &SymbolIndex,
) -> Vec<String> {
    let Some(return_value) = return_value else {
        return Vec::new();
    };

    return_value
        .variants
        .iter()
        .map(|(name, doc)| {
            let label = format_return_value_label(return_value, name, symbols);
            if label.is_empty() {
                format!("- {doc}")
            } else {
                format!("- {label}: {doc}")
            }
        })
        .collect()
}

fn format_return_value_label(
    return_value: &ReturnValueDoc,
    name: &str,
    symbols: &SymbolIndex,
) -> String {
    let name = name.trim();
    if name.is_empty() {
        return String::new();
    }

    if return_value.kind == "enum" && !return_value.type_name.trim().is_empty() {
        let type_name = return_value.type_name.trim();
        if symbols.contains_name(type_name) && symbols.contains_variant_path(type_name, name) {
            return format!("[`{type_name}::{name}`]");
        }
    }

    resolve_doc_symbol(name, symbols)
        .map_or_else(|| format!("`{name}`"), |symbol| format!("[`{symbol}`]"))
}

impl VisitMut for BindingDocumenter<'_> {
    fn visit_item_impl_mut(&mut self, item: &mut ItemImpl) {
        if item.trait_.is_none()
            && let syn::Type::Path(path) = item.self_ty.as_ref()
            && let Some(segment) = path.path.segments.last()
        {
            let owner = segment.ident.to_string();
            for member in &mut item.items {
                if let ImplItem::Fn(function) = member {
                    let doc = self
                        .docs
                        .find_member_doc(&owner, &function.sig.ident.to_string(), self.symbols)
                        .map(ToOwned::to_owned)
                        .or_else(|| self.bitfield_accessor_doc(&owner, function));
                    self.update_attrs(&mut function.attrs, doc);
                }
            }
        }
        visit_mut::visit_item_impl_mut(self, item);
    }

    fn visit_impl_item_const_mut(&mut self, item: &mut ImplItemConst) {
        self.document_item(&item.ident, &mut item.attrs);
        if !item.attrs.iter().any(|attr| attr.path().is_ident("doc"))
            && let syn::Expr::Path(value) = &item.expr
            && value.path.segments.len() == 2
        {
            let mut segments = value.path.segments.iter();
            if let (Some(owner), Some(variant)) = (segments.next(), segments.next())
                && self
                    .symbols
                    .contains_variant_path(&owner.ident.to_string(), &variant.ident.to_string())
            {
                self.update_attrs(
                    &mut item.attrs,
                    Some(format!("Alias for `{}::{}`.", owner.ident, variant.ident)),
                );
            }
        }
        visit_mut::visit_impl_item_const_mut(self, item);
    }

    fn visit_item_const_mut(&mut self, item: &mut ItemConst) {
        self.document_item(&item.ident, &mut item.attrs);
        visit_mut::visit_item_const_mut(self, item);
    }

    fn visit_item_enum_mut(&mut self, item: &mut ItemEnum) {
        let item_name = item.ident.to_string();
        self.document_item(&item.ident, &mut item.attrs);

        for variant in &mut item.variants {
            let member_name = variant.ident.to_string();
            let deprecated = self
                .docs
                .find_entry(&item_name, self.symbols)
                .is_some_and(|entry| entry.find_member_deprecated(&member_name));
            self.update_deprecated_attr(&mut variant.attrs, deprecated);
            self.update_attrs(
                &mut variant.attrs,
                self.docs
                    .find_member_doc(&item_name, &member_name, self.symbols)
                    .map(ToOwned::to_owned),
            );
        }
    }

    fn visit_item_struct_mut(&mut self, item: &mut ItemStruct) {
        let item_name = item.ident.to_string();
        self.document_item(&item.ident, &mut item.attrs);

        for (index, field) in item.fields.iter_mut().enumerate() {
            self.document_field(&item_name, index, field);
        }

        visit_mut::visit_item_struct_mut(self, item);
    }

    fn visit_item_union_mut(&mut self, item: &mut ItemUnion) {
        let item_name = item.ident.to_string();
        self.document_item(&item.ident, &mut item.attrs);

        for (index, field) in item.fields.named.iter_mut().enumerate() {
            self.document_field(&item_name, index, field);
        }

        visit_mut::visit_item_union_mut(self, item);
    }

    fn visit_item_type_mut(&mut self, item: &mut ItemType) {
        self.document_item(&item.ident, &mut item.attrs);
        visit_mut::visit_item_type_mut(self, item);
    }

    fn visit_item_static_mut(&mut self, item: &mut ItemStatic) {
        self.document_item(&item.ident, &mut item.attrs);
        visit_mut::visit_item_static_mut(self, item);
    }

    fn visit_foreign_item_fn_mut(&mut self, item: &mut ForeignItemFn) {
        self.document_foreign_fn(item);
        visit_mut::visit_foreign_item_fn_mut(self, item);
    }

    fn visit_variant_mut(&mut self, _item: &mut Variant) {}
}

fn normalize_doc_text(doc: &str, symbols: &SymbolIndex) -> String {
    let doc = doc.replace("\r\n", "\n");
    let doc = doc
        .trim_start()
        .strip_prefix("!<")
        .map_or(doc.as_str(), str::trim_start);
    rewrite_non_math_doc_text(doc, symbols)
}

fn rewrite_non_math_doc_text(doc: &str, symbols: &SymbolIndex) -> String {
    let mut out = String::with_capacity(doc.len());
    let mut text_buffer = Vec::new();
    let mut in_math_block = false;

    for line in doc.lines() {
        let trimmed = line.trim();
        let is_math_delimiter = trimmed == "$$" || trimmed == "$$.";

        if in_math_block {
            if let Some(rest) = trimmed.strip_prefix("$$.") {
                flush_line(&mut out, "$$.");
                in_math_block = false;
                if !rest.trim().is_empty() {
                    text_buffer.push(rest.trim_start());
                }
                continue;
            }

            let line = escape_rustdoc_literal_brackets(line);
            let line = escape_angle_bracket_placeholders(&line);
            flush_line(&mut out, &line);
            if is_math_delimiter {
                in_math_block = false;
            }
            continue;
        }

        if is_math_delimiter {
            flush_rewritten_text(&mut out, &mut text_buffer, symbols);
            flush_line(&mut out, line);
            in_math_block = true;
            continue;
        }

        text_buffer.push(line);
    }

    flush_rewritten_text(&mut out, &mut text_buffer, symbols);
    out
}

fn flush_rewritten_text(out: &mut String, text_buffer: &mut Vec<&str>, symbols: &SymbolIndex) {
    if text_buffer.is_empty() {
        return;
    }

    let text = text_buffer.join("\n");
    let text = escape_code_fence_brackets(&text);
    let text = link_simple_use_replacement_lines(&text, symbols);
    let text = normalize_emphasized_identifiers(&text, symbols);
    let text = rewrite_backticked_symbols(&text, symbols);
    let text = rewrite_internal_markdown_links(&text, None, symbols);
    let text = rewrite_external_code_symbol_links(&text, symbols);
    let text = demote_unresolved_qualified_links(&text, symbols);
    let text = escape_rustdoc_literal_brackets(&text);
    let text = escape_angle_bracket_placeholders(&text);
    if !out.is_empty() && !text.is_empty() {
        out.push('\n');
    }
    out.push_str(&text);
    text_buffer.clear();
}

fn normalize_emphasized_identifiers(text: &str, symbols: &SymbolIndex) -> String {
    text.lines()
        .map(|line| normalize_emphasized_identifiers_in_line(line, symbols))
        .collect::<Vec<_>>()
        .join("\n")
}

fn normalize_emphasized_identifiers_in_line(line: &str, symbols: &SymbolIndex) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;

    while let Some(start) = rest.find('*') {
        out.push_str(&rest[..start]);
        let after_start = &rest[start + 1..];
        if after_start.starts_with(' ') {
            out.push('*');
            rest = after_start;
            continue;
        }

        let Some(end) = after_start.find('*') else {
            out.push('*');
            out.push_str(after_start);
            return out;
        };

        let content = &after_start[..end];
        if let Some(replacement) = normalize_emphasized_identifier(content, symbols) {
            out.push_str(&replacement);
        } else {
            out.push('*');
            out.push_str(content);
            out.push('*');
        }

        rest = &after_start[end + 1..];
    }

    out.push_str(rest);
    out
}

fn normalize_emphasized_identifier(content: &str, symbols: &SymbolIndex) -> Option<String> {
    let content = content.trim();
    if content.len() == 1 {
        return None;
    }

    let symbol = content.strip_suffix("()").unwrap_or(content);
    if !is_rust_doc_symbol(symbol) {
        return None;
    }

    Some(
        resolve_doc_symbol(symbol, symbols)
            .map_or_else(|| format!("`{content}`"), |symbol| format!("[`{symbol}`]")),
    )
}

fn escape_code_fence_brackets(text: &str) -> String {
    let mut in_fence = false;
    text.lines()
        .map(|line| {
            let trimmed = line.trim_start();
            if trimmed.starts_with("```") {
                in_fence = !in_fence;
                return line.to_string();
            }
            if in_fence {
                escape_rustdoc_literal_brackets(line)
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn link_simple_use_replacement_lines(text: &str, symbols: &SymbolIndex) -> String {
    let mut out = String::with_capacity(text.len());

    for (line_index, line) in text.lines().enumerate() {
        if line_index > 0 {
            out.push('\n');
        }

        let trimmed = line.trim_start();
        let Some(replacement) = trimmed.strip_prefix("use ") else {
            out.push_str(line);
            continue;
        };

        let symbol = replacement.trim().trim_end_matches('.');
        let Some(symbol) = resolve_doc_symbol(symbol, symbols) else {
            out.push_str(line);
            continue;
        };

        let indent_len = line.len() - trimmed.len();
        out.push_str(&line[..indent_len]);
        out.push_str("use [`");
        out.push_str(&symbol);
        out.push_str("`].");
    }

    out
}

fn flush_line(out: &mut String, line: &str) {
    if !out.is_empty() {
        out.push('\n');
    }
    out.push_str(line);
}

fn demote_unresolved_qualified_links(text: &str, symbols: &SymbolIndex) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    while let Some(start) = rest.find("[`") {
        out.push_str(&rest[..start]);
        let after_start = &rest[start + 2..];
        let Some(end) = after_start.find("`]") else {
            out.push_str("[`");
            out.push_str(after_start);
            return out;
        };

        let symbol = &after_start[..end];
        if after_start[end + 2..].starts_with('(') {
            out.push_str("[`");
            out.push_str(symbol);
            out.push_str("`]");
            rest = &after_start[end + 2..];
            continue;
        }

        if let Some(resolved) = resolve_doc_symbol(symbol, symbols) {
            out.push_str("[`");
            out.push_str(&resolved);
            out.push_str("`]");
        } else if let Some((owner, _)) = symbol.split_once("::")
            && symbols.contains_alias(owner)
        {
            out.push_str("[`");
            out.push_str(symbol);
            out.push_str("`]");
        } else {
            out.push('`');
            out.push_str(symbol);
            out.push('`');
        }

        rest = &after_start[end + 2..];
    }

    out.push_str(rest);
    out
}

fn escape_rustdoc_literal_brackets(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    while let Some(start) = rest.find('[') {
        out.push_str(&rest[..start]);
        let after_start = &rest[start + 1..];

        let Some(end_label) = after_start.find(']') else {
            out.push('[');
            out.push_str(after_start);
            return out;
        };

        let label = &after_start[..end_label];
        let after_label = &after_start[end_label + 1..];
        if out.ends_with('\\')
            || is_markdown_link_target(after_label)
            || !is_literal_bracket_label(label)
        {
            out.push('[');
            out.push_str(label);
            out.push(']');
        } else {
            out.push_str("\\[");
            out.push_str(label);
            out.push_str("\\]");
        }

        rest = after_label;
    }

    out.push_str(rest);
    out
}

fn is_markdown_link_target(text: &str) -> bool {
    text.starts_with('(')
}

fn is_literal_bracket_label(label: &str) -> bool {
    !label.is_empty() && !label.contains('`')
}

fn escape_angle_bracket_placeholders(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    while let Some(start) = rest.find('<') {
        out.push_str(&rest[..start]);
        let after_start = &rest[start + 1..];
        if after_start.starts_with("http://") || after_start.starts_with("https://") {
            out.push('<');
            if let Some(end) = after_start.find('>') {
                out.push_str(&after_start[..=end]);
                rest = &after_start[end + 1..];
            } else {
                out.push_str(after_start);
                return out;
            }
        } else {
            out.push_str("&lt;");
            rest = after_start;
        }
    }

    out.push_str(rest);

    out
}

fn rewrite_backticked_symbols(text: &str, symbols: &SymbolIndex) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    while let Some(start) = rest.find('`') {
        out.push_str(&rest[..start]);
        let after_start = &rest[start + 1..];
        let Some(end) = after_start.find('`') else {
            out.push('`');
            out.push_str(after_start);
            return out;
        };

        let content = &after_start[..end];
        let already_linked = out.ends_with('[') && after_start[end + 1..].starts_with(']');
        if already_linked {
            out.push('`');
            out.push_str(content);
            out.push('`');
        } else if let Some(link) = rewrite_backticked_symbol(content, symbols) {
            out.push_str(&link);
        } else {
            out.push('`');
            out.push_str(content);
            out.push('`');
        }

        rest = &after_start[end + 1..];
    }

    out.push_str(rest);
    out
}

fn rewrite_backticked_symbol(content: &str, symbols: &SymbolIndex) -> Option<String> {
    let symbol = content
        .trim()
        .strip_suffix("()")
        .unwrap_or_else(|| content.trim());
    if matches!(symbol, "size_t") {
        return None;
    }
    resolve_doc_symbol(symbol, symbols).map(|symbol| format!("[`{symbol}`]"))
}

fn rewrite_internal_markdown_links(
    text: &str,
    type_token: Option<&str>,
    symbols: &SymbolIndex,
) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    while let Some(start) = rest.find('[') {
        out.push_str(&rest[..start]);
        let after_start = &rest[start + 1..];

        let Some(end_label) = after_start.find(']') else {
            out.push('[');
            out.push_str(after_start);
            return out;
        };
        let after_label = &after_start[end_label + 1..];
        if let Some(after_ref_start) = after_label.strip_prefix('[')
            && let Some(end_ref) = after_ref_start.find("](")
        {
            let after_target_start = &after_ref_start[end_ref + 2..];
            if let Some(end_target) = after_target_start.find(')') {
                let label = &after_start[..end_label];
                let target = &after_target_start[..end_target];
                out.push_str("[`");
                out.push('[');
                out.push_str(label);
                out.push_str("](");
                out.push_str(target);
                out.push(')');
                rest = &after_target_start[end_target + 1..];
                continue;
            }
        }

        let Some(after_bracket) = after_label.strip_prefix('(') else {
            out.push('[');
            rest = after_start;
            continue;
        };
        let Some(end_target) = after_bracket.find(')') else {
            out.push('[');
            rest = after_start;
            continue;
        };

        let label = &after_start[..end_label];
        let target = &after_bracket[..end_target];
        if let Some(link) = rewrite_internal_doc_link(label, target, type_token, symbols) {
            out.push_str(&link);
        } else if is_external_markdown_target(target)
            && let Some(link) = rewrite_symbol_markdown_link(label, type_token, symbols)
        {
            out.push_str(&link);
        } else if !is_external_markdown_target(target) {
            out.push_str(&format_markdown_label_as_text(label, type_token, symbols));
            out.push('(');
            out.push_str(target);
            out.push(')');
        } else {
            out.push('[');
            out.push_str(label);
            out.push_str("](");
            out.push_str(target);
            out.push(')');
        }

        rest = &after_bracket[end_target + 1..];
    }

    out.push_str(rest);
    out
}

fn is_external_markdown_target(target: &str) -> bool {
    target.starts_with("http://") || target.starts_with("https://")
}

fn rewrite_external_code_symbol_links(text: &str, symbols: &SymbolIndex) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    while let Some(start) = rest.find("[`") {
        out.push_str(&rest[..start]);
        let after_start = &rest[start + 2..];
        let Some(end_label) = after_start.find("`]") else {
            out.push_str("[`");
            rest = after_start;
            continue;
        };

        let label = &after_start[..end_label];
        let after_label = &after_start[end_label + 2..];
        let Some(after_target_start) = after_label.strip_prefix('(') else {
            out.push_str("[`");
            out.push_str(label);
            out.push_str("`]");
            rest = after_label;
            continue;
        };

        let Some(end_target) = after_target_start.find(')') else {
            out.push_str("[`");
            out.push_str(after_start);
            return out;
        };

        let target = &after_target_start[..end_target];
        out.push_str("[`");
        if is_external_markdown_target(target)
            && let Some(symbol) =
                resolve_doc_symbol(label.strip_suffix("()").unwrap_or(label), symbols)
        {
            out.push_str(&symbol);
            out.push_str("`]");
        } else {
            out.push_str(label);
            out.push_str("`](");
            out.push_str(target);
            out.push(')');
        }

        rest = &after_target_start[end_target + 1..];
    }

    out.push_str(rest);
    out
}

fn rewrite_symbol_markdown_link(
    label: &str,
    type_token: Option<&str>,
    symbols: &SymbolIndex,
) -> Option<String> {
    let mut label = unescape_markdown(label).trim().to_string();
    if let Some(type_token) = type_token {
        label = expand_type_placeholder(&label, type_token);
    }

    let bare_label = label.trim_matches('`');
    let symbol = resolve_doc_symbol(bare_label.strip_suffix("()").unwrap_or(bare_label), symbols)?;
    Some(format!("[`{symbol}`]"))
}

fn format_markdown_label_as_text(
    label: &str,
    type_token: Option<&str>,
    symbols: &SymbolIndex,
) -> String {
    let mut label = unescape_markdown(label).trim().to_string();
    if let Some(type_token) = type_token {
        label = expand_type_placeholder(&label, type_token);
    }

    let bare_label = label.trim_matches('`');
    if let Some(symbol) =
        resolve_doc_symbol(bare_label.strip_suffix("()").unwrap_or(bare_label), symbols)
    {
        return format!("`{symbol}`");
    }

    label
}

fn rewrite_internal_doc_link(
    label: &str,
    target: &str,
    type_token: Option<&str>,
    symbols: &SymbolIndex,
) -> Option<String> {
    if !target.starts_with('#') {
        return None;
    }

    let mut label = unescape_markdown(label).trim().to_string();
    if let Some(type_token) = type_token {
        label = expand_type_placeholder(&label, type_token);
    }
    let symbol = resolve_doc_symbol(label.strip_suffix("()").unwrap_or(&label), symbols)?;
    Some(format!("[`{symbol}`]"))
}

fn resolve_doc_symbol(symbol: &str, symbols: &SymbolIndex) -> Option<String> {
    if !is_rust_doc_symbol(symbol) {
        return None;
    }

    doc_symbol_candidates(symbol)
        .into_iter()
        .find_map(|candidate| {
            if let Some((owner, variant)) = candidate.split_once("::") {
                if symbols.contains_name(owner) && symbols.contains_variant_path(owner, variant) {
                    return Some(candidate);
                }

                if symbols.contains_name(variant) {
                    return Some(variant.to_string());
                }
            }

            if let Some(path) = symbols.resolve_variant_path(candidate.as_str()) {
                return Some(path);
            }

            if symbols.contains_alias(candidate.as_str()) {
                return None;
            }

            if symbols.contains_name(candidate.as_str()) {
                return Some(candidate);
            }
            None
        })
}

fn doc_symbol_candidates(symbol: &str) -> Vec<String> {
    let mut candidates = vec![symbol.to_string()];

    for suffix in ["_v2", "_v3", "_v4", "_64", "_v2_64", "_v3_64", "_v4_64"] {
        candidates.push(format!("{symbol}{suffix}"));
    }

    candidates
}

fn unescape_markdown(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut escaped = false;

    for ch in text.chars() {
        if escaped {
            out.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else {
            out.push(ch);
        }
    }

    if escaped {
        out.push('\\');
    }

    out
}

fn is_rust_doc_symbol(text: &str) -> bool {
    let mut chars = text.chars();
    let Some(first) = chars.next() else {
        return false;
    };

    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }

    chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | ':'))
}
