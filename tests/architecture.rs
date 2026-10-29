//! Enforce the production file DAG, including reexports and qualified calls.
//! Unit-test modules are consumers of production code, not runtime dependencies.
//! This checks source-level Rust paths; it does not expand macros or resolve types.
//! The checked graph is written to `target/tmp/module-graph.json`.

use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use syn::spanned::Spanned;
use syn::visit::{self, Visit};

type Module = Vec<String>;
type Graph = BTreeMap<Module, BTreeSet<Module>>;
/// The uses of each other module: line, the item after the module's path
/// (`*` for a glob, empty for the module itself) and the path as written.
type Uses = BTreeMap<Module, Vec<(usize, String, String)>>;

fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

fn module_of(path: &Path) -> Module {
    let mut parts: Module = path
        .with_extension("")
        .components()
        .map(|p| p.as_os_str().to_str().unwrap().to_owned())
        .collect();
    if matches!(parts.last().map(String::as_str), Some("lib" | "mod")) {
        parts.pop();
    }
    parts
}

struct References<'a> {
    modules: &'a BTreeSet<Module>,
    owner: Module,
    scope: Module,
    edges: Uses,
    line: usize,
    /// The module each `use` of a module binds, by scope and name.
    aliases: BTreeMap<(Module, String), Module>,
    /// Record `aliases` only, before any path is read.
    binding: bool,
}

impl References<'_> {
    /// `parts` from the crate root, the length of its root, and whether
    /// that root is explicit: `crate`, `self`, `super` or a `use`d module.
    fn absolute(&self, parts: &[String]) -> (Module, usize, bool) {
        let Some(first) = parts.first() else {
            return (vec![], 0, false);
        };
        let alias = self.aliases.get(&(self.scope.clone(), first.clone()));
        let (mut resolved, mut rest, rooted) = match (first.as_str(), alias) {
            ("crate" | "ptxroof", _) => (vec![], &parts[1..], true),
            ("self", _) => (self.scope.clone(), &parts[1..], true),
            ("super", _) => (self.scope.clone(), parts, true),
            (_, Some(module)) => (module.clone(), &parts[1..], true),
            _ => (self.scope.clone(), parts, false),
        };
        while rest.first().is_some_and(|s| s == "super") {
            resolved.pop().expect("super has a parent module");
            rest = &rest[1..];
        }
        let base = resolved.len();
        resolved.extend_from_slice(rest);
        (resolved, base, rooted)
    }

    fn path(&mut self, parts: &[String]) {
        // A bare local type or external crate must not fall back to the
        // enclosing module and invent an edge to a facade.
        let (full, base, rooted) = self.absolute(parts);
        let mut resolved = full.clone();
        while resolved.len() > base {
            if self.modules.contains(&resolved) {
                return self.record(resolved, &full, parts);
            }
            resolved.pop();
        }
        // Explicitly rooted symbols can be defined in the base module:
        // `super::Type`, `crate::version`, or a glob of that module.
        if rooted && self.modules.contains(&resolved) {
            self.record(resolved, &full, parts);
        }
    }

    fn record(&mut self, target: Module, full: &Module, parts: &[String]) {
        if target != self.owner {
            let item = full.get(target.len()).cloned().unwrap_or_default();
            let site = (self.line, item, parts.join("::"));
            self.edges.entry(target).or_default().push(site);
        }
    }

    fn bind(&mut self, name: String, path: &[String]) {
        let (module, ..) = self.absolute(path);
        if self.modules.contains(&module) {
            self.aliases.insert((self.scope.clone(), name), module);
        }
    }

    fn import(&mut self, tree: &syn::UseTree, prefix: &mut Module) {
        match tree {
            syn::UseTree::Path(p) => {
                self.line = p.ident.span().start().line;
                prefix.push(p.ident.to_string());
                self.import(&p.tree, prefix);
                prefix.pop();
            }
            syn::UseTree::Group(g) => {
                for item in &g.items {
                    self.import(item, prefix);
                }
            }
            syn::UseTree::Name(n) => {
                self.line = n.ident.span().start().line;
                let mut path = prefix.clone();
                if n.ident != "self" {
                    path.push(n.ident.to_string());
                }
                match (self.binding, path.last()) {
                    (true, Some(name)) => self.bind(name.clone(), &path),
                    (true, None) => {}
                    (false, _) => self.path(&path),
                }
            }
            syn::UseTree::Rename(n) => {
                self.line = n.ident.span().start().line;
                let mut path = prefix.clone();
                if n.ident != "self" {
                    path.push(n.ident.to_string());
                }
                match self.binding {
                    true => self.bind(n.rename.to_string(), &path),
                    false => self.path(&path),
                }
            }
            syn::UseTree::Glob(_) if !self.binding => {
                let mut path = prefix.clone();
                path.push("*".into());
                self.path(&path);
            }
            syn::UseTree::Glob(_) => {}
        }
    }
}

fn is_test(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| {
        a.path().is_ident("cfg")
            && a.parse_args::<syn::Path>()
                .is_ok_and(|p| p.is_ident("test"))
    })
}

impl<'ast> Visit<'ast> for References<'_> {
    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        if is_test(&item.attrs) {
            return;
        }
        // `mod child;` declares ownership, not a use of its implementation.
        if let Some((_, items)) = &item.content {
            self.scope.push(item.ident.to_string());
            for item in items {
                self.visit_item(item);
            }
            self.scope.pop();
        }
    }

    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        self.import(&item.tree, &mut vec![]);
    }

    fn visit_path(&mut self, path: &'ast syn::Path) {
        if self.binding {
            return;
        }
        self.line = path.span().start().line;
        self.path(
            &path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect::<Module>(),
        );
        visit::visit_path(self, path);
    }

    // `pub(in crate::analysis)` is visibility, not a dependency.
    fn visit_visibility(&mut self, _: &'ast syn::Visibility) {}
}

fn references(files: &BTreeMap<Module, String>) -> BTreeMap<Module, Uses> {
    let modules = files.keys().cloned().collect();
    files
        .iter()
        .map(|(owner, source)| {
            let file = syn::parse_file(source).unwrap();
            let mut refs = References {
                modules: &modules,
                owner: owner.clone(),
                scope: owner.clone(),
                edges: BTreeMap::new(),
                line: 0,
                aliases: BTreeMap::new(),
                binding: true,
            };
            refs.visit_file(&file);
            refs.binding = false;
            refs.visit_file(&file);
            (owner.clone(), refs.edges)
        })
        .collect()
}

fn dependencies(files: &BTreeMap<Module, String>) -> Graph {
    references(files)
        .into_iter()
        .map(|(owner, uses)| (owner, uses.into_keys().collect()))
        .collect()
}

/// Each production file by module: its path in the package and its source.
fn production() -> BTreeMap<Module, (String, String)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let src = root.join("src");
    let mut paths = vec![];
    sources(&src, &mut paths);
    paths
        .iter()
        .map(|p| {
            let path = p.strip_prefix(root).unwrap().to_str().unwrap().to_owned();
            let source = std::fs::read_to_string(p).unwrap();
            (module_of(p.strip_prefix(&src).unwrap()), (path, source))
        })
        .collect()
}

fn item(out: &mut Vec<Value>, name: String, kind: &str, span: proc_macro2::Span) {
    out.push(json!({"name": name, "kind": kind, "line": span.start().line}));
}

/// The items a macro call defines, `newtype_idx! { pub struct BlockId; }`.
fn macro_items(tokens: proc_macro2::TokenStream, out: &mut Vec<Value>) {
    let mut kind = None;
    for token in tokens {
        match token {
            proc_macro2::TokenTree::Group(g) => macro_items(g.stream(), out),
            proc_macro2::TokenTree::Ident(i) => match kind.take() {
                Some(k) => item(out, i.to_string(), k, i.span()),
                None => {
                    let word = i.to_string();
                    kind = ["struct", "enum", "type", "trait", "fn"]
                        .into_iter()
                        .find(|k| *k == word);
                }
            },
            proc_macro2::TokenTree::Punct(_) | proc_macro2::TokenTree::Literal(_) => kind = None,
        }
    }
}

fn reexports(tree: &syn::UseTree, prefix: &mut Vec<String>, out: &mut Vec<Value>) {
    let mut push = |name: String, from: String, span: proc_macro2::Span| {
        out.push(
            json!({"name": name, "kind": "reexport", "line": span.start().line, "from": from}),
        );
    };
    match tree {
        syn::UseTree::Path(p) => {
            prefix.push(p.ident.to_string());
            reexports(&p.tree, prefix, out);
            prefix.pop();
        }
        syn::UseTree::Group(g) => g.items.iter().for_each(|t| reexports(t, prefix, out)),
        syn::UseTree::Name(n) => push(
            n.ident.to_string(),
            format!("{}::{}", prefix.join("::"), n.ident),
            n.ident.span(),
        ),
        syn::UseTree::Rename(n) => push(
            n.rename.to_string(),
            format!("{}::{}", prefix.join("::"), n.ident),
            n.ident.span(),
        ),
        syn::UseTree::Glob(g) => push(
            "*".into(),
            format!("{}::*", prefix.join("::")),
            g.star_token.span(),
        ),
    }
}

/// The items a file defines outside `#[cfg(test)]`, methods as `Type::name`.
fn items(file: &syn::File) -> Vec<Value> {
    let mut out = vec![];
    for it in &file.items {
        match it {
            syn::Item::Fn(f) if !is_test(&f.attrs) => {
                item(&mut out, f.sig.ident.to_string(), "fn", f.sig.ident.span())
            }
            syn::Item::Struct(s) => item(&mut out, s.ident.to_string(), "struct", s.ident.span()),
            syn::Item::Enum(e) => item(&mut out, e.ident.to_string(), "enum", e.ident.span()),
            syn::Item::Type(t) => item(&mut out, t.ident.to_string(), "type", t.ident.span()),
            syn::Item::Trait(t) => item(&mut out, t.ident.to_string(), "trait", t.ident.span()),
            syn::Item::Const(c) => item(&mut out, c.ident.to_string(), "const", c.ident.span()),
            syn::Item::Static(s) => item(&mut out, s.ident.to_string(), "static", s.ident.span()),
            syn::Item::Mod(m) if !is_test(&m.attrs) => {
                item(&mut out, m.ident.to_string(), "mod", m.ident.span())
            }
            syn::Item::Macro(m) => match &m.ident {
                Some(id) => item(&mut out, id.to_string(), "macro", id.span()),
                None => macro_items(m.mac.tokens.clone(), &mut out),
            },
            syn::Item::Use(u) if !matches!(u.vis, syn::Visibility::Inherited) => {
                reexports(&u.tree, &mut vec![], &mut out)
            }
            syn::Item::Impl(i) if !is_test(&i.attrs) => {
                if let syn::Type::Path(t) = &*i.self_ty {
                    let ty = &t.path.segments.last().unwrap().ident;
                    for f in i.items.iter().filter_map(|f| match f {
                        syn::ImplItem::Fn(f) => Some(f),
                        _ => None,
                    }) {
                        item(
                            &mut out,
                            format!("{ty}::{}", f.sig.ident),
                            "fn",
                            f.sig.ident.span(),
                        );
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// The `//!` doc's paragraphs.
fn module_doc(file: &syn::File) -> Vec<String> {
    let mut paragraphs = vec![String::new()];
    for a in &file.attrs {
        if let syn::Meta::NameValue(nv) = &a.meta
            && nv.path.is_ident("doc")
            && let syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(s),
                ..
            }) = &nv.value
        {
            let line = s.value().trim().to_owned();
            let last = paragraphs.last_mut().unwrap();
            match (line.is_empty(), last.is_empty()) {
                (true, true) => {}
                (true, false) => paragraphs.push(String::new()),
                (false, true) => *last = line,
                (false, false) => *last = format!("{last} {line}"),
            }
        }
    }
    paragraphs.into_iter().filter(|p| !p.is_empty()).collect()
}

/// Lines of code before the file's `#[cfg(test)]` module, not blank and
/// not a comment.
fn code_lines(file: &syn::File, source: &str) -> usize {
    let tests = (file.items.iter())
        .find(|i| matches!(i, syn::Item::Mod(m) if is_test(&m.attrs)))
        .map_or(usize::MAX, |m| m.span().start().line);
    (source.lines().take(tests.saturating_sub(1)))
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with("//"))
        .count()
}

fn cycle(graph: &Graph) -> Option<Vec<Module>> {
    fn walk(
        graph: &Graph,
        node: &Module,
        done: &mut BTreeSet<Module>,
        stack: &mut Vec<Module>,
    ) -> Option<Vec<Module>> {
        if let Some(start) = stack.iter().position(|m| m == node) {
            let mut path = stack[start..].to_vec();
            path.push(node.clone());
            return Some(path);
        }
        if done.contains(node) {
            return None;
        }
        stack.push(node.clone());
        for next in &graph[node] {
            if let Some(path) = walk(graph, next, done, stack) {
                return Some(path);
            }
        }
        stack.pop();
        done.insert(node.clone());
        None
    }
    let mut done = BTreeSet::new();
    for node in graph.keys() {
        if let Some(path) = walk(graph, node, &mut done, &mut vec![]) {
            return Some(path);
        }
    }
    None
}

#[test]
fn production_files_form_a_dag() {
    let files = (production().into_iter())
        .map(|(module, (_, source))| (module, source))
        .collect();
    let graph = dependencies(&files);
    if let Some(path) = cycle(&graph) {
        panic!(
            "production dependency cycle: {}",
            path.iter()
                .map(|m| m.join("::"))
                .collect::<Vec<_>>()
                .join(" -> ")
        );
    }
    for (from, targets) in &graph {
        for to in targets {
            let allowed = match (
                from.first().map(String::as_str),
                to.first().map(String::as_str),
            ) {
                (Some("support"), to) => to == Some("support"),
                (Some("ptx"), to) => matches!(to, Some("ptx" | "support")),
                (Some("analysis" | "ssa"), Some("report" | "main")) => false,
                (Some("analysis" | "ssa"), Some("ptx")) => to.get(1).is_none_or(|m| m != "parse"),
                _ => true,
            };
            assert!(
                allowed,
                "forbidden dependency: {} -> {}",
                from.join("::"),
                to.join("::")
            );
        }
    }
}

#[test]
fn checker_follows_reexports_and_qualified_calls_but_excludes_tests() {
    let files = BTreeMap::from([
        (
            vec!["a".into()],
            "pub use crate::b::{run as forwarded};".into(),
        ),
        (vec!["b".into()], "fn run() { crate::c::run(); }".into()),
        (vec!["c".into()], "use crate::a::*;".into()),
    ]);
    assert!(cycle(&dependencies(&files)).is_some());
    let mut acyclic = files;
    acyclic.insert(
        vec!["c".into()],
        "#[cfg(test)] mod tests { use crate::a::*; }".into(),
    );
    assert!(cycle(&dependencies(&acyclic)).is_none());
}

#[test]
fn checker_follows_parent_symbols_and_ignores_visibility_and_ownership() {
    let root = vec![];
    let child = vec!["child".to_owned()];
    let mut files = BTreeMap::from([
        (root.clone(), "mod child; struct Input;".into()),
        (
            child.clone(),
            "use super::Input; pub(crate) fn run() {}".into(),
        ),
    ]);
    let graph = dependencies(&files);
    assert_eq!(graph[&child], BTreeSet::from([root.clone()]));
    assert!(graph[&root].is_empty());
    files.insert(
        root,
        "mod child; struct Input; fn run() { self::child::run(); }".into(),
    );
    assert!(cycle(&dependencies(&files)).is_some());
}

#[test]
fn checker_follows_paths_through_a_used_module() {
    let files = |a: &str| {
        BTreeMap::from([
            (vec!["a".into()], a.to_owned()),
            (vec!["b".into()], "pub mod c;".into()),
            (
                vec!["b".into(), "c".into()],
                "use crate::a::f; pub fn run() {}".into(),
            ),
        ])
    };
    for a in [
        "use crate::b; fn f() { b::c::run(); }",
        "use crate::b::{self as m}; fn f() { m::c::run(); }",
        "fn f() { use crate::b; b::c::run(); }",
    ] {
        assert!(cycle(&dependencies(&files(a))).is_some(), "{a}");
    }
    let unused = "use crate::b; fn f() { c::run(); }";
    assert!(cycle(&dependencies(&files(unused))).is_none());
}

/// `target/tmp/module-graph.json`: the graph `production_files_form_a_dag`
/// checks, each use's line, and each file's items, `//!` doc and size.
#[test]
fn the_checked_graph_is_written_out() {
    let files = production();
    let sources = (files.iter())
        .map(|(module, (_, source))| (module.clone(), source.clone()))
        .collect();
    let name = |m: &Module| match m.is_empty() {
        true => "crate".to_owned(),
        false => m.join("::"),
    };
    let modules: Vec<Value> = (files.iter())
        .map(|(module, (path, source))| {
            let file = syn::parse_file(source).unwrap();
            json!({
                "id": name(module),
                "file": path,
                "doc": module_doc(&file),
                "code": code_lines(&file, source),
                "items": items(&file),
            })
        })
        .collect();
    let edges: Vec<Value> = (references(&sources).iter())
        .flat_map(|(from, uses)| {
            (uses.iter())
                .map(|(to, sites)| json!({"from": name(from), "to": name(to), "uses": sites}))
        })
        .collect();
    let git = |args: &[&str]| {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .ok()
            .filter(|o| o.status.success())?;
        Some(String::from_utf8_lossy(&out.stdout).trim().to_owned())
    };
    let graph = json!({
        "root": env!("CARGO_MANIFEST_DIR"),
        "commit": git(&["rev-parse", "HEAD"]),
        "dirty": git(&["status", "--porcelain", "--", "src"]).map(|s| !s.is_empty()),
        "modules": modules,
        "edges": edges,
    });
    let path = concat!(env!("CARGO_TARGET_TMPDIR"), "/module-graph.json");
    std::fs::write(path, graph.to_string()).unwrap();
}
