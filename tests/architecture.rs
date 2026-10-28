//! Enforce the production file DAG, including reexports and qualified calls.
//! Unit-test modules are consumers of production code, not runtime dependencies.
//! This checks source-level Rust paths; it does not expand macros or resolve types.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use syn::visit::{self, Visit};

type Module = Vec<String>;
type Graph = BTreeMap<Module, BTreeSet<Module>>;

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
    edges: BTreeSet<Module>,
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
        let (mut resolved, base, rooted) = self.absolute(parts);
        while resolved.len() > base {
            if self.modules.contains(&resolved) {
                if resolved != self.owner {
                    self.edges.insert(resolved);
                }
                return;
            }
            resolved.pop();
        }
        // Explicitly rooted symbols can be defined in the base module:
        // `super::Type`, `crate::version`, or a glob of that module.
        if rooted && resolved != self.owner && self.modules.contains(&resolved) {
            self.edges.insert(resolved);
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
                let mut path = prefix.clone();
                if n.ident != "self" {
                    path.push(n.ident.to_string());
                }
                match self.binding {
                    true => self.bind(n.rename.to_string(), &path),
                    false => self.path(&path),
                }
            }
            syn::UseTree::Glob(_) if !self.binding => self.path(prefix),
            syn::UseTree::Glob(_) => {}
        }
    }
}

impl<'ast> Visit<'ast> for References<'_> {
    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        if item.attrs.iter().any(|a| {
            a.path().is_ident("cfg")
                && a.parse_args::<syn::Path>()
                    .is_ok_and(|p| p.is_ident("test"))
        }) {
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

fn dependencies(files: &BTreeMap<Module, String>) -> Graph {
    let modules = files.keys().cloned().collect();
    files
        .iter()
        .map(|(owner, source)| {
            let file = syn::parse_file(source).unwrap();
            let mut refs = References {
                modules: &modules,
                owner: owner.clone(),
                scope: owner.clone(),
                edges: BTreeSet::new(),
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
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut paths = vec![];
    sources(&src, &mut paths);
    let files = paths
        .iter()
        .map(|p| {
            (
                module_of(p.strip_prefix(&src).unwrap()),
                std::fs::read_to_string(p).unwrap(),
            )
        })
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
