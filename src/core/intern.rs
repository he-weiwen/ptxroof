//! Hand-rolled string interner (the flat-IR ground rule).
//!
//! All mnemonics, modifiers, and identifiers in the module IR are
//! `Symbol(u32)`; the classifier and every later consumer match on
//! integers. The interner is module-owned and owns its strings — PTX
//! inputs are at most a few MB, so there is no lifetime threading.

use std::collections::HashMap;

/// Interned string handle. Index into the owning [`Interner`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Symbol(pub u32);

#[derive(Debug, Default)]
pub struct Interner {
    strings: Vec<String>,
    map: HashMap<String, Symbol>,
}

impl Interner {
    pub fn intern(&mut self, text: &str) -> Symbol {
        if let Some(&sym) = self.map.get(text) {
            return sym;
        }
        let sym = Symbol(self.strings.len() as u32);
        self.strings.push(text.to_owned());
        self.map.insert(text.to_owned(), sym);
        sym
    }

    pub fn resolve(&self, sym: Symbol) -> &str {
        &self.strings[sym.0 as usize]
    }

    /// Look up an already-interned string without inserting — for
    /// read-only consumers of a built module (CFG, classifier), which
    /// hold `&Module` and compare symbols by id.
    pub fn get(&self, text: &str) -> Option<Symbol> {
        self.map.get(text).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_is_idempotent_and_resolve_round_trips() {
        let mut i = Interner::default();
        let a = i.intern("fma");
        let b = i.intern("fma");
        let c = i.intern("ld");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(i.resolve(a), "fma");
        assert_eq!(i.resolve(c), "ld");
    }
}
