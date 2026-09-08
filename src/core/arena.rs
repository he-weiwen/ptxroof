//! Typed index arenas (the flat-IR ground rule).
//!
//! The flat IR stores objects in module-level `Vec`s and refers to them
//! by integer handle, never by pointer. This module gives those handles
//! *types*: a handle minted for one arena cannot index another, and the
//! two list pools that used to share an untyped `Span` can no longer be
//! crossed — the mistake is now a compile error rather than a silent
//! read of the wrong pool.
//!
//! Modeled on rustc's `rustc_index` (`IndexVec<I, T>`, the `Idx` trait,
//! and `newtype_index!`), reduced to what this crate uses and
//! hand-rolled to honor the no-dependencies rule (§2). Two shapes:
//!   * [`IndexVec<I, T>`] — one handle per element (the operand arena,
//!     the CFG block arena).
//!   * [`IdxRange<I>`] — a typed `(start, len)` slice into a flat pool
//!     (the operand-list and modifier pools).

use std::marker::PhantomData;

/// A `u32` handle into an [`IndexVec`]. Mint distinct handle types with
/// [`newtype_idx!`] so the handles for different arenas are themselves
/// distinct types.
pub trait Idx: Copy {
    fn from_usize(i: usize) -> Self;
    fn index(self) -> usize;
}

/// Declare a `u32`-backed [`Idx`] newtype — this crate's stripped-down
/// `newtype_index!`. The `.0` field stays public so existing `Foo(0)`
/// construction and `id.0` reads keep working.
macro_rules! newtype_idx {
    ($(#[$m:meta])* $vis:vis struct $name:ident;) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        $vis struct $name(pub u32);

        impl $crate::core::arena::Idx for $name {
            #[inline]
            fn from_usize(i: usize) -> Self {
                debug_assert!(i <= u32::MAX as usize, "arena index overflowed u32");
                $name(i as u32)
            }
            #[inline]
            fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}
pub(crate) use newtype_idx;

/// A flat arena: push a `T` and get back its typed handle `I`; resolve a
/// handle with `arena[id]`. A handle minted for one instantiation cannot
/// index another — the wrong index type fails to compile.
#[derive(Debug, Clone)]
pub struct IndexVec<I: Idx, T> {
    raw: Vec<T>,
    // `I` appears in no field, so the compiler needs this zero-size tag
    // to track it and keep `IndexVec<A, _>` and `IndexVec<B, _>` apart.
    // Spelling copied verbatim from rustc's `IndexVec`.
    _marker: PhantomData<fn(&I)>,
}

impl<I: Idx, T> IndexVec<I, T> {
    pub fn new() -> Self {
        IndexVec {
            raw: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Append `v`, returning the handle that now resolves to it.
    #[inline]
    pub fn push(&mut self, v: T) -> I {
        let id = I::from_usize(self.raw.len());
        self.raw.push(v);
        id
    }

    pub fn len(&self) -> usize {
        self.raw.len()
    }

    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.raw.iter()
    }

    /// Iterate `(handle, &element)` pairs — the typed replacement for
    /// `vec.iter().enumerate()` with a hand-built `Id(i as u32)`.
    pub fn iter_enumerated(&self) -> impl Iterator<Item = (I, &T)> + '_ {
        self.raw
            .iter()
            .enumerate()
            .map(|(i, t)| (I::from_usize(i), t))
    }
}

impl<I: Idx, T> Default for IndexVec<I, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Idx, T> std::ops::Index<I> for IndexVec<I, T> {
    type Output = T;
    #[inline]
    fn index(&self, id: I) -> &T {
        &self.raw[id.index()]
    }
}

impl<I: Idx, T> std::ops::IndexMut<I> for IndexVec<I, T> {
    #[inline]
    fn index_mut(&mut self, id: I) -> &mut T {
        &mut self.raw[id.index()]
    }
}

/// A typed half-open range `[start, start + len)` into a flat pool. The
/// parameter records *which* pool the range belongs to: a range over the
/// operand-list pool is `IdxRange<OperandId>` (the pool holds
/// `OperandId`s) and a range over the modifier pool is `IdxRange<Symbol>`
/// — distinct types, so the two can no longer be passed interchangeably.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdxRange<I> {
    start: u32,
    len: u32,
    _marker: PhantomData<fn(&I)>,
}

impl<I> IdxRange<I> {
    #[inline]
    pub fn new(start: u32, len: u32) -> Self {
        IdxRange {
            start,
            len,
            _marker: PhantomData,
        }
    }

    /// The backing-pool slice indices this range covers.
    #[inline]
    pub fn range(self) -> std::ops::Range<usize> {
        self.start as usize..(self.start + self.len) as usize
    }

    pub fn len(self) -> usize {
        self.len as usize
    }

    pub fn is_empty(self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    newtype_idx! { struct FooId; }
    newtype_idx! { struct BarId; }

    #[test]
    fn push_returns_resolvable_handle() {
        let mut a: IndexVec<FooId, &str> = IndexVec::new();
        let x = a.push("zero");
        let y = a.push("one");
        assert_eq!(a[x], "zero");
        assert_eq!(a[y], "one");
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn iter_enumerated_yields_typed_handles() {
        let mut a: IndexVec<FooId, u8> = IndexVec::new();
        a.push(10);
        a.push(20);
        let pairs: Vec<(FooId, u8)> = a.iter_enumerated().map(|(i, &t)| (i, t)).collect();
        assert_eq!(pairs, [(FooId(0), 10), (FooId(1), 20)]);
    }

    // The payoff, asserted as a doc-level fact rather than runtime: a
    // `FooId` cannot index a `BarId` arena, and `IdxRange<FooId>` is not
    // an `IdxRange<BarId>`. Both lines fail to compile if uncommented —
    // which is the whole point of giving handles types.
    //
    //     let bars: IndexVec<BarId, u8> = IndexVec::new();
    //     let _ = bars[FooId(0)];                 // E0308 mismatched types
    //     let _: IdxRange<BarId> = IdxRange::<FooId>::new(0, 1); // E0308
    #[test]
    fn distinct_ranges_have_distinct_types() {
        let f: IdxRange<FooId> = IdxRange::new(2, 3);
        let _b: IdxRange<BarId> = IdxRange::new(2, 3);
        assert_eq!(f.range(), 2..5);
        assert_eq!(f.len(), 3);
        assert!(!f.is_empty());
        // Both handle types resolve to the same raw index, yet they are
        // different types — which is exactly what keeps `FooId` from
        // indexing a `BarId` arena.
        assert_eq!(FooId(0).index(), BarId(0).index());
    }
}
