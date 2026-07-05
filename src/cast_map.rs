//! Safe wrapper around [`UnsafeCastMapG`](crate::unsafe_cast_map::UnsafeCastMapG)
//! whose keyed lookups are checked against each slot's stored concrete type id.
//!
//! Where the low-level map's `get` / `get_mut` / `remove` are `unsafe` (the
//! caller must promise the key's metadata fits the slot), [`CastMapG`] makes
//! them safe: every value lives in a box that records its concrete
//! [`TypeId`] (see [`ConcreteTypeId`]), and a lookup recovers the type id
//! implied by the key's metadata (via [`type_id_from_meta`]) and compares it to
//! the slot's. A mismatch — wrong type, recycled slot, or a key minted by
//! another map for a different type — returns `None` instead of risking UB.
//!
//! Soundness, briefly: `slotmap`'s version check proves the slot is occupied
//! and current; the type-id check proves the key's metadata describes the
//! concrete type actually stored there. Together they make rebuilding `&T`
//! from data pointer + key metadata sound, with **no per-map identity**: keys
//! are plain [`CastKey`]s, and even a key from a different map is memory-safe
//! (it resolves if — and only if — the slot it names holds a value of the
//! key's type).
//!
//! This safety hinges on a stored type id, so the checked lookups require
//! `M::Value: ConcreteTypeId` — satisfied by [`CastBox`] (e.g. [`BoxCastMap`])
//! or any custom box that implements [`ConcreteTypeId`]. A plain `Box` won't
//! do; that is what [`UnsafeCastMapG`] is for.
//!
//! On the key side, lookups require `T: AnyHaver`: sized types always qualify;
//! trait objects qualify when the trait declares `AnyHaver` as a supertrait.
//! `dyn Any` does not, so `get::<dyn Any>` is a compile error — use
//! [`downcast_key`](CastMapG::downcast_key) or
//! [`get_by_inner_key`](CastMapG::get_by_inner_key) for erased access.
//!
//! [`CastMapG`] is generic over the backing `slotmap` map `M`
//! ([`slotmap::SlotMap`] or [`slotmap::DenseSlotMap`]); the backing key and
//! stored pointer types are read off `M` as `M::Key` and `M::Value`. The
//! concrete maps are exposed as aliases: [`CastMap`] (sparse) and
//! [`DenseCastMap`] (dense), plus the [`CastBox`]-storing [`BoxCastMap`] /
//! [`BoxDenseCastMap`].
//!
//! `detach` / `reattach` are deliberately **not** offered here: reattaching a
//! value of a different type would desynchronize the stored type id from the
//! slot. They live on the unsafe
//! [`UnsafeCastMapG`](crate::unsafe_cast_map::UnsafeCastMapG), reachable via
//! [`inner_mut`](CastMapG::inner_mut) if you accept that `unsafe` contract.

use std::any::{Any, TypeId};
use std::ops::{Deref, DerefMut};
use std::ptr::Pointee;

use slotmap::{DenseSlotMap, SlotMap};
use stable_deref_trait::StableDeref;

use crate::any_haver::{type_id_from_meta, AnyHaver};
use crate::cast_box::{CastBox, ConcreteTypeId};
use crate::cast_key::CastKey;
use crate::retype_ptr::RetypePtr;
use crate::slotmap_trait::{MTarget, SlotMapTrait};
use crate::unsafe_cast_map::{self, UnsafeCastMapG};

// ─── CastMapG ────────────────────────────────────────────────────────────────

/// A safe wrapper around [`UnsafeCastMapG`] that validates keyed lookups
/// against each slot's stored concrete [`TypeId`].
///
/// Its one type parameter is the backing `slotmap` map `M`; the stored smart
/// pointer is `M::Value` (e.g. `CastBox<dyn Any>`) and the output type is
/// `<M::Value as Deref>::Target`. Use the aliases [`CastMap`] / [`DenseCastMap`]
/// (or the [`CastBox`] forms [`BoxCastMap`] / [`BoxDenseCastMap`]) rather than
/// naming `M` directly.
pub struct CastMapG<M> {
    inner: UnsafeCastMapG<M>,
}

// ─── Clone ───────────────────────────────────────────────────────────────────

impl<M> Clone for CastMapG<M>
where
    M: SlotMapTrait + Clone,
{
    /// Clones the map. Because lookups are validated by slot version and
    /// stored type id — not per-map identity — keys from the original resolve
    /// on the clone too (they name the same slots holding the same types).
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        self.inner.clone_from(&source.inner);
    }
}

impl<M> Default for CastMapG<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref,
{
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ─── Basic methods ───────────────────────────────────────────────────────────

impl<M> CastMapG<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref,
{
    /// Creates a new, empty map.
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: UnsafeCastMapG::new(),
        }
    }

    /// Creates a new map with the given pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: UnsafeCastMapG::with_capacity(capacity),
        }
    }

    /// Returns true if the map is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the number of occupied elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns how many slots the backing storage can hold before reallocating.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Reserves capacity for at least `additional` more elements.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
    }

    /// Tries to reserve capacity for at least `additional` more elements.
    #[inline]
    pub fn try_reserve(
        &mut self,
        additional: usize,
    ) -> Result<(), std::collections::TryReserveError> {
        self.inner.try_reserve(additional)
    }

    /// Removes all elements from the map. Outstanding keys are invalidated.
    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    // ── inner accessors ───────────────────────────────────────────────────

    /// Consumes this map and returns the underlying [`UnsafeCastMapG`].
    #[inline]
    pub fn inner(self) -> UnsafeCastMapG<M> {
        self.inner
    }

    /// Returns a shared reference to the underlying [`UnsafeCastMapG`].
    #[inline]
    pub fn inner_ref(&self) -> &UnsafeCastMapG<M> {
        &self.inner
    }

    /// Returns a mutable reference to the underlying [`UnsafeCastMapG`].
    #[inline]
    pub fn inner_mut(&mut self) -> &mut UnsafeCastMapG<M> {
        &mut self.inner
    }

    // ── backing-key access (no type check needed: output-typed) ────────────

    /// Shared-reference lookup using the backing `slotmap` key directly.
    #[inline]
    pub fn get_by_inner_key(&self, key: M::Key) -> Option<&MTarget<M>> {
        self.inner.get_by_inner_key(key)
    }

    /// Removes an element by its backing `slotmap` key.
    #[inline]
    pub fn remove_by_inner_key(&mut self, key: M::Key) -> Option<M::Value> {
        self.inner.remove_by_inner_key(key)
    }

    /// Shared iterator over output references only.
    #[inline]
    pub fn values(&self) -> impl Iterator<Item = &MTarget<M>> + '_ {
        self.inner.values()
    }
}

// ── backing-key access requiring `&mut Output` ───────────────────────────────

impl<M> CastMapG<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref + DerefMut,
{
    /// Mutable-reference lookup using the backing `slotmap` key directly.
    #[inline]
    pub fn get_by_inner_key_mut(&mut self, key: M::Key) -> Option<&mut MTarget<M>> {
        self.inner.get_by_inner_key_mut(key)
    }

    /// Mutable iterator over output references only.
    #[inline]
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut MTarget<M>> + '_ {
        self.inner.values_mut()
    }

    /// Mutable disjoint lookup by backing `slotmap` keys. Returns `None` if any
    /// key is invalid or two keys alias the same slot.
    #[inline]
    pub fn get_disjoint_mut_by_inner_key<const N: usize>(
        &mut self,
        keys: [M::Key; N],
    ) -> Option<[&mut MTarget<M>; N]> {
        self.inner.get_disjoint_mut_by_inner_key(keys)
    }

    /// Like [`get_disjoint_mut_by_inner_key`](Self::get_disjoint_mut_by_inner_key)
    /// but without validity or uniqueness checks.
    ///
    /// # Safety
    /// Every key must address a live slot, and no two keys may alias one slot.
    #[inline]
    pub unsafe fn get_disjoint_unchecked_mut_by_inner_key<const N: usize>(
        &mut self,
        keys: [M::Key; N],
    ) -> [&mut MTarget<M>; N] {
        self.inner.get_disjoint_unchecked_mut_by_inner_key(keys)
    }
}

// ─── Core operations (require pointer metadata) ──────────────────────────────

impl<M> CastMapG<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    // ── insert ───────────────────────────────────────────────────────────

    /// Inserts a value and returns its [`CastKey`].
    #[inline]
    pub fn insert(&mut self, value: M::Value) -> CastKey<MTarget<M>, M::Key> {
        self.inner.insert(value)
    }

    /// Inserts a value produced by `func`, which receives the backing key.
    #[inline]
    pub fn insert_with_key(
        &mut self,
        func: impl FnOnce(M::Key) -> M::Value,
    ) -> CastKey<MTarget<M>, M::Key> {
        self.inner.insert_with_key(func)
    }

    /// Like [`insert_with_key`](Self::insert_with_key) but the closure may fail.
    #[inline]
    pub fn try_insert_with_key<E>(
        &mut self,
        func: impl FnOnce(M::Key) -> Result<M::Value, E>,
    ) -> Result<CastKey<MTarget<M>, M::Key>, E> {
        self.inner.try_insert_with_key(func)
    }

    // ── insert_sized ─────────────────────────────────────────────────────

    /// Inserts a concrete-typed smart pointer (coerced into `M::Value` on the
    /// way in), returning a [`CastKey`] typed to the concrete
    /// `ConcretePtr::Target` rather than the erased output type.
    #[inline]
    pub fn insert_sized<ConcretePtr>(
        &mut self,
        value: ConcretePtr,
    ) -> CastKey<ConcretePtr::Target, M::Key>
    where
        ConcretePtr: std::ops::CoerceUnsized<M::Value> + Deref,
        ConcretePtr::Target: Sized,
    {
        self.inner.insert_sized(value)
    }

    /// Like [`insert_sized`](Self::insert_sized) but the closure receives the
    /// typed key the value will live under.
    #[inline]
    pub fn insert_sized_with_key<ConcretePtr>(
        &mut self,
        func: impl FnOnce(CastKey<ConcretePtr::Target, M::Key>) -> ConcretePtr,
    ) -> CastKey<ConcretePtr::Target, M::Key>
    where
        ConcretePtr: std::ops::CoerceUnsized<M::Value> + Deref,
        ConcretePtr::Target: Sized,
    {
        self.inner.insert_sized_with_key(func)
    }

    /// Fallible version of [`insert_sized_with_key`](Self::insert_sized_with_key).
    #[inline]
    pub fn try_insert_sized_with_key<ConcretePtr, E>(
        &mut self,
        func: impl FnOnce(CastKey<ConcretePtr::Target, M::Key>) -> Result<ConcretePtr, E>,
    ) -> Result<CastKey<ConcretePtr::Target, M::Key>, E>
    where
        ConcretePtr: std::ops::CoerceUnsized<M::Value> + Deref,
        ConcretePtr::Target: Sized,
    {
        self.inner.try_insert_sized_with_key(func)
    }

    // ── insert_as ────────────────────────────────────────────────────────

    /// Inserts a smart pointer whose (possibly unsized) target differs from
    /// the map's output type, returning a key typed with the *source* type.
    #[inline]
    pub fn insert_as<SourcePtr>(
        &mut self,
        value: SourcePtr,
    ) -> CastKey<SourcePtr::Target, M::Key>
    where
        SourcePtr: std::ops::CoerceUnsized<M::Value> + StableDeref,
        SourcePtr::Target: Pointee<Metadata: Copy>,
    {
        self.inner.insert_as(value)
    }

    /// Inserts a smart pointer produced by `func`, returning a key typed with
    /// the source `SourcePtr::Target`. The closure receives the backing key.
    #[inline]
    pub fn insert_as_with_key<SourcePtr>(
        &mut self,
        func: impl FnOnce(M::Key) -> SourcePtr,
    ) -> CastKey<SourcePtr::Target, M::Key>
    where
        SourcePtr: std::ops::CoerceUnsized<M::Value> + StableDeref,
        SourcePtr::Target: Pointee<Metadata: Copy>,
    {
        self.inner.insert_as_with_key(func)
    }

    /// Fallible version of [`insert_as_with_key`](Self::insert_as_with_key).
    #[inline]
    pub fn try_insert_as_with_key<SourcePtr, E>(
        &mut self,
        func: impl FnOnce(M::Key) -> Result<SourcePtr, E>,
    ) -> Result<CastKey<SourcePtr::Target, M::Key>, E>
    where
        SourcePtr: std::ops::CoerceUnsized<M::Value> + StableDeref,
        SourcePtr::Target: Pointee<Metadata: Copy>,
    {
        self.inner.try_insert_as_with_key(func)
    }

    // ── cast_key_of ──────────────────────────────────────────────────────

    /// Converts a backing `slotmap` key into a [`CastKey`] by reading pointer
    /// metadata from the stored value. Returns `None` if the key is stale.
    #[inline]
    pub fn cast_key_of(&self, key: M::Key) -> Option<CastKey<MTarget<M>, M::Key>> {
        self.inner.cast_key_of(key)
    }

    // ── iterators ────────────────────────────────────────────────────────

    /// Lazy iterator over all [`CastKey`]s.
    #[inline]
    pub fn keys(&self) -> impl Iterator<Item = CastKey<MTarget<M>, M::Key>> + '_ {
        self.inner.keys()
    }

    /// Shared iterator over all occupied `(CastKey, &output)` pairs.
    #[inline]
    pub fn iter(&self) -> Iter<'_, M> {
        self.inner.iter()
    }

    /// Draining iterator. Removes all elements and yields them.
    #[inline]
    pub fn drain(&mut self) -> Drain<'_, M> {
        self.inner.drain()
    }
}

// ─── Checked typed lookups (safe — type-id validated) ────────────────────────

impl<M> CastMapG<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref + ConcreteTypeId,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    /// Mints a concrete-typed key from a backing `slotmap` key by comparing
    /// the slot's stored type id with `TypeId::of::<Concrete>()`. Returns
    /// `None` if the key is stale or the stored type differs.
    ///
    /// Takes the backing key directly (get one from any `CastKey` via
    /// [`inner_key`](CastKey::inner_key)): the check only needs the slot, so
    /// no pointer metadata is required.
    #[inline]
    pub fn downcast_key<Concrete: 'static>(
        &self,
        key: M::Key,
    ) -> Option<CastKey<Concrete, M::Key>> {
        let stored = self.inner.inner.get(key)?;
        if stored.concrete_type_id() == TypeId::of::<Concrete>() {
            // SAFETY: `()` metadata is trivially valid for the sized
            // `Concrete`, which the slot was just proven to hold.
            Some(unsafe { CastKey::from_raw_parts(key, ()) })
        } else {
            None
        }
    }

    /// Returns whether the key still resolves in this map: its slot is live
    /// *and* holds a value of the key's type.
    #[inline]
    pub fn contains_key<T: ?Sized + AnyHaver + Pointee>(
        &self,
        key: CastKey<T, M::Key>,
    ) -> bool
    where
        <T as Pointee>::Metadata: Copy,
    {
        self.get(key).is_some()
    }

    /// Typed lookup by [`CastKey`]. Returns `None` if the slot is
    /// vacant, the key is stale, or the key's type does not match the value at
    /// that slot.
    #[inline]
    pub fn get<T: ?Sized + AnyHaver + Pointee>(
        &self,
        key: CastKey<T, M::Key>,
    ) -> Option<&T>
    where
        <T as Pointee>::Metadata: Copy,
    {
        let stored = self.inner.inner.get(key.inner_key())?;
        let stored_tid = stored.concrete_type_id();
        let base: &MTarget<M> = &**stored;
        if stored_tid != type_id_from_meta::<T>(key.metadata()) {
            return None;
        }
        let data: *const () = (base as *const MTarget<M>).cast();
        // SAFETY: version check passed (slot live) and the key's
        // metadata-implied concrete type equals the stored concrete type, so
        // `metadata` is valid for the value behind `data`.
        Some(unsafe { &*std::ptr::from_raw_parts::<T>(data, key.metadata()) })
    }

    /// Shared-reference lookup without bounds, version, or type checks.
    ///
    /// # Safety
    /// - The key's slot must be occupied with the matching version.
    /// - The key's pointer metadata must be valid for the data in that slot.
    #[inline]
    pub unsafe fn get_unchecked<T: ?Sized + Pointee>(&self, key: CastKey<T, M::Key>) -> &T
    where
        <T as Pointee>::Metadata: Copy,
    {
        self.inner.get_unchecked(key)
    }

    /// Removes an element by its [`CastKey`], returning the owned smart
    /// pointer re-typed to `T`. Returns `None` if the key is stale or its type
    /// does not match the slot.
    #[inline]
    pub fn remove<'a, T: ?Sized + AnyHaver + Pointee>(
        &mut self,
        key: CastKey<T, M::Key>,
    ) -> Option<<M::Value as RetypePtr<'a>>::Retyped<T>>
    where
        <T as Pointee>::Metadata: Copy,
        M::Value: RetypePtr<'a>,
    {
        let stored = self.inner.inner.get(key.inner_key())?;
        if stored.concrete_type_id() != type_id_from_meta::<T>(key.metadata()) {
            return None;
        }
        // SAFETY: the type-id check just proved the key's metadata is valid
        // for the value in that slot.
        unsafe { self.inner.remove(key) }
    }
}

// ─── Checked operations requiring `&mut Output` ──────────────────────────────

impl<M> CastMapG<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref + DerefMut + ConcreteTypeId,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    /// Mutable typed lookup by [`CastKey`].
    /// Type-id validated, like [`get`](Self::get).
    #[inline]
    pub fn get_mut<T: ?Sized + AnyHaver + Pointee>(
        &mut self,
        key: CastKey<T, M::Key>,
    ) -> Option<&mut T>
    where
        <T as Pointee>::Metadata: Copy,
    {
        let stored = self.inner.inner.get_mut(key.inner_key())?;
        if stored.concrete_type_id() != type_id_from_meta::<T>(key.metadata()) {
            return None;
        }
        let base: &mut MTarget<M> = &mut **stored;
        let data: *mut () = (base as *mut MTarget<M>).cast();
        // SAFETY: as in `get`, version + type-id checks passed.
        Some(unsafe { &mut *std::ptr::from_raw_parts_mut::<T>(data, key.metadata()) })
    }

    /// Mutable-reference lookup without bounds, version, or type checks.
    ///
    /// # Safety
    /// - The key's slot must be occupied with the matching version.
    /// - The key's pointer metadata must be valid for the data in that slot.
    #[inline]
    pub unsafe fn get_unchecked_mut<T: ?Sized + Pointee>(
        &mut self,
        key: CastKey<T, M::Key>,
    ) -> &mut T
    where
        <T as Pointee>::Metadata: Copy,
    {
        self.inner.get_unchecked_mut(key)
    }

    /// Retains only elements for which `f(key, &mut output)` returns `true`.
    #[inline]
    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(CastKey<MTarget<M>, M::Key>, &mut MTarget<M>) -> bool,
    {
        self.inner.retain(f);
    }

    /// Mutable iterator over all occupied `(CastKey, &mut output)` pairs.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, M> {
        self.inner.iter_mut()
    }

    /// Mutable disjoint lookup typed by the keys' `T`, which may differ from
    /// the map's output type. All keys must share the pointee type `T`.
    /// Returns `None` if any key is stale, mistyped for its slot, or two keys
    /// alias the same slot.
    #[inline]
    pub fn get_disjoint_mut<T: ?Sized + AnyHaver + Pointee, const N: usize>(
        &mut self,
        keys: [CastKey<T, M::Key>; N],
    ) -> Option<[&mut T; N]>
    where
        <T as Pointee>::Metadata: Copy,
    {
        // Validate every key's type against its slot first (shared borrows),
        // then hand off to the unsafe disjoint lookup, which enforces
        // liveness and pairwise disjointness.
        for key in &keys {
            let stored = self.inner.inner.get(key.inner_key())?;
            if stored.concrete_type_id() != type_id_from_meta::<T>(key.metadata()) {
                return None;
            }
        }
        // SAFETY: each key's metadata was just validated for its slot.
        unsafe { self.inner.get_disjoint_mut(keys) }
    }

    /// Like [`get_disjoint_mut`](Self::get_disjoint_mut) but without validity,
    /// uniqueness, or type checks.
    ///
    /// # Safety
    /// - Every key must address a live slot, and no two keys may alias one slot.
    /// - Each key's pointer metadata must be valid for the data in its slot.
    #[inline]
    pub unsafe fn get_disjoint_unchecked_mut<T: ?Sized + Pointee, const N: usize>(
        &mut self,
        keys: [CastKey<T, M::Key>; N],
    ) -> [&mut T; N]
    where
        <T as Pointee>::Metadata: Copy,
    {
        self.inner.get_disjoint_unchecked_mut(keys)
    }
}

// ─── Index / IndexMut ────────────────────────────────────────────────────────

impl<M> std::ops::Index<CastKey<MTarget<M>, M::Key>> for CastMapG<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref + ConcreteTypeId,
    MTarget<M>: Pointee + AnyHaver,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    type Output = MTarget<M>;

    #[inline]
    fn index(&self, key: CastKey<MTarget<M>, M::Key>) -> &Self::Output {
        self.get(key).expect("invalid CastKey for this map")
    }
}

impl<M> std::ops::IndexMut<CastKey<MTarget<M>, M::Key>> for CastMapG<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref + DerefMut + ConcreteTypeId,
    MTarget<M>: Pointee + AnyHaver,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    #[inline]
    fn index_mut(&mut self, key: CastKey<MTarget<M>, M::Key>) -> &mut Self::Output {
        self.get_mut(key).expect("invalid CastKey for this map")
    }
}

// ─── Iterators ───────────────────────────────────────────────────────────────

// With per-map identity gone, keys need no re-wrapping: the checked map's
// iterators are the unsafe map's iterators.

/// Shared iterator over `(CastKey, &Target)` pairs.
pub type Iter<'a, M> = unsafe_cast_map::Iter<'a, M>;
/// Mutable iterator over `(CastKey, &mut Target)` pairs.
pub type IterMut<'a, M> = unsafe_cast_map::IterMut<'a, M>;
/// Draining iterator over `(CastKey, value)`, emptying the map.
pub type Drain<'a, M> = unsafe_cast_map::Drain<'a, M>;
/// Owning iterator over `(CastKey, value)` pairs.
pub type IntoIter<M> = unsafe_cast_map::IntoIter<M>;

impl<M> IntoIterator for CastMapG<M>
where
    M: SlotMapTrait,
    M::Value: StableDeref,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    type Item = (CastKey<MTarget<M>, M::Key>, M::Value);
    type IntoIter = IntoIter<M>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl<'a, M> IntoIterator for &'a CastMapG<M>
where
    M: SlotMapTrait + 'a,
    M::Value: StableDeref + 'a,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    type Item = (CastKey<MTarget<M>, M::Key>, &'a MTarget<M>);
    type IntoIter = Iter<'a, M>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, M> IntoIterator for &'a mut CastMapG<M>
where
    M: SlotMapTrait + 'a,
    M::Value: StableDeref + DerefMut + 'a,
    MTarget<M>: Pointee,
    <MTarget<M> as Pointee>::Metadata: Copy,
{
    type Item = (CastKey<MTarget<M>, M::Key>, &'a mut MTarget<M>);
    type IntoIter = IterMut<'a, M>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter_mut()
    }
}

// ─── Type aliases ────────────────────────────────────────────────────────────

/// Safe castable-key map backed by [`slotmap::SlotMap`] (sparse storage).
pub type CastMap<K, Ptr> = CastMapG<SlotMap<K, Ptr>>;

/// Safe castable-key map backed by [`slotmap::DenseSlotMap`] (contiguous
/// storage, fast iteration).
pub type DenseCastMap<K, Ptr> = CastMapG<DenseSlotMap<K, Ptr>>;

/// Convenience alias: [`CastMap`] storing [`CastBox<T>`] (e.g. `dyn Any`).
/// `CastBox` implements [`ConcreteTypeId`], which the checked lookups require.
pub type BoxCastMap<K, T> = CastMap<K, CastBox<T>>;

/// Convenience alias: [`DenseCastMap`] storing [`CastBox<T>`] (e.g. `dyn Any`).
pub type BoxDenseCastMap<K, T> = DenseCastMap<K, CastBox<T>>;
