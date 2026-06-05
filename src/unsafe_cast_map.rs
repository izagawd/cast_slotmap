//! Low-level cast map over [`slotmap::SlotMap`] without per-map identity checks.
//!
//! [`UnsafeCastMap`] supports typed lookups via [`CastKey`], but `get`,
//! `get_mut`, `remove`, and `downcast_key` are **`unsafe`**: the caller must
//! ensure the key's pointer metadata is valid for the data stored at that slot.
//! For a safe wrapper that checks a per-map [`MapId`](crate::map_id::MapId), see
//! [`CastMap`](crate::cast_map::CastMap).
//!
//! ## Relationship to `slotmap`
//! Every method forwards to the underlying `slotmap::SlotMap`. Mutating methods
//! (`insert*`, `remove`, `reserve`, `clear`, `retain`, `drain`) take `&mut self`
//! because that is `SlotMap`'s signature. There is intentionally **no**
//! `get_slot`, `get_by_index_only`, `reset`, or `unsafe_clone`/`clone_mut`
//! family here: `slotmap` exposes none of those, and faking them would mean
//! reaching past its public API. For the same reason `iter` is a plain safe
//! shared iterator (no `unsafe_iter`): `slotmap::SlotMap::get` borrows `&self`
//! while `insert` borrows `&mut self`, so a live reference can never coexist
//! with an insert.

use std::any::{Any, TypeId};
use std::collections::TryReserveError;
use std::ops::{Deref, DerefMut};
use std::ptr::Pointee;

use slotmap::{Key, SlotMap};

use crate::cast_key::CastKey;
use crate::retype_ptr::RetypePtr;
use stable_deref_trait::StableDeref;

// ─── Conversion helper ───────────────────────────────────────────────────────

/// Build a cast key from a `slotmap` key and a reference (for pointer metadata).
#[inline]
fn to_castable<K: Key, O: ?Sized + Pointee>(key: K, reference: &O) -> CastKey<O, K>
where
    <O as Pointee>::Metadata: Copy,
{
    let metadata = std::ptr::metadata(reference as *const O);
    CastKey { key, metadata }
}

// ─── UnsafeCastMap ───────────────────────────────────────────────────────

/// A [`slotmap::SlotMap`] wrapper that supports typed lookups via [`CastKey`].
///
/// `Ptr` is the stored smart pointer (e.g. `Box<dyn Any>`); it must implement
/// [`StableDeref`] so that pointer-metadata casts are sound. The map's "output"
/// type is `<Ptr as Deref>::Target`.
pub struct UnsafeCastMap<K: Key, Ptr>
where
    Ptr: StableDeref,
{
    pub(crate) inner: SlotMap<K, Ptr>,
}

// ─── Clone ───────────────────────────────────────────────────────────────────

impl<K: Key, Ptr> Clone for UnsafeCastMap<K, Ptr>
where
    Ptr: StableDeref,
    SlotMap<K, Ptr>: Clone,
{
    /// Cloning preserves every slot's key and version, so keys valid on the
    /// original stay valid on the clone (the safe
    /// [`CastMap`](crate::cast_map::CastMap) layer, by contrast, mints a fresh
    /// [`MapId`](crate::map_id::MapId) on clone).
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

impl<K: Key, Ptr> Default for UnsafeCastMap<K, Ptr>
where
    Ptr: StableDeref,
{
    fn default() -> Self {
        Self::new()
    }
}

// ─── Basic methods (no pointer metadata needed) ──────────────────────────────

impl<K: Key, Ptr> UnsafeCastMap<K, Ptr>
where
    Ptr: StableDeref,
{
    /// Creates a new, empty map.
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: SlotMap::with_key(),
        }
    }

    /// Creates a new map with the given pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: SlotMap::with_capacity_and_key(capacity),
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
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.inner.try_reserve(additional)
    }

    /// Removes all elements from the map. Outstanding keys are invalidated
    /// (`slotmap` bumps slot versions on clear).
    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Returns whether the backing key is still live (delegates to
    /// [`slotmap::SlotMap::contains_key`]).
    #[inline]
    pub fn contains_key<T: ?Sized + Pointee>(&self, key: CastKey<T, K>) -> bool
    where
        <T as Pointee>::Metadata: Copy,
    {
        self.inner.contains_key(key.inner_key())
    }

    // ── backing-key access ───────────────────────────────────────────────

    /// Shared-reference lookup using the backing `slotmap` key directly.
    #[inline]
    pub fn get_by_inner_key(&self, key: K) -> Option<&Ptr::Target> {
        self.inner.get(key).map(|p| &**p)
    }

    /// Removes an element by its backing `slotmap` key, returning the pointer.
    #[inline]
    pub fn remove_by_inner_key(&mut self, key: K) -> Option<Ptr> {
        self.inner.remove(key)
    }

    /// Shared iterator over output references only.
    #[inline]
    pub fn values(&self) -> impl Iterator<Item = &Ptr::Target> + '_ {
        self.inner.values().map(|p| &**p)
    }
}

// ── backing-key access requiring `&mut Output` ───────────────────────────────

impl<K: Key, Ptr> UnsafeCastMap<K, Ptr>
where
    Ptr: StableDeref + DerefMut,
{
    /// Mutable-reference lookup using the backing `slotmap` key directly.
    #[inline]
    pub fn get_by_inner_key_mut(&mut self, key: K) -> Option<&mut Ptr::Target> {
        self.inner.get_mut(key).map(|p| &mut **p)
    }

    /// Mutable iterator over output references only.
    #[inline]
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut Ptr::Target> + '_ {
        self.inner.values_mut().map(|p| &mut **p)
    }

    /// Mutable disjoint lookup by backing `slotmap` keys, yielding erased output
    /// references. Returns `None` if any key is invalid or two keys alias the
    /// same slot.
    #[inline]
    pub fn get_disjoint_mut_by_inner_key<const N: usize>(
        &mut self,
        keys: [K; N],
    ) -> Option<[&mut Ptr::Target; N]> {
        let stored = self.inner.get_disjoint_mut(keys)?;
        Some(stored.map(|p| &mut **p))
    }

    /// Like [`get_disjoint_mut_by_inner_key`](Self::get_disjoint_mut_by_inner_key)
    /// but without validity or uniqueness checks.
    ///
    /// # Safety
    /// Every key must be valid for this map and no two keys may alias one slot.
    #[inline]
    pub unsafe fn get_disjoint_unchecked_mut_by_inner_key<const N: usize>(
        &mut self,
        keys: [K; N],
    ) -> [&mut Ptr::Target; N] {
        let stored = self.inner.get_disjoint_unchecked_mut(keys);
        stored.map(|p| &mut **p)
    }
}

// ─── Core operations (require pointer metadata) ──────────────────────────────

impl<K: Key, Ptr> UnsafeCastMap<K, Ptr>
where
    Ptr: StableDeref,
    Ptr::Target: Pointee,
    <Ptr::Target as Pointee>::Metadata: Copy,
{
    /// Attempts to downcast a `CastKey<dyn Any, ..>` to a concrete-typed key.
    /// Returns `None` if the actual type doesn't match `Concrete`.
    ///
    /// # Safety
    /// The key's metadata must be a valid vtable for `dyn Any` pointing at the
    /// data stored in that slot.
    #[inline]
    pub unsafe fn downcast_key<Concrete: 'static>(
        &self,
        key: CastKey<dyn Any, K>,
    ) -> Option<CastKey<Concrete, K>> {
        let stored = self.inner.get(key.inner_key())?;
        let base: &Ptr::Target = &**stored;
        let data_as_any: &dyn Any =
            &*std::ptr::from_raw_parts(base as *const Ptr::Target as *const (), key.metadata());
        if data_as_any.type_id() == TypeId::of::<Concrete>() {
            Some(CastKey {
                key: key.key,
                metadata: (),
            })
        } else {
            None
        }
    }

    // ── insert ───────────────────────────────────────────────────────────

    /// Inserts a smart pointer, returning a key with metadata.
    #[inline]
    pub fn insert(&mut self, value: Ptr) -> CastKey<Ptr::Target, K> {
        self.insert_with_key(|_| value)
    }

    /// Inserts a smart pointer produced by `func`, which receives the backing
    /// key that will identify the inserted element.
    #[inline]
    pub fn insert_with_key(&mut self, func: impl FnOnce(K) -> Ptr) -> CastKey<Ptr::Target, K> {
        self.try_insert_with_key(|key| Ok::<_, ()>(func(key)))
            .unwrap()
    }

    /// Like [`insert_with_key`](Self::insert_with_key) but the closure may
    /// return `Err`, in which case nothing is inserted.
    #[inline]
    pub fn try_insert_with_key<E>(
        &mut self,
        func: impl FnOnce(K) -> Result<Ptr, E>,
    ) -> Result<CastKey<Ptr::Target, K>, E> {
        let inner_key = self.inner.try_insert_with_key(func)?;
        let reference = self
            .inner
            .get(inner_key)
            .expect("just-inserted key is live");
        Ok(to_castable::<K, Ptr::Target>(inner_key, &**reference))
    }

    // ── insert_sized ─────────────────────────────────────────────────────

    /// Inserts a concrete-typed smart pointer, returning a [`CastKey`] whose
    /// metadata is for `ConcretePtr::Target` (not the erased output).
    #[inline]
    pub fn insert_sized<ConcretePtr>(
        &mut self,
        value: ConcretePtr,
    ) -> CastKey<ConcretePtr::Target, K>
    where
        ConcretePtr: std::ops::CoerceUnsized<Ptr> + Deref,
        ConcretePtr::Target: Sized,
    {
        self.insert_sized_with_key(|_| value)
    }

    /// Inserts a concrete smart pointer produced by `func`, which receives the
    /// fully-typed [`CastKey`].
    #[inline]
    pub fn insert_sized_with_key<ConcretePtr>(
        &mut self,
        func: impl FnOnce(CastKey<ConcretePtr::Target, K>) -> ConcretePtr,
    ) -> CastKey<ConcretePtr::Target, K>
    where
        ConcretePtr: std::ops::CoerceUnsized<Ptr> + Deref,
        ConcretePtr::Target: Sized,
    {
        self.try_insert_sized_with_key(|key| Ok::<_, ()>(func(key)))
            .unwrap()
    }

    /// Like [`insert_sized_with_key`](Self::insert_sized_with_key) but the
    /// closure may return `Err`.
    #[inline]
    pub fn try_insert_sized_with_key<ConcretePtr, E>(
        &mut self,
        func: impl FnOnce(CastKey<ConcretePtr::Target, K>) -> Result<ConcretePtr, E>,
    ) -> Result<CastKey<ConcretePtr::Target, K>, E>
    where
        ConcretePtr: std::ops::CoerceUnsized<Ptr> + Deref,
        ConcretePtr::Target: Sized,
    {
        let mut saved_key: Option<CastKey<ConcretePtr::Target, K>> = None;

        self.inner.try_insert_with_key(|k| -> Result<Ptr, E> {
            // `ConcretePtr::Target: Sized` => its pointer metadata is `()`.
            let typed_key = CastKey {
                key: k,
                metadata: (),
            };
            saved_key = Some(typed_key);
            let concrete: ConcretePtr = func(typed_key)?;
            Ok(concrete)
        })?;

        Ok(saved_key.unwrap())
    }

    // ── insert_as ──────────────────────────────────────────────────────────

    /// Inserts a smart pointer whose target type differs from the map's output,
    /// returning a key typed with the *source* type.
    #[inline]
    pub fn insert_as<SourcePtr>(&mut self, value: SourcePtr) -> CastKey<SourcePtr::Target, K>
    where
        SourcePtr: std::ops::CoerceUnsized<Ptr> + Deref,
        SourcePtr::Target: Pointee<Metadata: Copy>,
    {
        self.insert_as_with_key(|_| value)
    }

    /// Inserts a smart pointer produced by `func`, returning a key typed with
    /// the source `SourcePtr::Target`.
    #[inline]
    pub fn insert_as_with_key<SourcePtr>(
        &mut self,
        func: impl FnOnce(K) -> SourcePtr,
    ) -> CastKey<SourcePtr::Target, K>
    where
        SourcePtr: std::ops::CoerceUnsized<Ptr> + Deref,
        SourcePtr::Target: Pointee<Metadata: Copy>,
    {
        self.try_insert_as_with_key(|key| Ok::<_, ()>(func(key)))
            .unwrap()
    }

    /// Like [`insert_as_with_key`](Self::insert_as_with_key) but the closure may
    /// return `Err`.
    #[inline]
    pub fn try_insert_as_with_key<SourcePtr, E>(
        &mut self,
        func: impl FnOnce(K) -> Result<SourcePtr, E>,
    ) -> Result<CastKey<SourcePtr::Target, K>, E>
    where
        SourcePtr: std::ops::CoerceUnsized<Ptr> + Deref,
        SourcePtr::Target: Pointee<Metadata: Copy>,
    {
        let mut saved_metadata: Option<<SourcePtr::Target as Pointee>::Metadata> = None;

        let inner_key = self.inner.try_insert_with_key(|k| -> Result<Ptr, E> {
            let concrete: SourcePtr = func(k)?;
            saved_metadata = Some(std::ptr::metadata(&*concrete as *const SourcePtr::Target));
            Ok(concrete)
        })?;

        let metadata = saved_metadata.unwrap();
        Ok(CastKey {
            key: inner_key,
            metadata,
        })
    }

    // ── cast_key_of ──────────────────────────────────────────────────────

    /// Converts a backing `slotmap` key into a full [`CastKey`] by reading the
    /// stored value's pointer metadata. Returns `None` if the key is stale.
    #[inline]
    pub fn cast_key_of(&self, key: K) -> Option<CastKey<Ptr::Target, K>> {
        let reference = self.inner.get(key)?;
        Some(to_castable::<K, Ptr::Target>(key, &**reference))
    }

    // ── typed lookups (shared) ─────────────────────────────────────────────

    /// Cross-typed shared-reference lookup. Reconstructs a fat pointer to `T`
    /// from the stored output's data pointer and the key's metadata.
    ///
    /// # Safety
    /// The key's pointer metadata must be valid for the data stored at that
    /// slot (e.g. for a trait object, the correct vtable for the concrete type).
    #[inline]
    pub unsafe fn get<T: ?Sized + Pointee>(&self, key: CastKey<T, K>) -> Option<&T>
    where
        <T as Pointee>::Metadata: Copy,
    {
        let stored = self.inner.get(key.inner_key())?;
        let base: &Ptr::Target = &**stored;
        let data_ptr: *const () = (base as *const Ptr::Target).cast();
        let fat_ptr: *const T = std::ptr::from_raw_parts(data_ptr, key.metadata());
        Some(&*fat_ptr)
    }

    /// Shared-reference lookup without bounds or version checks.
    ///
    /// # Safety
    /// - The key's slot must be occupied with the matching version.
    /// - The key's pointer metadata must be valid for the data in that slot.
    #[inline]
    pub unsafe fn get_unchecked<T: ?Sized + Pointee>(&self, key: CastKey<T, K>) -> &T
    where
        <T as Pointee>::Metadata: Copy,
    {
        let stored = self.inner.get_unchecked(key.inner_key());
        let base: &Ptr::Target = &**stored;
        let data_ptr: *const () = (base as *const Ptr::Target).cast();
        let fat_ptr: *const T = std::ptr::from_raw_parts(data_ptr, key.metadata());
        &*fat_ptr
    }

    /// Removes an element by its [`CastKey`], returning the owned smart pointer
    /// re-typed to `T`.
    ///
    /// # Safety
    /// The key's pointer metadata must be valid for the data stored at that slot.
    #[inline]
    pub unsafe fn remove<'a, T: ?Sized + Pointee>(
        &mut self,
        key: CastKey<T, K>,
    ) -> Option<<Ptr as RetypePtr<'a>>::Retyped<T>>
    where
        <T as Pointee>::Metadata: Copy,
        Ptr: RetypePtr<'a>,
    {
        let stored = self.inner.remove(key.inner_key())?;
        Some(stored.retype::<T>(key.metadata()))
    }

    // ── iterators ────────────────────────────────────────────────────────

    /// Lazy iterator over all [`CastKey`]s.
    #[inline]
    pub fn keys(&self) -> impl Iterator<Item = CastKey<Ptr::Target, K>> + '_ {
        self.inner
            .iter()
            .map(|(k, p)| to_castable::<K, Ptr::Target>(k, &**p))
    }

    /// Shared iterator over all occupied `(CastKey, &output)` pairs (safe).
    #[inline]
    pub fn iter(&self) -> Iter<'_, K, Ptr> {
        Iter {
            inner: self.inner.iter(),
        }
    }

    // ── drain ────────────────────────────────────────────────────────────

    /// Draining iterator. Removes all elements and yields them.
    #[inline]
    pub fn drain(&mut self) -> Drain<'_, K, Ptr> {
        Drain {
            inner: self.inner.drain(),
        }
    }
}

// ─── Core operations requiring `&mut Output` ─────────────────────────────────

impl<K: Key, Ptr> UnsafeCastMap<K, Ptr>
where
    Ptr: StableDeref + DerefMut,
    Ptr::Target: Pointee,
    <Ptr::Target as Pointee>::Metadata: Copy,
{
    /// Cross-typed mutable-reference lookup.
    ///
    /// # Safety
    /// The key's pointer metadata must be valid for the data stored at that slot.
    #[inline]
    pub unsafe fn get_mut<T: ?Sized + Pointee>(&mut self, key: CastKey<T, K>) -> Option<&mut T>
    where
        <T as Pointee>::Metadata: Copy,
    {
        let stored = self.inner.get_mut(key.inner_key())?;
        let base: &mut Ptr::Target = &mut **stored;
        let data_ptr: *mut () = (base as *mut Ptr::Target).cast();
        let fat_ptr: *mut T = std::ptr::from_raw_parts_mut(data_ptr, key.metadata());
        Some(&mut *fat_ptr)
    }

    /// Mutable-reference lookup without bounds or version checks.
    ///
    /// # Safety
    /// - The key's slot must be occupied with the matching version.
    /// - The key's pointer metadata must be valid for the data in that slot.
    #[inline]
    pub unsafe fn get_unchecked_mut<T: ?Sized + Pointee>(&mut self, key: CastKey<T, K>) -> &mut T
    where
        <T as Pointee>::Metadata: Copy,
    {
        let stored = self.inner.get_unchecked_mut(key.inner_key());
        let base: &mut Ptr::Target = &mut **stored;
        let data_ptr: *mut () = (base as *mut Ptr::Target).cast();
        let fat_ptr: *mut T = std::ptr::from_raw_parts_mut(data_ptr, key.metadata());
        &mut *fat_ptr
    }

    /// Retains only elements for which `f(key, &mut output)` returns `true`.
    #[inline]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(CastKey<Ptr::Target, K>, &mut Ptr::Target) -> bool,
    {
        self.inner.retain(|inner_key, stored| {
            let patched = to_castable::<K, Ptr::Target>(inner_key, &**stored);
            f(patched, &mut **stored)
        })
    }

    /// Mutable iterator over all occupied `(CastKey, &mut output)` pairs (safe).
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, Ptr> {
        IterMut {
            inner: self.inner.iter_mut(),
        }
    }

    /// Cross-typed mutable disjoint lookup. All keys must share the pointee type
    /// `T`; each fat pointer to `T` is rebuilt from that key's own metadata.
    /// Returns `None` if any key is invalid or two keys alias the same slot.
    ///
    /// # Safety
    /// Each key's pointer metadata must be valid for the data in its slot.
    #[inline]
    pub unsafe fn get_disjoint_mut<T: ?Sized + Pointee, const N: usize>(
        &mut self,
        keys: [CastKey<T, K>; N],
    ) -> Option<[&mut T; N]>
    where
        <T as Pointee>::Metadata: Copy,
    {
        let metadata = keys.map(|k| k.metadata());
        let inner_keys = keys.map(|k| k.inner_key());
        let stored = self.inner.get_disjoint_mut(inner_keys)?;
        let mut i = 0;
        let out = stored.map(|p| {
            let meta = metadata[i];
            i += 1;
            let base: &mut Ptr::Target = &mut **p;
            let data_ptr: *mut () = (base as *mut Ptr::Target).cast();
            unsafe { &mut *std::ptr::from_raw_parts_mut(data_ptr, meta) }
        });
        Some(out)
    }

    /// Like [`get_disjoint_mut`](Self::get_disjoint_mut) but without validity or
    /// uniqueness checks.
    ///
    /// # Safety
    /// - Every key must be valid for this map and no two keys may alias one slot.
    /// - Each key's pointer metadata must be valid for the data in its slot.
    #[inline]
    pub unsafe fn get_disjoint_unchecked_mut<T: ?Sized + Pointee, const N: usize>(
        &mut self,
        keys: [CastKey<T, K>; N],
    ) -> [&mut T; N]
    where
        <T as Pointee>::Metadata: Copy,
    {
        let metadata = keys.map(|k| k.metadata());
        let inner_keys = keys.map(|k| k.inner_key());
        let stored = self.inner.get_disjoint_unchecked_mut(inner_keys);
        let mut i = 0;
        stored.map(|p| {
            let meta = metadata[i];
            i += 1;
            let base: &mut Ptr::Target = &mut **p;
            let data_ptr: *mut () = (base as *mut Ptr::Target).cast();
            unsafe { &mut *std::ptr::from_raw_parts_mut(data_ptr, meta) }
        })
    }
}

/// Convenience alias: [`UnsafeCastMap`] storing `Box<T>` (e.g. `dyn Any`).
pub type UnsafeBoxCastMap<K, T> = UnsafeCastMap<K, Box<T>>;

// ─── Iter (shared) ───────────────────────────────────────────────────────────

pub struct Iter<'a, K: Key, Ptr: StableDeref>
where
    K: 'a,
    Ptr: 'a,
{
    inner: slotmap::basic::Iter<'a, K, Ptr>,
}

impl<'a, K: Key, Ptr> Iterator for Iter<'a, K, Ptr>
where
    K: 'a,
    Ptr: StableDeref + 'a,
    Ptr::Target: Pointee,
    <Ptr::Target as Pointee>::Metadata: Copy,
{
    type Item = (CastKey<Ptr::Target, K>, &'a Ptr::Target);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (k, p) = self.inner.next()?;
        let r: &'a Ptr::Target = &**p;
        Some((to_castable::<K, Ptr::Target>(k, r), r))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

// ─── IterMut ─────────────────────────────────────────────────────────────────

pub struct IterMut<'a, K: Key, Ptr: StableDeref>
where
    K: 'a,
    Ptr: 'a,
{
    inner: slotmap::basic::IterMut<'a, K, Ptr>,
}

impl<'a, K: Key, Ptr> Iterator for IterMut<'a, K, Ptr>
where
    K: 'a,
    Ptr: StableDeref + DerefMut + 'a,
    Ptr::Target: Pointee,
    <Ptr::Target as Pointee>::Metadata: Copy,
{
    type Item = (CastKey<Ptr::Target, K>, &'a mut Ptr::Target);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (k, stored) = self.inner.next()?;
        let patched = to_castable::<K, Ptr::Target>(k, &**stored);
        Some((patched, &mut **stored))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

// ─── Drain ───────────────────────────────────────────────────────────────────

pub struct Drain<'a, K: Key, Ptr: StableDeref>
where
    K: 'a,
    Ptr: 'a,
{
    inner: slotmap::basic::Drain<'a, K, Ptr>,
}

impl<'a, K: Key, Ptr> Iterator for Drain<'a, K, Ptr>
where
    K: 'a,
    Ptr: StableDeref + 'a,
    Ptr::Target: Pointee,
    <Ptr::Target as Pointee>::Metadata: Copy,
{
    type Item = (CastKey<Ptr::Target, K>, Ptr);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (k, value) = self.inner.next()?;
        let patched = to_castable::<K, Ptr::Target>(k, &*value);
        Some((patched, value))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

// ─── IntoIter (owning) ───────────────────────────────────────────────────────

pub struct IntoIter<K: Key, Ptr: StableDeref> {
    inner: slotmap::basic::IntoIter<K, Ptr>,
}

impl<K: Key, Ptr> Iterator for IntoIter<K, Ptr>
where
    Ptr: StableDeref,
    Ptr::Target: Pointee,
    <Ptr::Target as Pointee>::Metadata: Copy,
{
    type Item = (CastKey<Ptr::Target, K>, Ptr);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (k, value) = self.inner.next()?;
        let patched = to_castable::<K, Ptr::Target>(k, &*value);
        Some((patched, value))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K: Key, Ptr> IntoIterator for UnsafeCastMap<K, Ptr>
where
    Ptr: StableDeref,
    Ptr::Target: Pointee,
    <Ptr::Target as Pointee>::Metadata: Copy,
{
    type Item = (CastKey<Ptr::Target, K>, Ptr);
    type IntoIter = IntoIter<K, Ptr>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.inner.into_iter(),
        }
    }
}

impl<'a, K: Key, Ptr> IntoIterator for &'a UnsafeCastMap<K, Ptr>
where
    K: 'a,
    Ptr: StableDeref + 'a,
    Ptr::Target: Pointee,
    <Ptr::Target as Pointee>::Metadata: Copy,
{
    type Item = (CastKey<Ptr::Target, K>, &'a Ptr::Target);
    type IntoIter = Iter<'a, K, Ptr>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K: Key, Ptr> IntoIterator for &'a mut UnsafeCastMap<K, Ptr>
where
    K: 'a,
    Ptr: StableDeref + DerefMut + 'a,
    Ptr::Target: Pointee,
    <Ptr::Target as Pointee>::Metadata: Copy,
{
    type Item = (CastKey<Ptr::Target, K>, &'a mut Ptr::Target);
    type IntoIter = IterMut<'a, K, Ptr>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}
