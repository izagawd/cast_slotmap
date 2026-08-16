use std::any::TypeId;
use std::marker::{PhantomData, Unsize};
use std::mem::MaybeUninit;
use std::num::NonZeroUsize;
use std::ops::{CoerceUnsized, DispatchFromDyn, Receiver};
use std::ptr::{NonNull, Pointee};

use slotmap::{DefaultKey, Key, KeyData};

use crate::any_haver::AnyHaver;
use crate::cast_key::CastKey;


/// Does `K` fit in a pointer address on this target?
#[inline]
const fn packs_in_ptr<K>() -> bool {
    size_of::<K>() <= size_of::<usize>()
}

/// A borrowed, dyn-dispatchable form of a [`CastKey`].
///
/// Obtain one with [`CastKey::as_dyn`] (or `From<&CastKey>`); recover the key
/// with [`DynKey::key`] (or `Into<CastKey>`). Use it as a trait-method
/// receiver:
///
/// ```ignore
/// trait Component {
///     fn tick(self: DynKey<'_, Self>, world: &mut World);
/// }
/// let dk: DynKey<'_, dyn Component> = key.as_dyn();
/// dk.tick(&mut world); // virtual call through the key's metadata
/// ```
pub struct DynKey<'a, T: ?Sized, K: Key = DefaultKey> {
    /// Address = the key's raw bytes (or a pointer to the borrowed key's `K`
    /// field if a key's size doesn't fit a pointer address); metadata = the `CastKey`'s pointer
    /// metadata. Never dereferenced as a `T`.
    ptr: NonNull<T>,
    _borrow: PhantomData<&'a K>,
}

impl<'a, T: ?Sized, K: Key> Clone for DynKey<'a, T, K> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, T: ?Sized, K: Key> Copy for DynKey<'a, T, K> {}

// SAFETY: a `DynKey` is semantically a `&'a CastKey<T, K>` (on the borrow
// path) or a by-value copy of the key's bits (on the packed path); it is
// `Copy`, never hands out `&mut`, and never dereferences `ptr` as a `T`.
// Sending or sharing it across threads therefore only permits *reading* the
// borrowed `CastKey`, so both impls require exactly `CastKey<T, K>: Sync`.
unsafe impl<'a, T: ?Sized + Pointee, K: Key> Send for DynKey<'a, T, K>
where
    <T as Pointee>::Metadata: Copy,
    CastKey<T, K>: Sync,
{
}
unsafe impl<'a, T: ?Sized + Pointee, K: Key> Sync for DynKey<'a, T, K>
where
    <T as Pointee>::Metadata: Copy,
    CastKey<T, K>: Sync,
{
}

// Dyn-dispatch machinery: `DynKey` is a single (fat) pointer plus 1-ZSTs, the
// exact shape `DispatchFromDyn` requires of a receiver.
impl<'a, T: ?Sized + Unsize<U>, U: ?Sized, K: Key> CoerceUnsized<DynKey<'a, U, K>>
for DynKey<'a, T, K>
{
}
impl<'a, T: ?Sized + Unsize<U>, U: ?Sized, K: Key> DispatchFromDyn<DynKey<'a, U, K>>
for DynKey<'a, T, K>
{
}

// A receiver without `Deref`: the key alone cannot reach the value (that needs
// the map), so only dispatch — not `*dyn_key` — is offered.
impl<'a, T: ?Sized, K: Key> Receiver for DynKey<'a, T, K> {
    type Target = T;
}

impl<'a, T: ?Sized + Pointee, K: Key> DynKey<'a, T, K>
where
    <T as Pointee>::Metadata: Copy,
{
    /// Borrows a [`CastKey`] into its dyn-dispatchable form.
    #[inline]
    pub fn new(key: &'a CastKey<T, K>) -> Self {
        let thin: NonNull<()> = if const { packs_in_ptr::<K>() } {
            // SAFETY: it is never dereferenced, just stored as a pointer (not in a pointer)
            let addr = unsafe {
                let mut bits = MaybeUninit::<usize>::zeroed();
                bits.as_mut_ptr().cast::<K>().write_unaligned(key.key);
                NonZeroUsize::new_unchecked(bits.assume_init())
            };
            NonNull::without_provenance(addr)
        } else {
            // The key does not fit in a pointer on this target: point at the
            // borrowed key's backing `K` field. `K` is the same type for
            // every `T`, so `key()` can read it back even after an unsizing
            // coercion changes `T` (reading the whole `CastKey<T, K>` could
            // not: its layout differs per `T`). Valid for 'a; provenance is
            // preserved through from/to_raw_parts.
            NonNull::from(&key.key).cast()
        };
        Self {
            ptr: NonNull::from_raw_parts(thin, key.metadata()),
            _borrow: PhantomData,
        }
    }

    /// Recovers the [`CastKey`] this `DynKey` was made from.
    #[inline]
    pub fn key(self) -> CastKey<T, K> {
        let (thin, metadata) = self.ptr.to_raw_parts();
        if const { packs_in_ptr::<K>() } {
            // SAFETY: reads the first `size_of::<K>()` bytes of `thin`
            // itself, which is guaranteed to be K
            let key: K = unsafe { core::mem::transmute_copy(&thin) };
            CastKey::from_raw_parts(key, metadata)
        } else {
            // SAFETY: on this path, `thin` points at the `K` field of the
            // `CastKey` borrowed by `new`, which is still alive, represented by the lifetime 'a.
            // `K` is `Copy`, so reading it using unsafe is also fine
            let k = unsafe { thin.cast::<K>().read() };
            CastKey::from_raw_parts(k, metadata)
        }
    }
}

impl<'a, T: ?Sized + AnyHaver + Pointee, K: Key> DynKey<'a, T, K>
where
    <T as Pointee>::Metadata: Copy,
{
    /// Downcasts to a sized `Concrete` using only the key's metadata; no map
    /// involved. Returns `None` on a type mismatch. See [`CastKey::downcast`].
    #[inline]
    pub fn downcast<Concrete: 'static>(self) -> Option<DynKey<'a, Concrete, K>> {
        // `self.ptr` is already a fat pointer with the right metadata, and
        // `haver_type_id` never reads the data half, so no null-data pointer
        // needs building.
        let fat: *const T = self.ptr.as_ptr();
        (fat.haver_type_id() == TypeId::of::<Concrete>()).then(|| {
            let (thin, _) = self.ptr.to_raw_parts();
            DynKey {
                // The address half does not depend on `T`
                // `Concrete` is sized, so the new metadata is `()`.
                ptr: NonNull::from_raw_parts(thin, ()),
                _borrow: PhantomData,
            }
        })
    }
}

impl<'a, T: ?Sized + Pointee, K: Key> From<&'a CastKey<T, K>> for DynKey<'a, T, K>
where
    <T as Pointee>::Metadata: Copy,
{
    #[inline]
    fn from(key: &'a CastKey<T, K>) -> Self {
        Self::new(key)
    }
}

impl<'a, T: ?Sized + Pointee, K: Key> From<DynKey<'a, T, K>> for CastKey<T, K>
where
    <T as Pointee>::Metadata: Copy,
{
    #[inline]
    fn from(key: DynKey<'a, T, K>) -> Self {
        key.key()
    }
}