//! [`DynKey`]: a [`CastKey`](crate::cast_key::CastKey) borrowed into a shape
//! that can be a **method receiver on trait objects**.
//!
//! `CastKey<dyn Trait>` itself is `Sized` (the vtable is an ordinary field),
//! so `&CastKey<dyn Trait>` is a thin pointer and cannot dynamically dispatch.
//! `DynKey` re-expresses the same information as a single fat `NonNull<T>`:
//! the *metadata* half carries the key's pointer metadata (the vtable for
//! `dyn` targets), and the *address* half smuggles the backing `slotmap` key.
//! That makes `DynKey` layout-compatible with dyn dispatch, so traits can
//! declare methods as `fn m(self: DynKey<Self>, ...)` and be called through
//! `DynKey<dyn Trait>`.
//!
//! The address is either the key's packed [`KeyData`] (64-bit targets: it
//! always fits, see below) or, when it does not fit (32-bit targets), a real
//! pointer to the borrowed `CastKey` — which the `'a` borrow keeps alive. The
//! smuggled address is **never dereferenced** in the packed case.
//!
//! Nonzero guarantee: `KeyData::as_ffi()` places the slot version — a
//! `NonZeroU32` in `slotmap` — in the high 32 bits, so the packed value is
//! always ≥ `1 << 32`. `NonNull` is therefore sound for every key, and
//! `Option<DynKey>` stays pointer-sized.

use std::marker::{PhantomData, Unsize};
use std::num::NonZeroUsize;
use std::ops::{CoerceUnsized, DispatchFromDyn, Receiver};
use std::ptr::{NonNull, Pointee};

use slotmap::{DefaultKey, Key, KeyData};

use crate::cast_key::CastKey;

/// Does a packed [`KeyData`] (a `u64`) fit in a pointer address?
#[inline]
const fn fits_inline() -> bool {
    size_of::<u64>() <= size_of::<usize>()
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
    /// Address = packed `KeyData` (or a pointer to the borrowed `CastKey` when
    /// packing does not fit); metadata = the `CastKey`'s pointer metadata.
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
        let thin: NonNull<()> = if const { fits_inline() } {
            let packed = key.inner_key().data().as_ffi() as usize;
            // `as_ffi` puts the NonZeroU32 version in the high 32 bits, so
            // `packed >= 1 << 32`; the unwrap can never fire and folds away.
            NonNull::without_provenance(NonZeroUsize::new(packed).unwrap())
        } else {
            // Does not fit (32-bit target): point at the borrowed key itself.
            // Valid for 'a; provenance is preserved through from/to_raw_parts.
            NonNull::from(key).cast()
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
        if const { fits_inline() } {
            let key = K::from(KeyData::from_ffi(thin.addr().get() as u64));
            // SAFETY: `key`/`metadata` round-trip the exact values of the
            // `CastKey` given to `new`, whose construction already vouched
            // for the metadata.
            unsafe { CastKey::from_raw_parts(key, metadata) }
        } else {
            // SAFETY: on this path `thin` points at the `CastKey` borrowed by
            // `new`, still alive for 'a; `CastKey` is `Copy`.
            unsafe { thin.cast::<CastKey<T, K>>().read() }
        }
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
