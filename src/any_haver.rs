//! Recovering a concrete [`TypeId`] from pointer metadata, with no live value.
//!
//! [`AnyHaver`] is an **`unsafe` trait**, blanket-implemented for every
//! `'static` **sized** type. Unsized types get it only through a supertrait
//! bound (`trait Foo: AnyHaver`), which places `haver_type_id` in `dyn Foo`'s
//! vtable so the call dispatches virtually to the concrete type's impl.
//! Consequently `dyn Any` (no such supertrait) simply does not implement
//! `AnyHaver` — asking the checked map for `&dyn Any` is a compile error
//! rather than a silent miss; use `downcast_key` / `get_by_inner_key` for
//! erased access.
//!
//! The single method takes a **raw** `*const Self` rather than `&self`, so it
//! can be invoked on a dangling/null data pointer: only the metadata (vtable)
//! is consulted. That is what lets [`type_id_from_meta`] turn a
//! [`CastKey`](crate::cast_key::CastKey)'s stored metadata into a [`TypeId`]
//! without ever dereferencing anything.
//!
//! Dispatch summary for `type_id_from_meta::<T>(meta)`:
//! - `T` sized          → `TypeId::of::<T>()`              (static, metadata is `()`)
//! - `T = dyn AnyHaver` → the *concrete* type's `TypeId`   (virtual, via the vtable)
//! - `T = dyn Foo` where `Foo: AnyHaver` → the concrete type's `TypeId` through
//!   `dyn Foo`'s vtable (supertrait methods live in the vtable).

use std::any::TypeId;
use std::ptr::Pointee;

/// Exposes the concrete [`TypeId`] through a raw-`self` method so it works on
/// metadata alone. Blanket-implemented for all `'static` **sized** types;
/// reach it on trait objects via a supertrait bound (`trait Foo: AnyHaver`).
///
/// # Safety
/// The checked maps rebuild typed references based on this value:
/// `haver_type_id` must return the [`TypeId`] of the true concrete `Self`.
/// A lying implementation makes those lookups unsound.
pub unsafe trait AnyHaver: 'static {
    /// Returns the [`TypeId`] of the (possibly type-erased) `Self`.
    ///
    /// Takes `*const Self` instead of `&self` so it is callable on a null /
    /// dangling data pointer — the body never reads through the pointer.
    #[inline]
    fn haver_type_id(self: *const Self) -> TypeId {
        TypeId::of::<Self>()
    }
}

// SAFETY: for a sized `T`, `TypeId::of::<Self>()` *is* the concrete type id.
unsafe impl<T: 'static> AnyHaver for T {}

/// Recovers a [`TypeId`] from a value's pointer `metadata`, without a value.
///
/// Builds a `*const T` from a null data address and the supplied metadata,
/// then asks it for its `TypeId`. 
#[inline]
pub fn type_id_from_meta<T: ?Sized + AnyHaver + Pointee>(
    metadata: <T as Pointee>::Metadata,
) -> TypeId {
    let fat: *const T = std::ptr::from_raw_parts(std::ptr::null::<()>(), metadata);
    fat.haver_type_id()
}
