//! Recovering a concrete [`TypeId`] from pointer metadata, with no live value.
//!
//! [`AnyHaver`] is blanket-implemented for every `'static` type (sized or not).
//! Its single method takes a **raw** `*const Self` rather than `&self`, so it
//! can be invoked on a dangling/null data pointer: only the metadata (vtable)
//! is consulted. That is what lets [`type_id_from_meta`] turn a
//! [`CastKey`](crate::cast_key::CastKey)'s stored metadata into a [`TypeId`]
//! without ever dereferencing anything.
//!
//! Dispatch summary for `type_id_from_meta::<T>(meta)`:
//! - `T` sized          → `TypeId::of::<T>()`              (static, metadata is `()`)
//! - `T = dyn AnyHaver` → the *concrete* type's `TypeId`   (virtual, via the vtable)
//! - `T = dyn Foo` where `Foo: AnyHaver` → the concrete type's `TypeId` through
//!   `dyn Foo`'s vtable, relying on the supertrait method dispatching virtually.

use std::any::TypeId;
use std::ptr::Pointee;

/// Blanket-implemented for all `'static` types; exposes the concrete [`TypeId`]
/// through a raw-`self` method so it works on metadata alone.
pub trait AnyHaver: 'static {
    /// Returns the [`TypeId`] of the (possibly type-erased) `Self`.
    ///
    /// Takes `*const Self` instead of `&self` so it is callable on a null /
    /// dangling data pointer — the body never reads through the pointer.
    #[inline]
    fn haver_type_id(self: *const Self) -> TypeId {
        TypeId::of::<Self>()
    }
}

impl<T: 'static + ?Sized> AnyHaver for T {}

/// Recovers a [`TypeId`] from a value's pointer `metadata`, without a value.
///
/// Builds a fat pointer with a null data address and the supplied metadata,
/// then asks it for its `TypeId`. No memory is read.
#[inline]
pub fn type_id_from_meta<T: ?Sized + AnyHaver + Pointee>(
    metadata: <T as Pointee>::Metadata,
) -> TypeId {
    let fat: *const T = std::ptr::from_raw_parts(std::ptr::null::<()>(), metadata);
    fat.haver_type_id()
}
