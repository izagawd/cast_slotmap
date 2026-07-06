//! [`TypeTaggedPtr`]: a smart pointer paired with the concrete [`TypeId`] of
//! its pointee, plus the [`ConcreteTypeId`] extension trait the checked maps
//! validate against. [`TypeTaggedBox`] is its owning-[`Box`] alias.
//!
//! [`CastMapG`](crate::cast_map::CastMapG)'s safety comes from comparing a
//! key's metadata-implied type id against the type id stored next to the
//! value. Something has to *store* that type id: [`TypeTaggedPtr`] does, for any
//! wrapped pointer (`Box`, `Rc`, `Arc`, `&T`, `&mut T`, ...). Custom stored
//! pointer types can participate by implementing [`ConcreteTypeId`].

use std::any::TypeId;
use std::ops::{CoerceUnsized, Deref, DerefMut};
use std::ptr::Pointee;

use stable_deref_trait::StableDeref;

use crate::retype_ptr::RetypePtr;

// ─── TypeTaggedPtr ───────────────────────────────────────────────────────────

/// A smart pointer `P` (`Box<T>`, `Rc<T>`, `Arc<T>`, `&T`, `&mut T`, ...)
/// paired with the concrete [`TypeId`] of the value it points at, kept even
/// after the pointer is coerced to a trait object.
///
/// The type id is captured at construction ([`from_ptr`](Self::from_ptr) /
/// `TypeTaggedBox::new`) and preserved across unsizing coercions
/// (`TypeTaggedPtr<Box<Dog>> -> TypeTaggedPtr<Box<dyn Animal>>`), since
/// unsizing only touches the inner pointer.
///
/// # Invariant
/// `type_id` is always the [`TypeId`] of the concrete type of `ptr`'s
/// pointee. [`CastMapG`](crate::cast_map::CastMapG)'s checked lookups
/// rebuild typed references from it (see [`ConcreteTypeId`]); the `unsafe` escape hatches
/// ([`from_raw_parts`](Self::from_raw_parts), [`inner_mut`](Self::inner_mut))
/// make the caller responsible for keeping it true.
pub struct TypeTaggedPtr<P> {
    ptr: P,
    type_id: TypeId,
}

/// [`TypeTaggedPtr`] wrapping a [`Box`]: an owning, pointer-stable form
/// that remembers the concrete [`TypeId`] of the value it was constructed
/// from.
pub type TypeTaggedBox<T: ?Sized> = TypeTaggedPtr<Box<T>>;

impl<T: 'static> TypeTaggedPtr<Box<T>> {
    /// Boxes `value`, recording `TypeId::of::<T>()`.
    #[inline]
    pub fn new(value: T) -> Self {
        Self::from_ptr(Box::new(value))
    }
}

impl<P> TypeTaggedPtr<P> {
    /// Wraps `ptr`, recording `TypeId::of::<P::Target>()`. For an
    /// already-erased pointer whose concrete type id you know, use
    /// [`from_raw_parts`](Self::from_raw_parts).
    #[inline]
    pub fn from_ptr(ptr: P) -> Self
    where
        P: Deref,
        P::Target: Sized + 'static,
    {
        Self {
            type_id: TypeId::of::<P::Target>(),
            ptr,
        }
    }

    /// Assembles a `TypeTaggedPtr` from a pointer and an already-known type id.
    ///
    /// # Safety
    /// `type_id` must be the [`TypeId`] of the concrete type of `ptr`'s
    /// pointee (the struct invariant). A wrong type id lets
    /// [`CastMapG`](crate::cast_map::CastMapG)'s checked lookups reinterpret
    /// the value as another type, which is undefined behavior.
    #[inline]
    pub unsafe fn from_raw_parts(ptr: P, type_id: TypeId) -> Self {
        Self { ptr, type_id }
    }

    /// Shared access to the wrapped pointer itself (for the pointee, use
    /// [`Deref`]).
    #[inline]
    pub fn inner_ref(&self) -> &P {
        &self.ptr
    }

    /// Exclusive access to the wrapped pointer itself.
    ///
    /// # Safety
    /// `&mut P` can replace or re-point the pointer — e.g. swap in a
    /// different `Box<dyn Any>` — while the recorded type id stays put. When the
    /// borrow ends, `type_id` must still be the concrete type id of the (possibly
    /// new) pointee, per the struct invariant.
    #[inline]
    pub unsafe fn inner_mut(&mut self) -> &mut P {
        &mut self.ptr
    }

    /// Unwraps the pointer, discarding the recorded type id.
    #[inline]
    pub fn inner(self) -> P {
        self.ptr
    }
}

impl<P: Deref> Deref for TypeTaggedPtr<P> {
    type Target = P::Target;
    #[inline]
    fn deref(&self) -> &P::Target {
        &self.ptr
    }
}

impl<P: DerefMut> DerefMut for TypeTaggedPtr<P> {
    #[inline]
    fn deref_mut(&mut self) -> &mut P::Target {
        &mut self.ptr
    }
}

impl<P: CoerceUnsized<Q>, Q> CoerceUnsized<TypeTaggedPtr<Q>> for TypeTaggedPtr<P> {}


unsafe impl<P: StableDeref> StableDeref for TypeTaggedPtr<P> {}


unsafe impl<'a, P: RetypePtr<'a>> RetypePtr<'a> for TypeTaggedPtr<P> {
    type Retyped<U: ?Sized + 'a> = TypeTaggedPtr<P::Retyped<U>>;
    #[inline]
    unsafe fn retype<U: ?Sized>(self, meta: <U as Pointee>::Metadata) -> Self::Retyped<U> {
        TypeTaggedPtr {
            ptr: self.ptr.retype(meta),
            type_id: self.type_id,
        }
    }
}

// ─── ConcreteTypeId ──────────────────────────────────────────────────────────

/// A stored pointer that knows the concrete [`TypeId`] of its pointee.
///
/// This is the extension point for [`CastMapG`](crate::cast_map::CastMapG):
/// its checked lookups read it to validate a key's type. The crate implements
/// it for [`TypeTaggedPtr`] (and thus [`TypeTaggedBox`]), but it is deliberately a
/// public, standalone trait — to use your own stored pointer type with
/// [`CastMapG`](crate::cast_map::CastMapG), implement `ConcreteTypeId` for it
/// (alongside `Deref` + [`StableDeref`], which any stored pointer needs).
/// Nothing here assumes `TypeTaggedPtr` specifically.
///
/// Why store the type id instead of asking the value? Not every stored
/// pointer answers correctly: `Box<dyn Any>` could, but for a `Box<dyn Foo>`
/// where `Foo` is not an `Any` subtrait, `type_id` resolves statically to
/// `TypeId::of::<dyn Foo>()` — not the underlying type's — and
/// special-casing the stored pointers that answer correctly would make
/// [`CastMapG`](crate::cast_map::CastMapG)'s behavior depend confusingly on
/// the stored pointer's type. An
/// explicitly stored type id works uniformly — and is also a performance win: a
/// plain field read per lookup instead of a virtual call to ask the value.
///
/// # Safety
/// [`CastMapG`](crate::cast_map::CastMapG)'s checked lookups (`get`,
/// `get_mut`, `remove`, `get_disjoint_mut`, `downcast_key`) rebuild typed
/// references based on this value: `concrete_type_id` must return the
/// [`TypeId`] of the concrete type of its current pointee. A
/// wrong answer lets one of those safe lookups reinterpret the value as
/// another type, which is undefined behavior.
pub unsafe trait ConcreteTypeId {
    /// The concrete type id of this pointer's pointee.
    fn concrete_type_id(&self) -> TypeId;
}

// SAFETY: `type_id` is captured from the concrete, sized `P::Target` in
// `from_ptr` (or vouched for by the caller of `from_raw_parts`) and only ever
// carried across unsizing coercions / `retype` — whose contract requires the
// value to actually be the target type — while `inner_mut`'s contract makes
// any caller who swaps the pointer keep the type id accurate.
unsafe impl<P> ConcreteTypeId for TypeTaggedPtr<P> {
    #[inline]
    fn concrete_type_id(&self) -> TypeId {
        self.type_id
    }
}
