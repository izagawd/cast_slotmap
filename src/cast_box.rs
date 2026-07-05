//! An owning box that remembers the concrete [`TypeId`] of its value, plus the
//! [`ConcreteTypeId`] extension trait the checked maps validate against.
//!
//! [`CastMapG`](crate::cast_map::CastMapG)'s safety comes from comparing a
//! key's metadata-implied type id against the type id stored next to the
//! value. Something has to *store* that id: [`CastBox`] does. Custom owning
//! boxes can participate by implementing [`ConcreteTypeId`].

use std::any::TypeId;
use std::marker::Unsize;
use std::ops::{CoerceUnsized, Deref, DerefMut};
use std::ptr::Pointee;

use stable_deref_trait::StableDeref;

use crate::retype_ptr::RetypePtr;

// ─── CastBox ─────────────────────────────────────────────────────────────────

/// An owning, pointer-stable box that remembers the concrete [`TypeId`] of the
/// value it was constructed from, even after the value is coerced to a trait
/// object.
///
/// The type id is captured by [`CastBox::new`] and stored in the handle; it is
/// preserved across unsizing coercions (`CastBox<Dog> -> CastBox<dyn Animal>`),
/// since unsizing only touches the inner `Box`.
pub struct CastBox<T: ?Sized> {
    type_id: TypeId,
    inner: Box<T>,
}

impl<T: 'static> CastBox<T> {
    /// Boxes `value`, recording `TypeId::of::<T>()`.
    #[inline]
    pub fn new(value: T) -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            inner: Box::new(value),
        }
    }
}

impl<T: ?Sized> Deref for CastBox<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        &self.inner
    }
}

impl<T: ?Sized> DerefMut for CastBox<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}

impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<CastBox<U>> for CastBox<T> {}

// The deref target lives behind the inner `Box`, whose address is stable and
// unaffected by moving the `CastBox` handle.
unsafe impl<T: ?Sized> StableDeref for CastBox<T> {}

// Re-type the inner box's tail (used by `remove`) and carry the id across
// unchanged — same operation as the bare `Box<O>` impl.
unsafe impl<'a, O: ?Sized> RetypePtr<'a> for CastBox<O> {
    type Retyped<U: ?Sized + 'a> = CastBox<U>;
    #[inline]
    unsafe fn retype<U: ?Sized>(self, meta: <U as Pointee>::Metadata) -> CastBox<U> {
        let data: *mut () = Box::into_raw(self.inner).cast();
        CastBox {
            type_id: self.type_id,
            inner: Box::from_raw(std::ptr::from_raw_parts_mut(data, meta)),
        }
    }
}

// ─── ConcreteTypeId ──────────────────────────────────────────────────────────

/// A stored value that knows the concrete [`TypeId`] of what it owns.
///
/// This is the extension point for [`CastMapG`](crate::cast_map::CastMapG):
/// its checked lookups read it to validate a key's type. The crate implements
/// it for [`CastBox`], but it is deliberately a public, box-level trait — to
/// use your own owning box with the checked maps, implement `ConcreteTypeId`
/// for it (alongside `Deref` + [`StableDeref`], which any stored pointer
/// needs). Nothing here assumes `CastBox` specifically.
pub trait ConcreteTypeId {
    /// The concrete type id of the value this box owns.
    fn concrete_type_id(&self) -> TypeId;
}

impl<T: ?Sized> ConcreteTypeId for CastBox<T> {
    #[inline]
    fn concrete_type_id(&self) -> TypeId {
        self.type_id
    }
}
