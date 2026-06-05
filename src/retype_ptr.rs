//! Reconstruct a re-typed smart pointer from the erased pointee plus a target's
//! pointer metadata. Used by `remove::<T>` to hand back e.g. a `Box<Dog>` from a
//! map storing `Box<dyn Any>`.

use std::ptr::Pointee;
use std::rc::Rc;
use std::sync::Arc;

/// Reconstructs a re-typed version of a stored smart pointer from the erased
/// pointee plus a target's pointer metadata. `Box`/`Rc`/`Arc`/`&T`/`&mut T` are
/// provided; custom pointers implement it themselves.
///
/// # Safety
/// `retype` must reconstruct the *same* allocation/ownership as `self`, only
/// changing the static pointee type.
pub unsafe trait RetypePtr<'a> {
    /// The same pointer kind re-pointed at `U` (`Box<O>` -> `Box<U>`, ...).
    type Retyped<U: ?Sized + 'a>;

    /// # Safety
    /// The value behind `self` must actually be a `U`, and `metadata` must be
    /// its correct pointer metadata.
    unsafe fn retype<U: ?Sized>(self, metadata: <U as Pointee>::Metadata) -> Self::Retyped<U>;
}

unsafe impl<'a, O: ?Sized> RetypePtr<'a> for Box<O> {
    type Retyped<U: ?Sized + 'a> = Box<U>;
    #[inline]
    unsafe fn retype<U: ?Sized>(self, meta: <U as Pointee>::Metadata) -> Box<U> {
        let data: *mut () = Box::into_raw(self).cast();
        Box::from_raw(std::ptr::from_raw_parts_mut(data, meta))
    }
}

unsafe impl<'a, O: ?Sized> RetypePtr<'a> for Rc<O> {
    type Retyped<U: ?Sized + 'a> = Rc<U>;
    #[inline]
    unsafe fn retype<U: ?Sized>(self, meta: <U as Pointee>::Metadata) -> Rc<U> {
        let data: *const () = Rc::into_raw(self).cast();
        Rc::from_raw(std::ptr::from_raw_parts(data, meta))
    }
}

unsafe impl<'a, O: ?Sized> RetypePtr<'a> for Arc<O> {
    type Retyped<U: ?Sized + 'a> = Arc<U>;
    #[inline]
    unsafe fn retype<U: ?Sized>(self, meta: <U as Pointee>::Metadata) -> Arc<U> {
        let data: *const () = Arc::into_raw(self).cast();
        Arc::from_raw(std::ptr::from_raw_parts(data, meta))
    }
}

unsafe impl<'a, O: ?Sized> RetypePtr<'a> for &'a O {
    type Retyped<U: ?Sized + 'a> = &'a U;
    #[inline]
    unsafe fn retype<U: ?Sized>(self, meta: <U as Pointee>::Metadata) -> &'a U {
        let data: *const () = (self as *const O).cast();
        &*std::ptr::from_raw_parts(data, meta)
    }
}

unsafe impl<'a, O: ?Sized> RetypePtr<'a> for &'a mut O {
    type Retyped<U: ?Sized + 'a> = &'a mut U;
    #[inline]
    unsafe fn retype<U: ?Sized>(self, meta: <U as Pointee>::Metadata) -> &'a mut U {
        let data: *mut () = (self as *mut O).cast();
        &mut *std::ptr::from_raw_parts_mut(data, meta)
    }
}
