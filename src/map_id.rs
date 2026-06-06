//! Per-map identity used by [`CastMap`](crate::cast_map::CastMap)
//! to bind keys to the map that created them.

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global counter for map identifiers. Starts at 1 so that 0 is never a valid
/// map id.
static NEXT_MAP_ID: AtomicUsize = AtomicUsize::new(1);

/// A unique map identifier, stored inside each
/// [`StableCastKey`](crate::cast_key::StableCastKey) and checked on every keyed
/// access so that a key from one map cannot be used on another.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct MapId(pub(crate) NonZeroUsize);

impl MapId {
    /// Construct a `MapId` from a raw `NonZeroUsize`.
    ///
    /// # Safety
    /// The caller must ensure the value is a valid, previously issued map id.
    #[inline]
    pub unsafe fn from_non_zero_usize(number: NonZeroUsize) -> MapId {
        MapId(number)
    }

    /// Returns the underlying `NonZeroUsize` value.
    #[inline]
    pub fn get_underlying_non_zero_usize(&self) -> NonZeroUsize {
        self.0
    }

    /// Requests a fresh, globally unique map id.
    ///
    /// Ids start at 1; 0 is reserved as the invalid/vacant sentinel. Panics on
    /// the (astronomically unlikely) event of a `usize` overflow.
    #[inline]
    pub fn next() -> Self {
        // `try_update` returns the *previous* value on success. The counter
        // starts at 1 and only ever increases, and `checked_add` rules out a
        // wrap to 0, so the returned value is always non-zero.
        let previous = NEXT_MAP_ID
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |raw| raw.checked_add(1))
            .expect("MapId counter overflow");

        MapId(NonZeroUsize::new(previous).expect("MapId counter overflow"))
    }
}
