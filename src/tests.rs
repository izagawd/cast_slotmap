//! Tests for the cast slot maps. These exercise the public surface and the
//! `MapId` / version-based soundness checks. They mirror `stable_gen_map`'s
//! castable tests, adapted to `slotmap`'s `&mut self` mutation model.

use std::any::Any;

use crate::cast_key::StableCastKey;
use crate::cast_map::BoxCastMap;
use crate::unsafe_cast_map::UnsafeBoxCastMap;
use crate::DefaultKey;

#[derive(Debug, PartialEq)]
struct Dog {
    name: String,
}

#[derive(Debug, PartialEq)]
struct Cat {
    lives: u32,
}

type AnyMap = BoxCastMap<DefaultKey, dyn Any>;

// ─── insert_sized + typed get ────────────────────────────────────────────────

#[test]
fn insert_sized_then_typed_get() {
    let mut map: AnyMap = AnyMap::new();
    let key: StableCastKey<Dog> = map.insert_sized(Box::new(Dog { name: "Rex".into() }));
    assert_eq!(map.get(key).unwrap().name, "Rex");
    assert_eq!(map.len(), 1);
    assert!(!map.is_empty());
}

#[test]
fn typed_get_through_erased_key() {
    let mut map: AnyMap = AnyMap::new();
    let dog_key: StableCastKey<Dog> = map.insert_sized(Box::new(Dog { name: "Fido".into() }));
    let dyn_key: StableCastKey<dyn Any> = dog_key.upcast::<dyn Any>();

    let as_any: &dyn Any = map.get(dyn_key).unwrap();
    assert_eq!(as_any.downcast_ref::<Dog>().unwrap().name, "Fido");
}

// ─── upcast + downcast_key ───────────────────────────────────────────────────

#[test]
fn downcast_key_right_and_wrong_type() {
    let mut map: AnyMap = AnyMap::new();
    let dog_key: StableCastKey<Dog> = map.insert_sized(Box::new(Dog { name: "Spot".into() }));
    let dyn_key: StableCastKey<dyn Any> = dog_key.upcast::<dyn Any>();

    let recovered: StableCastKey<Dog> = map.downcast_key::<Dog>(dyn_key).unwrap();
    assert_eq!(map.get(recovered).unwrap().name, "Spot");

    assert!(map.downcast_key::<Cat>(dyn_key).is_none());
}

// ─── insert_as ───────────────────────────────────────────────────────────────

#[test]
fn insert_as_keeps_source_type() {
    let mut map: AnyMap = AnyMap::new();
    let key: StableCastKey<Cat> = map.insert_as(Box::new(Cat { lives: 9 }) as Box<Cat>);
    assert_eq!(map.get(key).unwrap().lives, 9);
}

// ─── get_mut ─────────────────────────────────────────────────────────────────

#[test]
fn get_mut_mutates() {
    let mut map: AnyMap = AnyMap::new();
    let key: StableCastKey<Cat> = map.insert_sized(Box::new(Cat { lives: 9 }));
    map.get_mut(key).unwrap().lives -= 1;
    assert_eq!(map.get(key).unwrap().lives, 8);
}

// ─── remove returns a re-typed Box ───────────────────────────────────────────

#[test]
fn remove_returns_boxed_concrete() {
    let mut map: AnyMap = AnyMap::new();
    let key: StableCastKey<Dog> = map.insert_sized(Box::new(Dog { name: "Bud".into() }));

    let removed: Box<Dog> = map.remove(key).unwrap();
    assert_eq!(removed.name, "Bud");

    // The key is now stale (slotmap bumps the slot version on remove).
    assert!(map.get(key).is_none());
    assert!(map.is_empty());
}

// ─── version invalidation ────────────────────────────────────────────────────

#[test]
fn stale_key_after_remove_reinsert() {
    let mut map: AnyMap = AnyMap::new();
    let k1: StableCastKey<Cat> = map.insert_sized(Box::new(Cat { lives: 1 }));
    let _ = map.remove(k1).unwrap();
    let k2: StableCastKey<Cat> = map.insert_sized(Box::new(Cat { lives: 2 }));

    // Old key does not alias the new value.
    assert!(map.get(k1).is_none());
    assert_eq!(map.get(k2).unwrap().lives, 2);
}

// ─── clear invalidates ───────────────────────────────────────────────────────

#[test]
fn clear_invalidates_keys() {
    let mut map: AnyMap = AnyMap::new();
    let key: StableCastKey<Dog> = map.insert_sized(Box::new(Dog { name: "Rex".into() }));
    map.clear();
    assert!(map.get(key).is_none());
    assert!(map.is_empty());
}

// ─── MapId: cross-map misuse returns None ────────────────────────────────────

#[test]
fn cross_map_key_is_rejected() {
    let mut a: AnyMap = AnyMap::new();
    let mut b: AnyMap = AnyMap::new();
    assert_ne!(a.map_id(), b.map_id());

    let ka: StableCastKey<Dog> = a.insert_sized(Box::new(Dog { name: "A".into() }));

    assert!(b.get(ka).is_none());
    assert!(!b.contains_key(ka));
    assert!(b.remove(ka).is_none());

    let dyn_ka: StableCastKey<dyn Any> = ka.upcast::<dyn Any>();
    assert!(b.downcast_key::<Dog>(dyn_ka).is_none());

    // Original still works.
    assert_eq!(a.get(ka).unwrap().name, "A");
}

// ─── contains_key ────────────────────────────────────────────────────────────

#[test]
fn contains_key_tracks_liveness() {
    let mut map: AnyMap = AnyMap::new();
    let key: StableCastKey<Cat> = map.insert_sized(Box::new(Cat { lives: 9 }));
    assert!(map.contains_key(key));
    let _ = map.remove(key);
    assert!(!map.contains_key(key));
}

// ─── iter / values / keys ────────────────────────────────────────────────────

#[test]
fn iter_and_values_and_keys() {
    let mut map: AnyMap = AnyMap::new();
    map.insert_sized(Box::new(Dog { name: "a".into() }));
    map.insert_sized(Box::new(Cat { lives: 1 }));
    map.insert_sized(Box::new(Cat { lives: 2 }));

    assert_eq!(map.iter().count(), 3);
    assert_eq!(map.values().count(), 3);
    assert_eq!(map.keys().count(), 3);

    let cat_lives_sum: u32 = map
        .values()
        .filter_map(|v| v.downcast_ref::<Cat>())
        .map(|c| c.lives)
        .sum();
    assert_eq!(cat_lives_sum, 3);
}

// ─── iter_mut / values_mut ───────────────────────────────────────────────────

#[test]
fn iter_mut_mutates_all() {
    let mut map: AnyMap = AnyMap::new();
    map.insert_sized(Box::new(Cat { lives: 1 }));
    map.insert_sized(Box::new(Cat { lives: 1 }));

    for (_k, v) in map.iter_mut() {
        if let Some(c) = v.downcast_mut::<Cat>() {
            c.lives += 10;
        }
    }

    let total: u32 = map
        .values()
        .filter_map(|v| v.downcast_ref::<Cat>())
        .map(|c| c.lives)
        .sum();
    assert_eq!(total, 22);
}

// ─── retain ──────────────────────────────────────────────────────────────────

#[test]
fn retain_keeps_matching() {
    let mut map: AnyMap = AnyMap::new();
    map.insert_sized(Box::new(Cat { lives: 1 }));
    map.insert_sized(Box::new(Cat { lives: 9 }));
    map.insert_sized(Box::new(Dog { name: "z".into() }));

    map.retain(|_k, v| v.downcast_ref::<Cat>().map_or(false, |c| c.lives > 5));
    assert_eq!(map.len(), 1);
    assert!(map.values().next().unwrap().is::<Cat>());
}

// ─── drain ───────────────────────────────────────────────────────────────────

#[test]
fn drain_empties_and_yields() {
    let mut map: AnyMap = AnyMap::new();
    map.insert_sized(Box::new(Cat { lives: 3 }));
    map.insert_sized(Box::new(Cat { lives: 4 }));

    let drained: Vec<Box<dyn Any>> = map.drain().map(|(_k, v)| v).collect();
    assert_eq!(drained.len(), 2);
    assert!(map.is_empty());
}

// ─── Index / IndexMut (via the erased key) ───────────────────────────────────

#[test]
fn index_via_dyn_key() {
    let mut map: AnyMap = AnyMap::new();
    let dog_key: StableCastKey<Dog> = map.insert_sized(Box::new(Dog { name: "Idx".into() }));
    let dyn_key: StableCastKey<dyn Any> = dog_key.upcast::<dyn Any>();

    let as_any: &dyn Any = &map[dyn_key];
    assert_eq!(as_any.downcast_ref::<Dog>().unwrap().name, "Idx");
}

// ─── capacity / with_capacity ────────────────────────────────────────────────

#[test]
fn with_capacity_reserves() {
    let map: AnyMap = AnyMap::with_capacity(16);
    assert!(map.capacity() >= 16);
    assert!(map.is_empty());
}

// ─── Clone gives a fresh identity ────────────────────────────────────────────

#[test]
fn clone_gets_fresh_map_id() {
    // `Box<u32>` is `Clone`, so this map is `Clone` (a `Box<dyn Any>` map is not,
    // exactly as `dyn Any` is not `Clone`).
    let mut map: BoxCastMap<DefaultKey, u32> = BoxCastMap::new();
    let key: StableCastKey<u32> = map.insert(Box::new(7u32));
    let id1 = map.map_id();

    let clone = map.clone();
    assert_ne!(id1, clone.map_id());

    // The original key is valid on the original, rejected by the clone.
    assert_eq!(*map.get(key).unwrap(), 7);
    assert!(clone.get(key).is_none());

    // The clone still holds the data, reachable under fresh keys.
    let values: Vec<&u32> = clone.values().collect();
    assert_eq!(values.len(), 1);
    assert_eq!(*values[0], 7);
}

// ─── insert_with_key sees its own key ────────────────────────────────────────

#[test]
fn insert_sized_with_key_threads_key() {
    let mut map: AnyMap = AnyMap::new();
    // `insert_sized` erases a concrete `u32` into the `dyn Any` map; the closure
    // receives the final, typed key.
    let key: StableCastKey<u32> = map.insert_sized_with_key(|_k| Box::new(123u32));
    assert_eq!(*map.get(key).unwrap(), 123);
}

// ─── UnsafeCastMap direct usage ──────────────────────────────────────────

#[test]
fn unsafe_map_typed_roundtrip() {
    let mut map: UnsafeBoxCastMap<DefaultKey, dyn Any> = UnsafeBoxCastMap::new();
    let key = map.insert_sized(Box::new(Dog { name: "U".into() }));

    // SAFETY: `key` was just minted by this map and still addresses the value.
    let d: &Dog = unsafe { map.get(key).unwrap() };
    assert_eq!(d.name, "U");

    // SAFETY: same key, still valid.
    let removed: Box<Dog> = unsafe { map.remove(key).unwrap() };
    assert_eq!(removed.name, "U");
    assert!(map.is_empty());
}

#[test]
fn unsafe_map_detach_reattach() {
    let mut map: UnsafeBoxCastMap<DefaultKey, dyn Any> = UnsafeBoxCastMap::new();
    let key = map.insert_sized(Box::new(Dog { name: "Rex".into() }));

    // SAFETY: `key` was just minted by this map and still addresses the value.
    let mut dog: Box<Dog> = unsafe { map.detach(key).unwrap() };
    assert_eq!(dog.name, "Rex");
    // SAFETY: the slot is detached, so the lookup misses without using metadata.
    assert!(unsafe { map.get(key) }.is_none());
    assert!(map.is_empty());

    // Reattach the (mutated) value. `reattach` takes the map's erased-target
    // key, so upcast the typed `CastKey<Dog>` to `CastKey<dyn Any>`; the
    // `Box<Dog>` value unsizes to `Box<dyn Any>` at the call site.
    dog.name = "Max".into();
    map.reattach(key.upcast::<dyn Any>(), dog);
    // SAFETY: a `Dog` is back in the slot, so `key`'s metadata is still correct.
    assert_eq!(unsafe { map.get(key) }.unwrap().name, "Max");
    assert_eq!(map.len(), 1);

    // Backing-key detach/reattach with an already-erased pointer.
    let ik = key.inner_key();
    let erased: Box<dyn Any> = map.detach_by_inner_key(ik).unwrap();
    assert_eq!(erased.downcast_ref::<Dog>().unwrap().name, "Max");
    map.reattach_by_inner_key(ik, Box::new(Dog { name: "Zed".into() }) as Box<dyn Any>);
    // SAFETY: still a `Dog`, so `key`'s metadata remains valid.
    assert_eq!(unsafe { map.get(key) }.unwrap().name, "Zed");
}

// ─── try_insert_with_key error path leaves the map untouched ─────────────────

#[test]
fn try_insert_error_is_noop() {
    let mut map: BoxCastMap<DefaultKey, u32> = BoxCastMap::new();
    let res: Result<StableCastKey<u32>, &str> =
        map.try_insert_with_key(|_k| Err("nope"));
    assert_eq!(res.err(), Some("nope"));
    assert!(map.is_empty());
}

#[test]
fn get_disjoint_mut_basic() {
    let mut map: BoxCastMap<DefaultKey, u32> = BoxCastMap::new();
    let k1: StableCastKey<u32> = map.insert(Box::new(10u32));
    let k2: StableCastKey<u32> = map.insert(Box::new(20u32));

    let [a, b] = map.get_disjoint_mut([k1, k2]).unwrap();
    *a += 1;
    *b += 2;
    assert_eq!(*map.get(k1).unwrap(), 11);
    assert_eq!(*map.get(k2).unwrap(), 22);

    // aliasing keys are rejected
    assert!(map.get_disjoint_mut([k1, k1]).is_none());

    // disjoint access by backing key, too
    let [c] = map.get_disjoint_mut_by_inner_key([k2.inner_key()]).unwrap();
    *c += 100;
    assert_eq!(*map.get(k2).unwrap(), 122);
}

// ─────────────────────────────────────────────────────────────────────────────
// Dense variants: identical behaviour, backed by DenseSlotMap.
// ─────────────────────────────────────────────────────────────────────────────

mod dense {
    use std::any::Any;

    use super::{Cat, Dog};
    use crate::cast_map::BoxDenseCastMap;
    use crate::unsafe_cast_map::UnsafeBoxDenseCastMap;
    use crate::cast_key::StableCastKey;
    use crate::DefaultKey;

    type AnyMap = BoxDenseCastMap<DefaultKey, dyn Any>;

    #[test]
    fn insert_sized_then_typed_get() {
        let mut map: AnyMap = AnyMap::new();
        let key: StableCastKey<Dog> = map.insert_sized(Box::new(Dog { name: "Rex".into() }));
        assert_eq!(map.get(key).unwrap().name, "Rex");
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn erased_get_and_downcast_key() {
        let mut map: AnyMap = AnyMap::new();
        let dog_key: StableCastKey<Dog> = map.insert_sized(Box::new(Dog { name: "Fido".into() }));
        let dyn_key: StableCastKey<dyn Any> = dog_key.upcast::<dyn Any>();

        let as_any: &dyn Any = map.get(dyn_key).unwrap();
        assert_eq!(as_any.downcast_ref::<Dog>().unwrap().name, "Fido");

        let recovered: StableCastKey<Dog> = map.downcast_key::<Dog>(dyn_key).unwrap();
        assert_eq!(map.get(recovered).unwrap().name, "Fido");
        assert!(map.downcast_key::<Cat>(dyn_key).is_none());
    }

    #[test]
    fn get_mut_then_remove_invalidates() {
        let mut map: AnyMap = AnyMap::new();
        let key: StableCastKey<Cat> = map.insert_sized(Box::new(Cat { lives: 9 }));
        map.get_mut(key).unwrap().lives -= 1;
        assert_eq!(map.get(key).unwrap().lives, 8);

        let removed: Box<Cat> = map.remove(key).unwrap();
        assert_eq!(removed.lives, 8);
        assert!(map.get(key).is_none()); // stale after remove
        assert!(map.is_empty());
    }

    #[test]
    fn cross_map_key_is_rejected() {
        let mut a: AnyMap = AnyMap::new();
        let mut b: AnyMap = AnyMap::new();
        assert_ne!(a.map_id(), b.map_id());

        let ka: StableCastKey<Dog> = a.insert_sized(Box::new(Dog { name: "A".into() }));
        assert!(b.get(ka).is_none());
        assert!(!b.contains_key(ka));
        assert_eq!(a.get(ka).unwrap().name, "A");
    }

    #[test]
    fn iter_values_keys_retain() {
        let mut map: AnyMap = AnyMap::new();
        map.insert_sized(Box::new(Dog { name: "a".into() }));
        map.insert_sized(Box::new(Cat { lives: 1 }));
        map.insert_sized(Box::new(Cat { lives: 9 }));

        assert_eq!(map.iter().count(), 3);
        assert_eq!(map.values().count(), 3);
        assert_eq!(map.keys().count(), 3);

        map.retain(|_k, v| v.downcast_ref::<Cat>().map_or(false, |c| c.lives > 5));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn clone_gets_fresh_map_id() {
        let mut map: BoxDenseCastMap<DefaultKey, u32> = BoxDenseCastMap::new();
        let key: StableCastKey<u32> = map.insert(Box::new(7u32));
        let id1 = map.map_id();

        let clone = map.clone();
        assert_ne!(id1, clone.map_id());
        assert_eq!(*map.get(key).unwrap(), 7);
        assert!(clone.get(key).is_none());

        let values: Vec<&u32> = clone.values().collect();
        assert_eq!(values.len(), 1);
        assert_eq!(*values[0], 7);
    }

    #[test]
    fn unsafe_dense_map_roundtrip() {
        let mut map: UnsafeBoxDenseCastMap<DefaultKey, dyn Any> = UnsafeBoxDenseCastMap::new();
        let key = map.insert_sized(Box::new(Dog { name: "U".into() }));

        // SAFETY: key was just minted by this map and still addresses the value.
        let d: &Dog = unsafe { map.get(key).unwrap() };
        assert_eq!(d.name, "U");

        // SAFETY: same key, still valid.
        let removed: Box<Dog> = unsafe { map.remove(key).unwrap() };
        assert_eq!(removed.name, "U");
        assert!(map.is_empty());
    }

    #[test]
    fn get_disjoint_mut_typed() {
        let mut map: BoxDenseCastMap<DefaultKey, u32> = BoxDenseCastMap::new();
        let k1: StableCastKey<u32> = map.insert(Box::new(1u32));
        let k2: StableCastKey<u32> = map.insert(Box::new(2u32));
        let k3: StableCastKey<u32> = map.insert(Box::new(3u32));

        let [a, b, c] = map.get_disjoint_mut([k1, k2, k3]).unwrap();
        *a += 10;
        *b += 20;
        *c += 30;
        assert_eq!(*map.get(k1).unwrap(), 11);
        assert_eq!(*map.get(k2).unwrap(), 22);
        assert_eq!(*map.get(k3).unwrap(), 33);

        // aliasing keys are rejected
        assert!(map.get_disjoint_mut([k1, k1]).is_none());
    }
}
