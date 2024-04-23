use kivi::btree::{BTree, BTreeStorage, Key, Value};

use rand::seq::SliceRandom;
use rand::{RngCore, SeedableRng};
use rand_pcg::Pcg32;
use std::collections::HashMap;

fn insert_kv(tree: &mut BTree, reference: &mut HashMap<u64, u64>, rng: &mut Pcg32) {
    let key = rng.next_u64();
    let value = rng.next_u64();

    reference.insert(key, value);

    let keybuf = key.to_le_bytes();
    let valuebuf = value.to_le_bytes();

    let key = Key::new(&keybuf).unwrap();
    let value = Value::new(&valuebuf).unwrap();

    tree.insert(key, value);
}

fn delete_kv(tree: &mut BTree, reference: &mut HashMap<u64, u64>, rng: &mut Pcg32) {
    let key = **reference.keys().collect::<Vec<&u64>>().choose(rng).unwrap();
    reference.remove(&key);

    let keybuf = key.to_le_bytes();
    let key = Key::new(&keybuf).unwrap();
    tree.delete(key);
}

fn check_consistency(tree: &BTree, reference: &HashMap<u64, u64>) {
    for (key, ref_val) in reference.iter() {
        let keybuf = key.to_le_bytes();
        let key = Key::new(&keybuf).unwrap();
        let value = tree.get(key).unwrap();
        let value = u64::from_le_bytes(value.try_into().unwrap());

        assert_eq!(value, *ref_val);
    }
}

fn main() {
    let mut tree = BTree::new(BTreeStorage::in_memory());
    let mut reference: HashMap<u64, u64> = HashMap::new();
    let mut rng = Pcg32::seed_from_u64(42);

    for _ in 0..5000 {
        insert_kv(&mut tree, &mut reference, &mut rng);
    }

    check_consistency(&tree, &reference);

    for _ in 0..5000 {
        insert_kv(&mut tree, &mut reference, &mut rng);
        delete_kv(&mut tree, &mut reference, &mut rng);
        check_consistency(&tree, &reference);
    }
}
