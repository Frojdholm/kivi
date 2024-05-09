use kivi::btree::BTree;

use rand::seq::SliceRandom;
use rand::{RngCore, SeedableRng};
use rand_pcg::Pcg32;
use std::collections::HashMap;

fn insert_kv(tree: &mut BTree, reference: &mut HashMap<u64, u64>, rng: &mut Pcg32) {
    let key = rng.next_u64();
    let value = rng.next_u64();

    reference.insert(key, value);

    tree.insert(&key.to_le_bytes(), &value.to_le_bytes()).unwrap();
}

fn delete_kv(tree: &mut BTree, reference: &mut HashMap<u64, u64>, rng: &mut Pcg32) {
    let key = **reference.keys().collect::<Vec<&u64>>().choose(rng).unwrap();
    reference.remove(&key);
    tree.delete(&key.to_le_bytes()).unwrap();
}

fn check_consistency(tree: &BTree, reference: &HashMap<u64, u64>) {
    for (key, ref_val) in reference.iter() {
        let value = tree.get(&key.to_le_bytes()).ok().flatten().unwrap();
        let value = u64::from_le_bytes(value.try_into().unwrap());

        assert_eq!(value, *ref_val);
    }
}

fn main() {
    let mut tree = BTree::in_memory();
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
