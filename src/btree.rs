use crate::storage::PageStorage;

use bnode::Node;

pub type Result<T> = std::result::Result<T, BTreeError>;

pub const MAX_KEY_SIZE: usize = 1000;
pub const MAX_VAL_SIZE: usize = 3000;

type Ptr = u64;

/// Represents all the possible errors when dealing with B-Trees.
#[derive(Debug)]
pub enum BTreeError {
    EmptyKey,
    KeyTooLong,
    ValueTooLong,
}

/// A B+-Tree backed by some storage.
#[derive(Debug)]
pub struct BTree<T: PageStorage> {
    root: Ptr,
    storage: T,
}

impl<T: PageStorage> BTree<T> {
    /// Create a new BTree.
    ///
    /// # Parameters
    ///
    /// - storage: The storage where tree nodes should be stored.
    pub fn new(storage: T) -> Self {
        let mut storage = storage;

        let root = if let Some(data) = storage.read(0) {
            assert!(data.len() >= 8, "master page does not contain a pointer");
            let ptr = u64::from_le_bytes(data[0..8].try_into().unwrap());
            assert!(storage.read(ptr).is_some(), "root pointer does not exist");
            ptr
        } else {
            storage
                .allocate_ptr(0)
                .expect("master page already allocated");

            let node = Node::empty_leaf();
            let ptr = storage.allocate(&node.data).expect("allocation error");

            storage
                .write(0, &ptr.to_le_bytes())
                .expect("allocation error");

            ptr
        };

        Self { root, storage }
    }

    /// Insert or update a key-value in the tree.
    ///
    /// # Parameters
    ///
    /// - key: The key to insert or update. The key must not be empty.
    /// - value: The value to insert.
    ///
    /// # Returns
    ///
    /// An `Ok` result signifies that insertion succeeded.
    pub fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<bool> {
        if key.is_empty() {
            return Err(BTreeError::EmptyKey);
        }
        if key.len() > MAX_KEY_SIZE {
            return Err(BTreeError::KeyTooLong);
        }
        if value.len() > MAX_VAL_SIZE {
            return Err(BTreeError::ValueTooLong);
        }

        // TODO: Deal with allocation failures.

        let root = Node::from_bytes(
            self.storage
                .read(self.root)
                .expect("invalid tree root pointer"),
        );
        let root = root.insert(key, value, &mut self.storage);

        let nodes = root.split();
        if nodes.len() == 1 {
            let root = self
                .storage
                .allocate(&nodes[0].data)
                .expect("root node too large");
            self.storage.deallocate(self.root);
            self.root = root;
        } else {
            let root = Node::new_internal(self.root);
            let root = root.insert_internal(0, nodes, &mut self.storage);
            let root = self
                .storage
                .allocate(&root.data)
                .expect("root node too large");
            self.storage.deallocate(self.root);
            self.root = root;
        }

        self.storage
            .write(0, &self.root.to_le_bytes())
            .expect("allocation error");

        // TODO: bool should signify if a new value was created or if the value was updated.
        Ok(true)
    }

    /// Get a value from the tree.
    ///
    /// # Parameters
    ///
    /// - key: The key to lookup
    ///
    /// # Returns
    ///
    /// If the key is found `Some(value)` is returned, otherwise `None` is
    /// returned. Note that a key exceeding the max size or an empty key always
    /// will return `None`.
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        if key.is_empty() {
            return None;
        }
        if key.len() > MAX_KEY_SIZE {
            return None;
        }
        let root = Node::from_bytes(
            self.storage
                .read(self.root)
                .expect("invalid tree root node"),
        );
        root.find_key(key, &self.storage)
    }
}

mod bnode {
    use crate::storage::PageStorage;
    use crate::PAGE_SIZE;

    use super::Ptr;

    const HEADER_SIZE: usize = 4;
    const KV_HEADER_SIZE: usize = 4;

    #[derive(Debug)]
    enum NodeType {
        Internal,
        Leaf,
    }

    /// A node in a B+-tree.
    ///
    /// The data structure has some special properties:
    ///
    /// 1. It is meant to be stored on disk, which is why it is represented as
    ///    a vector of bytes. This makes storing and loading it from memory easy.
    /// 2. The nodes are meant to be immutable, which means that any operation
    ///    that modifies a node will return a new node with the changes.
    ///
    /// Because of the first property the whole tree will likely not be loaded
    /// into memory at once. Nodes are instead dynamically loaded from storage
    /// when needed.
    ///
    /// There are two types of nodes, leaf and internal nodes. Leaf nodes store
    /// actual key-values while internal nodes store pointers to other nodes.
    /// Note that the pointers stored in internal nodes are not normal pointers,
    /// but pointers from [`SequentialStorage`]. Internal nodes have keys that
    /// correspond to the first key of the pointed to nodes. Keys in the node
    /// are always sorted.
    ///
    /// The binary representation of the parts making up the B-Tree are (all
    /// integer types are stored in little-endian (LE) format):
    ///
    /// Internal nodes
    /// ```text
    /// node_type (1)  u16,
    /// num_values     u16,
    /// pointer_list   u64 * num_values,
    /// offset_list    u16 * num_values,
    /// key_value_list (variable)
    /// ```
    ///
    /// Leaf nodes
    /// ```text
    /// node_type (2)  u16,
    /// num_values     u16,
    /// offset_list    u16 * num_values,
    /// key_value_list (variable)
    /// ```
    ///
    /// Key-values
    /// ```text
    /// key_length   u16,
    /// value_length u16,
    /// key          u8 * key_length,
    /// value        u8 * value_length,
    /// ```
    ///
    /// The node type is `1_u16` for internal nodes or `2_u16` for leaf nodes.
    /// The elements in the offset list is the offset in bytes from the start
    /// of the key-value list to the end of the corresponding key-value. Thus,
    /// the size of the key-value list in bytes is the last element in the
    /// offset list. Since internal nodes only store keys the value length is
    /// always 0.
    ///
    /// Nodes should always fit inside of [`PAGE_SIZE`]. As such the maximum
    /// key and value lengths need to be limited. Larger nodes can temporarily
    /// be created in memory, but will have to be split before being stored.
    ///
    /// Finally, for key lookup to work correctly there must always be a
    /// smallest key in the tree. This is ensured by having an empty key as a
    /// sentinel value.
    pub struct Node {
        pub data: Vec<u8>,
    }

    impl Node {
        /// Create a node from the binary data.
        ///
        /// # Parameters
        ///
        /// - `data`: The binary data representing the node.
        ///
        /// # Panics
        ///
        /// If the `data` is not a valid node.
        pub fn from_bytes(data: Vec<u8>) -> Self {
            let node = Self { data };
            node.verify_node_layout();
            node
        }

        /// Create an empty leaf node.
        ///
        /// The node has an empty key (`&[]`) as a sentinel value to ensure that
        /// there is always a smallest key in the tree.
        pub fn empty_leaf() -> Self {
            let data = vec![
                2, 0, // leaf node
                1, 0, // 1 key-value
                4, 0, // offset to end of node
                0, 0, // empty key length
                0, 0, // empty value length
            ];
            Self { data }
        }

        /// Create an new internal node with a single child node.
        ///
        /// Note: The function assumes that the child node pointed to by `ptr`
        /// has the sentinel value (`&[]`) as its first key.
        ///
        /// # Parameters
        ///
        /// - `ptr`: A pointer to the single child node.
        pub fn new_internal(ptr: Ptr) -> Self {
            let data = vec![
                1, 0, // internal node
                1, 0, // 1 key-value
                0, 0, 0, 0, 0, 0, 0, 0, // pointer
                4, 0, // offset to end of node
                0, 0, // empty key length
                0, 0, // empty value length
            ];
            let mut node = Self { data };
            node.set_pointer(0, ptr);
            node
        }

        pub fn find_key<T: PageStorage>(&self, key: &[u8], storage: &T) -> Option<Vec<u8>> {
            let index = self.find_index(key);
            match self.node_type() {
                NodeType::Internal => {
                    let ptr = self.pointer(index);
                    let node = Node::from_bytes(storage.read(ptr).expect("invalid node pointer"));
                    node.find_key(key, storage)
                }
                NodeType::Leaf => {
                    if self.key(index) == key {
                        let value: Vec<u8> = self.value(index).into();
                        Some(value)
                    } else {
                        None
                    }
                }
            }
        }

        /// Insert or update a key-value in the node.
        ///
        /// If the key exists the value will be updated, otherwise a new value
        /// will be inserted.
        ///
        /// Note that the nodes are immutable and this will create a new node.
        ///
        /// # Parameters
        ///
        /// - key: The key to insert or update.
        /// - value: The value to insert.
        /// - storage: The storage where the nodes are stored.
        ///
        /// # Returns
        ///
        /// The newly created node.
        pub fn insert<T: PageStorage>(&self, key: &[u8], value: &[u8], storage: &mut T) -> Self {
            let index = self.find_index(key);

            match self.node_type() {
                NodeType::Leaf => {
                    if self.key(index) == key {
                        self.update_leaf(index, key, value)
                    } else {
                        self.insert_leaf(index + 1, key, value)
                    }
                }
                NodeType::Internal => {
                    let ptr = self.pointer(index);
                    let node = Node::from_bytes(storage.read(ptr).expect("invalid node pointer"));

                    storage.deallocate(ptr);

                    let node = node.insert(key, value, storage);
                    let nodes = node.split();
                    self.insert_internal(index, nodes, storage)
                }
            }
        }

        /// Split a node.
        ///
        /// Depending on the size of the key-values stored in the node it can
        /// be split into 2 or 3 subnodes. If the node does not need to be
        /// split it will just be returned.
        ///
        /// # Returns
        ///
        /// A vector of new nodes.
        pub fn split(self) -> Vec<Self> {
            if self.data.len() <= PAGE_SIZE {
                return vec![self];
            }

            // TODO: This function will likely lead to small "right" nodes when
            // splitting into 3 nodes. Look into making it more balanced.

            let mut nodes = Vec::new();

            let half_size = self.size() / 2;

            let mut left_count = 0;
            let mut left_size = HEADER_SIZE;

            for i in 0..self.num_values() {
                if left_size + self.kv_size(i) > PAGE_SIZE || left_size >= half_size {
                    break;
                }
                left_size += self.kv_size(i);
                left_count += 1;
            }

            nodes.push(Self {
                data: vec![0; left_size],
            });

            nodes[0].set_header(self.node_type(), left_count);

            append_range(&mut nodes[0], &self, 0, 0, left_count);

            let mut middle_count = 0;
            let mut middle_size = HEADER_SIZE;

            for i in left_count..self.num_values() {
                if middle_size + self.kv_size(i) > PAGE_SIZE {
                    break;
                }
                middle_size += self.kv_size(i);
                middle_count += 1;
            }

            nodes.push(Self {
                data: vec![0; middle_size],
            });

            nodes[1].set_header(self.node_type(), middle_count);

            append_range(&mut nodes[1], &self, 0, left_count, middle_count);

            let mut right_count = 0;
            let mut right_size = HEADER_SIZE;

            for i in left_count + middle_count..self.num_values() {
                right_size += self.kv_size(i);
                right_count += 1;
            }

            if right_count > 0 {
                nodes.push(Self {
                    data: vec![0; right_size],
                });
                nodes[2].set_header(self.node_type(), right_count);
                append_range(
                    &mut nodes[2],
                    &self,
                    0,
                    left_count + middle_count,
                    right_count,
                );
            }

            for node in &nodes {
                node.verify_node_layout();
            }

            nodes
        }

        /// Insert into an internal node.
        ///
        /// Note that nodes are immutable and this will return a new node.
        ///
        /// The returned node might be too large to store and might have to be
        /// split.
        ///
        /// # Parameters
        ///
        /// - index: The index in the node to be inserted after.
        /// - nodes: A list of nodes to insert.
        /// - storage: The storage where nodes are stored.
        ///
        /// # Returns
        ///
        /// The new internal node.
        pub fn insert_internal<T: PageStorage>(
            &self,
            index: usize,
            nodes: Vec<Self>,
            storage: &mut T,
        ) -> Self {
            let inc = nodes.len();

            // size = 10 (u64 + u16, pointer + offset) + KV_size
            let element_size = |n: &Node, i| 10 + 4 + n.key(i).len();

            let new_kv_size = nodes.iter().map(|n| element_size(n, 0)).sum::<usize>();
            let old_kv_size = element_size(self, index);

            let mut node = Self {
                data: vec![0; new_kv_size + self.data.len() - old_kv_size],
            };

            node.set_header(NodeType::Internal, self.num_values() + inc - 1);

            append_range(&mut node, self, 0, 0, index);
            for (i, n) in nodes.into_iter().enumerate() {
                append_kv(&mut node, index + i, n.key(0), b"");
                let ptr = storage.allocate(&n.data).expect("TODO: clean up");
                node.set_pointer(index + i, ptr);
            }
            append_range(
                &mut node,
                self,
                index + inc,
                index + 1, // skip the updated key in the old node
                self.num_values() - index - 1,
            );

            node.verify_node_layout();
            node
        }

        fn verify_node_layout(&self) {
            // Check that node_type and num_values exist
            assert!(self.data.len() >= HEADER_SIZE);

            let node_type = u16::from_le_bytes(self.data[0..2].try_into().unwrap());
            assert!(node_type == 1 || node_type == 2);

            // Check that pointer and offset lists exist. This only needs to access num_values,
            // which has been checked to exist above.
            assert!(self.data.len() >= self.kv_position(0));

            // Check that size is correct. This needs to access the last element of the offset
            // list, which has been checked to exist above.
            assert_eq!(self.data.len(), self.size());

            // Offsets need to be strictly increasing since key-values can't have non-positive
            // size (even an empty node as size 4 (key-length + value-length)).
            for i in 1..self.num_values() {
                assert!(self.offset(i - 1) < self.offset(i))
            }
        }

        fn find_index(&self, key: &[u8]) -> usize {
            // TODO: Bisect, otherwise tree insertion will be O(n^2)
            let mut found = 0;
            for i in 0..self.num_values() {
                if key >= self.key(i) {
                    found = i;
                } else {
                    break;
                }
            }
            found
        }

        fn update_leaf(&self, index: usize, key: &[u8], value: &[u8]) -> Self {
            assert!(matches!(self.node_type(), NodeType::Leaf));

            let kv_size = key.len() + value.len() + KV_HEADER_SIZE;
            let old_kv_size = key.len() + self.value(index).len() + KV_HEADER_SIZE;

            let mut node = Self {
                data: vec![0; kv_size + self.data.len() - old_kv_size],
            };

            node.set_header(self.node_type(), self.num_values());

            append_range(&mut node, self, 0, 0, index);
            append_kv(&mut node, index, key, value);
            append_range(
                &mut node,
                self,
                index + 1,
                index + 1, // skip the updated key in the old node
                self.num_values() - index - 1,
            );

            node.verify_node_layout();
            node
        }

        fn insert_leaf(&self, index: usize, key: &[u8], value: &[u8]) -> Self {
            assert!(matches!(self.node_type(), NodeType::Leaf));

            let kv_size = key.len() + value.len() + KV_HEADER_SIZE;

            // New size = 2 (u16, offset) + kv_size + previous_size
            let mut node = Self {
                data: vec![0; 2 + kv_size + self.data.len()],
            };

            node.set_header(self.node_type(), self.num_values() + 1);

            append_range(&mut node, self, 0, 0, index);
            append_kv(&mut node, index, key, value);
            append_range(&mut node, self, index + 1, index, self.num_values() - index);

            node.verify_node_layout();
            node
        }

        fn key(&self, index: usize) -> &[u8] {
            assert!(index < self.num_values());

            let offset = self.kv_position(index);
            let offset_next = self.kv_position(index + 1);

            let kv = &self.data[offset..offset_next];
            &kv[KV_HEADER_SIZE..KV_HEADER_SIZE + key_length(kv)]
        }

        fn value(&self, index: usize) -> &[u8] {
            assert!(index < self.num_values());

            let offset = self.kv_position(index);
            let offset_next = self.kv_position(index + 1);

            let kv = &self.data[offset..offset_next];
            let keylen = key_length(kv);
            let valuelen = value_length(kv);

            &kv[KV_HEADER_SIZE + keylen..KV_HEADER_SIZE + keylen + valuelen]
        }

        fn pointer(&self, index: usize) -> Ptr {
            assert!(matches!(self.node_type(), NodeType::Internal));
            assert!(index < self.num_values());

            let offset = HEADER_SIZE + index * 8;

            u64::from_le_bytes(self.data[offset..offset + 8].try_into().unwrap())
        }

        fn set_pointer(&mut self, index: usize, pointer: Ptr) {
            assert!(matches!(self.node_type(), NodeType::Internal));
            assert!(index < self.num_values());

            let offset = HEADER_SIZE + index * 8;
            self.data[offset..offset + 8].copy_from_slice(&pointer.to_le_bytes());
        }

        fn set_header(&mut self, node_type: NodeType, num_values: usize) {
            let node_type = match node_type {
                NodeType::Internal => 1_u16,
                NodeType::Leaf => 2_u16,
            };
            let num_values = num_values as u16;

            self.data[0..2].copy_from_slice(&node_type.to_le_bytes());
            self.data[2..4].copy_from_slice(&num_values.to_le_bytes());
        }

        fn node_type(&self) -> NodeType {
            match u16::from_le_bytes(self.data[0..2].try_into().unwrap()) {
                1 => NodeType::Internal,
                2 => NodeType::Leaf,
                _ => unreachable!(),
            }
        }

        fn offset(&self, index: usize) -> usize {
            assert!(index <= self.num_values());

            if index == 0 {
                return 0;
            }

            let offset = self.offset_section_start() + 2 * (index - 1);
            u16::from_le_bytes(self.data[offset..offset + 2].try_into().unwrap()) as usize
        }

        fn set_offset(&mut self, index: usize, offset: usize) {
            assert!(index <= self.num_values());

            // Do nothing at index 0 since it's not stored in the offset list.
            // This is mostly for symmetry with Node::offset.
            if index == 0 {
                return;
            }

            let offset = offset as u16;
            let start = self.offset_section_start() + 2 * (index - 1);

            self.data[start..start + 2].copy_from_slice(&offset.to_le_bytes());
        }

        fn num_values(&self) -> usize {
            u16::from_le_bytes(self.data[2..4].try_into().unwrap()) as usize
        }

        fn offset_section_start(&self) -> usize {
            match self.node_type() {
                NodeType::Leaf => HEADER_SIZE,
                NodeType::Internal => HEADER_SIZE + self.num_values() * 8,
            }
        }

        fn kv_position(&self, index: usize) -> usize {
            let list_item_size = match self.node_type() {
                NodeType::Internal => 10,
                NodeType::Leaf => 2,
            };

            HEADER_SIZE + self.num_values() * list_item_size + self.offset(index)
        }

        fn kv_size(&self, index: usize) -> usize {
            let list_item_size = match self.node_type() {
                NodeType::Internal => 10,
                NodeType::Leaf => 2,
            };

            list_item_size + self.kv_position(index + 1) - self.kv_position(index)
        }

        fn size(&self) -> usize {
            self.kv_position(self.num_values())
        }
    }

    fn append_range(
        node: &mut Node,
        other: &Node,
        start_node: usize,
        start_other: usize,
        count: usize,
    ) {
        if count == 0 {
            return;
        }

        if let NodeType::Internal = node.node_type() {
            for i in 0..count {
                node.set_pointer(start_node + i, other.pointer(start_other + i));
            }
        }

        for i in 1..count + 1 {
            node.set_offset(
                start_node + i,
                node.offset(start_node) + other.offset(start_other + i) - other.offset(start_other),
            );
        }

        // Copy over KV pairs
        let node_copy_start = node.kv_position(start_node);
        let node_copy_end = node.kv_position(start_node + count);
        let other_copy_start = other.kv_position(start_other);
        let other_copy_end = other.kv_position(start_other + count);

        node.data[node_copy_start..node_copy_end]
            .copy_from_slice(&other.data[other_copy_start..other_copy_end]);
    }

    fn append_kv(node: &mut Node, index: usize, key: &[u8], value: &[u8]) {
        let kv_start = node.kv_position(index);
        node.data[kv_start..kv_start + 2].copy_from_slice(&(key.len() as u16).to_le_bytes());
        node.data[kv_start + 2..kv_start + 4].copy_from_slice(&(value.len() as u16).to_le_bytes());
        node.data[kv_start + 4..kv_start + 4 + key.len()].copy_from_slice(key);
        node.data[kv_start + 4 + key.len()..kv_start + 4 + key.len() + value.len()]
            .copy_from_slice(value);

        node.set_offset(
            index + 1,
            node.offset(index) + KV_HEADER_SIZE + key.len() + value.len(),
        );
    }

    fn key_length(kv: &[u8]) -> usize {
        u16::from_le_bytes(kv[0..2].try_into().unwrap()) as usize
    }

    fn value_length(kv: &[u8]) -> usize {
        u16::from_le_bytes(kv[2..4].try_into().unwrap()) as usize
    }

    impl std::fmt::Debug for Node {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let keys: Vec<&[u8]> = (0..self.num_values()).map(|i| self.key(i)).collect();
            let values: Vec<&[u8]> = (0..self.num_values()).map(|i| self.value(i)).collect();
            write!(
            f,
            "Node {{\n    node_type: {:?},\n    num_values: {},\n    keys: {:?},\n    values: {:?}\n}}",
            self.node_type(),
            self.num_values(),
            keys,
            values
        )
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_leaf_node_binary_representation() {
            // Create a leaf node with a single key-value pair
            let data = vec![
                2, 0, // leaf node
                1, 0, // 1 key-value
                // offset list
                12, 0, // offset to the end of the key-value
                // key-values
                3, 0, // key length
                5, 0, // value length
                b'k', b'e', b'y', // key
                b'v', b'a', b'l', b'u', b'e', // value
            ];
            let node = Node::from_bytes(data);

            assert_eq!(node.num_values(), 1);
            assert_eq!(node.key(0), b"key");
            assert_eq!(node.value(0), b"value");
        }

        #[test]
        fn test_internal_node_binary_representation() {
            // Create an internal node with a single pointer
            let data = vec![
                1, 0, // internal node
                1, 0, // 1 key
                // pointer list
                1, 0, 0, 0, 0, 0, 0, 0, // Ptr(1)
                // offset list
                7, 0, // offset to the end of the key
                // keys
                3, 0, // key length
                0, 0, // no value
                b'k', b'e', b'y', // key
            ];
            let node = Node::from_bytes(data);

            assert_eq!(node.num_values(), 1);
            assert_eq!(node.pointer(0), 1);
            assert_eq!(node.key(0), b"key");
            assert_eq!(node.value(0), b"");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::VecStorage;

    #[test]
    fn test_insert_and_update_single_key_in_tree() {
        let key: &[u8] = b"key";
        let mut tree = BTree::new(VecStorage::new());

        assert!(tree.get(b"non-existent").is_none());
        tree.insert(key, b"value").unwrap();
        assert_eq!(tree.get(key).unwrap(), b"value");
        assert!(tree.get(b"non-existent").is_none());

        tree.insert(key, b"eulav").unwrap();
        assert_eq!(tree.get(key).unwrap(), b"eulav");
        assert!(tree.get(b"non-existent").is_none());
    }

    #[test]
    fn test_insert_empty_key_in_tree() {
        let mut tree = BTree::new(VecStorage::new());

        assert!(tree.get(b"non-existent").is_none());
        assert!(matches!(
            tree.insert(b"", b"value"),
            Err(BTreeError::EmptyKey)
        ));
        assert!(tree.get(b"non-existent").is_none());
    }

    #[test]
    fn test_insert_many_in_tree() {
        let mut tree = BTree::new(VecStorage::new());
        let value = [0_u8; 100];

        // Insert multiple pages of data to ensure the tree can handle splits
        for i in 0_u64..100_u64 {
            let key = i.to_le_bytes();
            assert!(tree.insert(&key, &value).is_ok());
        }
        for i in (100_u64..200_u64).rev() {
            let key = i.to_le_bytes();
            assert!(tree.insert(&key, &value).is_ok());
        }

        assert!(tree.get(b"non-existent").is_none());

        for i in 0_u64..200_u64 {
            let key = i.to_le_bytes();
            assert_eq!(tree.get(&key).unwrap(), &value);
        }

        // Check that the storage can be loaded correctly into a new BTree
        let tree_clone = BTree::new(tree.storage.clone());
        assert!(tree_clone.get(b"non-existent").is_none());
        for i in 0_u64..200_u64 {
            let key = i.to_le_bytes();
            assert_eq!(tree_clone.get(&key).unwrap(), &value);
        }
    }
}
