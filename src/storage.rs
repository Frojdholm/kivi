use crate::PAGE_SIZE;

#[derive(Debug)]
pub struct WriteError {}

#[derive(Debug)]
pub struct AlreadyAllocated {}

/// Represents a storage where access is sequential.
///
/// Operations are more efficient if pointers are accessed in order. In
/// particular allocating a higher pointer might require all lower pointers
/// to be allocated.
pub trait SequentialStorage {
    /// Allocate a pointer.
    ///
    /// Note that allocating a pointer might require all lower pointers to be
    /// allocated.
    ///
    /// # Parameters
    /// - ptr: The pointer to allocate.
    ///
    /// # Returns
    ///
    /// An `Ok` result is returned if the pointer is successfully allocated.
    fn allocate_ptr(&mut self, ptr: u64) -> Result<(), AlreadyAllocated>;

    /// Allocate data.
    ///
    /// # Parameters
    ///
    /// - data: The data to store. The data will be copied into the storage.
    ///
    /// # Returns
    ///
    /// On succeful allocation `Ok(pointer)` is returned. If the data could not
    /// be written `Err(WriteError {})` is returned.
    fn allocate(&mut self, data: &[u8]) -> Result<u64, WriteError>;

    /// Deallocate a pointer.
    ///
    /// Implemenations are not required to deallocate the memory and it is not
    /// guaranteed that allocating a deallocated pointer will succeed.
    ///
    /// # Parameters
    ///
    /// - ptr: The pointer to deallocate.
    fn deallocate(&mut self, ptr: u64);

    /// Read a pointer from the storage.
    ///
    /// # Parameters
    ///
    /// - ptr: The pointer to read.
    ///
    /// # Returns
    ///
    /// If the pointer is allocated the data is copied from the storage and
    /// returned as `Some(data)`, otherwise `None` is returned.
    fn read(&self, ptr: u64) -> Option<Vec<u8>>;

    /// Write to a pointer from in the storage.
    ///
    /// # Parameters
    ///
    /// - ptr: The pointer to write to.
    /// - data: The data to write.
    ///
    /// # Returns
    ///
    /// An `Ok` result is returned if the write was successful.
    fn write(&mut self, ptr: u64, data: &[u8]) -> Result<(), WriteError>;
}

#[derive(Debug, Default)]
pub struct VecStorage {
    data: Vec<Page>,
}

impl VecStorage {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
}

impl SequentialStorage for VecStorage {
    fn allocate_ptr(&mut self, ptr: u64) -> Result<(), AlreadyAllocated> {
        let index = ptr as usize;
        if self.data.len() > index {
            return Err(AlreadyAllocated {});
        }

        while self.data.len() <= index {
            self.data.push(Page::empty());
        }

        Ok(())
    }

    fn allocate(&mut self, data: &[u8]) -> Result<u64, WriteError> {
        let index = self.data.len() as u64;
        self.data.push(Page::new(data)?);
        Ok(index)
    }

    fn deallocate(&mut self, _: u64) {
        // NOOP
    }

    fn read(&self, ptr: u64) -> Option<Vec<u8>> {
        self.data.get(ptr as usize).map(|page| {
            let size = page.used as usize;
            page.bytes[..size].into()
        })
    }

    fn write(&mut self, ptr: u64, data: &[u8]) -> Result<(), WriteError> {
        if let Some(page) = self.data.get_mut(ptr as usize) {
            if page.bytes.len() < data.len() {
                return Err(WriteError {});
            }

            page.bytes[..data.len()].copy_from_slice(data);
            page.used = data.len() as u16;

            Ok(())
        } else {
            Err(WriteError {})
        }
    }
}

#[derive(Debug)]
struct Page {
    used: u16,
    bytes: [u8; PAGE_SIZE],
}

impl Page {
    fn empty() -> Self {
        Self {
            used: 0,
            bytes: [0; PAGE_SIZE],
        }
    }

    fn new(data: &[u8]) -> Result<Self, WriteError> {
        if data.len() > PAGE_SIZE {
            return Err(WriteError {});
        }

        let mut bytes = [0; PAGE_SIZE];
        bytes[..data.len()].copy_from_slice(data);
        Ok(Self {
            used: data.len() as u16,
            bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn test_allocated_pointers_are_unique() {
        let mut storage = VecStorage::new();
        let mut ptrs = HashSet::new();

        // Allocate a number of pointers
        for _ in 0..20 {
            let ptr = storage.allocate(&[]).unwrap();
            assert!(!ptrs.contains(&ptr));
            ptrs.insert(ptr);
        }

        // Then allocate and deallocate some pointers and make sure that the
        // returned pointers are still unique
        for _ in 0..80 {
            let mut mid = ptrs.iter().skip(ptrs.len() / 2);
            let mid = *mid.next().unwrap();
            ptrs.remove(&mid);
            storage.deallocate(mid);

            let ptr = storage.allocate(&[]).unwrap();
            assert!(!ptrs.contains(&ptr));
            ptrs.insert(ptr);
        }
    }

    #[test]
    fn test_empty_storage_get_should_return_none() {
        let storage = VecStorage::new();
        assert!(storage.read(1).is_none());
        assert!(storage.read(100).is_none());
        assert!(storage.read(12).is_none());
        assert!(storage.read(2).is_none());
        assert!(storage.read(3).is_none());
    }

    #[test]
    fn test_allocated_pointer_can_be_read() {
        let mut storage = VecStorage::new();
        let ptr = storage.allocate(b"Hello world").unwrap();
        assert_eq!(storage.read(ptr), Some(b"Hello world".to_vec()));
    }
}