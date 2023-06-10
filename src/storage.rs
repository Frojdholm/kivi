use crate::PAGE_SIZE;

#[derive(Debug)]
pub enum WriteError {
    WriteFailed,
    DataTooLarge,
    PageNotFound,
}

#[derive(Debug)]
pub struct AlreadyAllocated {}

/// Represents a storage where allocation size is limited to pages.
///
/// Operations are more efficient if pointers are accessed in order. In
/// particular allocating a higher pointer might require all lower pointers
/// to be allocated.
pub trait PageStorage: PageAllocate + PageRead + PageWrite {}

pub trait PageAllocate {
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
}

pub trait PageRead {
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
}

pub trait PageWrite {
    /// Write to a pointer in the storage.
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

#[derive(Debug, Default, Clone)]
pub struct VecStorage {
    data: Vec<Page>,
    freelist: Vec<u64>,
}

impl VecStorage {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            freelist: Vec::new(),
        }
    }
}

impl PageStorage for VecStorage {}

impl PageAllocate for VecStorage {
    fn allocate_ptr(&mut self, ptr: u64) -> Result<(), AlreadyAllocated> {
        if let Some(pos) = self.freelist.iter().position(|x| *x == ptr) {
            self.freelist.remove(pos);
            return Ok(());
        }

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
        if let Some(ptr) = self.freelist.pop() {
            self.data[ptr as usize].write(data)?;
            Ok(ptr)
        } else {
            let index = self.data.len() as u64;
            self.data.push(Page::new(data)?);
            Ok(index)
        }
    }

    fn deallocate(&mut self, ptr: u64) {
        self.freelist.push(ptr);
    }
}

impl PageRead for VecStorage {
    fn read(&self, ptr: u64) -> Option<Vec<u8>> {
        self.data.get(ptr as usize).map(|page| page.bytes.clone())
    }
}

impl PageWrite for VecStorage {
    fn write(&mut self, ptr: u64, data: &[u8]) -> Result<(), WriteError> {
        if let Some(page) = self.data.get_mut(ptr as usize) {
            if page.bytes.len() < data.len() {
                return Err(WriteError::DataTooLarge);
            }

            page.bytes[..data.len()].copy_from_slice(data);

            Ok(())
        } else {
            Err(WriteError::PageNotFound)
        }
    }
}

#[derive(Debug, Clone)]
struct Page {
    bytes: Vec<u8>,
}

impl Page {
    fn empty() -> Self {
        Self {
            bytes: vec![0; PAGE_SIZE],
        }
    }

    fn new(data: &[u8]) -> Result<Self, WriteError> {
        if data.len() > PAGE_SIZE {
            return Err(WriteError::DataTooLarge);
        }

        let mut bytes = vec![0; PAGE_SIZE];
        bytes[..data.len()].copy_from_slice(data);
        Ok(Self { bytes })
    }

    fn write(&mut self, data: &[u8]) -> Result<(), WriteError> {
        if data.len() > PAGE_SIZE {
            return Err(WriteError::DataTooLarge);
        }

        self.bytes[..data.len()].copy_from_slice(data);
        Ok(())
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
        assert!(storage.read(ptr).unwrap().starts_with(b"Hello world"));
    }
}
