use crate::PAGE_SIZE;

/// Error returned when data is larger than a page.
#[derive(Debug, Clone, Copy)]
pub struct DataTooLargeError {
    /// The size of the data that was attempted to be written
    pub size: usize,
}

/// Error returned when the page being written to is not allocated.
#[derive(Debug, Clone, Copy)]
pub struct PageNotAllocatedError {
    /// The pointer that was attempted to be written to
    pub ptr: u64,
}

/// Error returned when the page being allocated is already allocated.
#[derive(Debug, Clone, Copy)]
pub struct AlreadyAllocatedError {
    /// The pointer that was attempted to be allocated
    pub ptr: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct PageBuffer<'a> {
    pub buf: &'a [u8],
}

impl<'a> PageBuffer<'a> {
    pub fn new(buf: &'a [u8]) -> Result<Self, DataTooLargeError> {
        if buf.len() > PAGE_SIZE {
            return Err(DataTooLargeError { size: buf.len() });
        }
        Ok(Self { buf })
    }
}

/// Represents a storage where allocation size is limited to pages.
///
/// Operations are more efficient if pointers are accessed in order. In
/// particular allocating a higher pointer might require all lower pointers
/// to be allocated.
///
/// TODO: Add a compacting operation that removes holes in the storage.
pub trait PageStorage {
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
    fn allocate_ptr(&mut self, ptr: u64) -> Result<(), AlreadyAllocatedError>;

    /// Allocate data.
    ///
    /// # Parameters
    ///
    /// - data: The data to store. The data will be copied into the storage.
    fn allocate(&mut self, data: &[u8]) -> u64;

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
    fn write(&mut self, ptr: u64, data: &[u8]) -> Result<(), PageNotAllocatedError>;
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

impl PageStorage for VecStorage {
    fn allocate_ptr(&mut self, ptr: u64) -> Result<(), AlreadyAllocatedError> {
        if let Some(pos) = self.freelist.iter().position(|x| *x == ptr) {
            self.freelist.remove(pos);
            return Ok(());
        }

        let index = ptr as usize;
        if self.data.len() > index {
            return Err(AlreadyAllocatedError { ptr });
        }

        while self.data.len() <= index {
            self.data.push(Page::empty());
        }

        Ok(())
    }

    fn allocate(&mut self, data: &[u8]) -> u64 {
        assert!(data.len() <= PAGE_SIZE);
        if let Some(ptr) = self.freelist.pop() {
            self.data[ptr as usize].write(data);
            ptr
        } else {
            let index = self.data.len() as u64;
            self.data.push(Page::new(data));
            index
        }
    }

    fn read(&self, ptr: u64) -> Option<Vec<u8>> {
        self.data.get(ptr as usize).map(|page| page.bytes.clone())
    }

    fn write(&mut self, ptr: u64, data: &[u8]) -> Result<(), PageNotAllocatedError> {
        assert!(data.len() <= PAGE_SIZE);
        if let Some(page) = self.data.get_mut(ptr as usize) {
            page.bytes[..data.len()].copy_from_slice(data);

            Ok(())
        } else {
            Err(PageNotAllocatedError { ptr })
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

    fn new(data: &[u8]) -> Self {
        assert!(data.len() <= PAGE_SIZE);
        let mut bytes = vec![0; PAGE_SIZE];
        bytes[..data.len()].copy_from_slice(data);
        Self { bytes }
    }

    fn write(&mut self, data: &[u8]) {
        assert!(data.len() <= PAGE_SIZE);
        self.bytes[..data.len()].copy_from_slice(data);
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
            let ptr = storage.allocate(&[]);
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
        let ptr = storage.allocate(b"Hello world");
        assert!(storage.read(ptr).unwrap().starts_with(b"Hello world"));
    }
}
