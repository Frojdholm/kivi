//! Allocation and memory handling routines.
//!
//! This module contains the base building blocks for interacting directly
//! with memory. Storage backends defined in these modules are meant to be
//! wrapped in higher level APIs to handle memory efficiently.

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

/// A byte buffer guaranteed to fit into a page.
#[derive(Debug, Clone, Copy)]
pub struct PageBuffer<'a> {
    /// The underlying byte slice.
    pub buf: &'a [u8],
}

impl<'a> PageBuffer<'a> {
    /// Create a new `PageBuffer`.
    ///
    /// # Errors
    ///
    /// If the underlying buffer is larger than a page a [`DataTooLargeError`]
    /// is returned instead.
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
#[allow(clippy::module_name_repetitions)]
pub trait PageStorage {
    /// Allocate a pointer.
    ///
    /// Note that allocating a pointer might require all lower pointers to be
    /// allocated.
    ///
    /// # Errors
    ///
    /// If the pointer being allocated is already allocated this function returns
    /// an error instead.
    fn allocate_ptr(&mut self, ptr: u64) -> Result<(), AlreadyAllocatedError>;

    /// Allocate a page and write data to it.
    ///
    /// If the buffer does not fit into a page it will be truncated.
    fn allocate(&mut self, data: &[u8]) -> u64;

    /// Read a pointer from the storage.
    ///
    /// If the pointer is allocated the data is copied from the storage and
    /// returned as `Some(data)`, otherwise `None` is returned.
    fn read(&self, ptr: u64) -> Option<Vec<u8>>;

    /// Write data to a pointer in the storage.
    ///
    /// If the buffer does not fit into a page it will be truncated.
    ///
    /// # Errors
    ///
    /// If the pointer hasn't been allocated this function returns an error
    /// instead.
    fn write(&mut self, ptr: u64, data: &[u8]) -> Result<(), PageNotAllocatedError>;
}

/// An in-memory storage.
#[derive(Debug, Default, Clone)]
#[allow(clippy::module_name_repetitions)]
pub struct VecStorage {
    data: Vec<Page>,
    freelist: Vec<u64>,
}

impl VecStorage {
    /// Create an empty `VecStorage`
    #[must_use]
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

        let index: usize = ptr
            .try_into()
            .expect("the number of pages should always fit in usize");
        if self.data.len() > index {
            return Err(AlreadyAllocatedError { ptr });
        }

        while self.data.len() <= index {
            self.data.push(Page::empty());
        }

        Ok(())
    }

    fn allocate(&mut self, data: &[u8]) -> u64 {
        let data = &data[..PAGE_SIZE.min(data.len())];
        if let Some(ptr) = self.freelist.pop() {
            let index: usize = ptr
                .try_into()
                .expect("the number of pages should always fit in usize");
            self.data[index].write(data);
            ptr
        } else {
            let index = self.data.len() as u64;
            self.data.push(Page::new(data));
            index
        }
    }

    fn read(&self, ptr: u64) -> Option<Vec<u8>> {
        let index: usize = ptr
            .try_into()
            .expect("the number of pages should always fit in usize");
        self.data.get(index).map(|page| page.bytes.clone())
    }

    fn write(&mut self, ptr: u64, data: &[u8]) -> Result<(), PageNotAllocatedError> {
        let data = &data[..PAGE_SIZE.min(data.len())];
        let index: usize = ptr
            .try_into()
            .expect("the number of pages should always fit in usize");
        if let Some(page) = self.data.get_mut(index) {
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
