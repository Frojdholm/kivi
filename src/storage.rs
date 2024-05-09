//! Allocation and memory handling routines.
//!
//! This module contains the base building blocks for interacting directly
//! with memory. Storage backends defined in these modules are meant to be
//! wrapped in higher level APIs to handle memory efficiently.

use std::io;

use crate::PAGE_SIZE;

/// Error returned when data is larger than a page.
#[derive(Debug, Clone, Copy)]
pub struct DataTooLargeError {
    /// The size of the data that was attempted to be written
    pub size: usize,
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
    /// Sync write operations if needed
    ///
    /// # Errors
    ///
    /// If the underlying io operation fails this functions errors.
    fn sync(&mut self) -> io::Result<()> {
        Ok(())
    }

    /// Return the number of allocated pages.
    fn allocated(&self) -> u64;

    /// Allocate a pointer.
    ///
    /// Note that allocating a pointer might require all lower pointers to be
    /// allocated.
    ///
    /// If the pointer has already been allocated this function is a noop.
    ///
    /// # Errors
    ///
    /// If the underlying io operation fails this functions errors.
    fn allocate_ptr(&mut self, ptr: u64) -> io::Result<()>;

    /// Allocate a page and write data to it.
    ///
    /// If the buffer does not fit into a page it will be truncated.
    ///
    /// # Errors
    ///
    /// If the underlying io operation fails this functions errors.
    ///
    /// # Returns
    ///
    /// The newly allocated pointer
    fn allocate(&mut self, data: &[u8]) -> io::Result<u64>;

    /// Read a pointer from the storage.
    ///
    /// If the pointer is allocated the data is copied from the storage and
    /// returned as `Some(data)`, otherwise `None` is returned.
    ///
    /// # Errors
    ///
    /// If the underlying io operation fails this functions errors.
    fn read(&self, ptr: u64) -> io::Result<Option<Vec<u8>>>;

    /// Write data to a pointer in the storage.
    ///
    /// If the buffer does not fit into a page it will be truncated.
    ///
    /// # Errors
    ///
    /// If the underlying io operation fails this functions errors.
    ///
    /// # Panics
    ///
    /// This function panics if the pointer hasn't been allocated.
    fn write(&mut self, ptr: u64, data: &[u8]) -> io::Result<()>;
}

/// An in-memory storage.
#[derive(Debug, Default, Clone)]
#[allow(clippy::module_name_repetitions)]
pub struct VecStorage {
    data: Vec<Page>,
}

impl VecStorage {
    /// Create an empty `VecStorage`
    #[must_use]
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
}

impl PageStorage for VecStorage {
    fn allocated(&self) -> u64 {
        self.data.len() as u64
    }

    fn allocate_ptr(&mut self, ptr: u64) -> io::Result<()> {
        let index: usize = ptr
            .try_into()
            .expect("the number of pages should always fit in usize");
        if self.data.len() > index {
            return Ok(());
        }

        while self.data.len() <= index {
            self.data.push(Page::empty());
        }
        Ok(())
    }

    fn allocate(&mut self, data: &[u8]) -> io::Result<u64> {
        let data = &data[..PAGE_SIZE.min(data.len())];
        let index = self.data.len() as u64;

        self.data.push(Page::new(data));
        Ok(index)
    }

    fn read(&self, ptr: u64) -> io::Result<Option<Vec<u8>>> {
        let index: usize = ptr
            .try_into()
            .expect("the number of pages should always fit in usize");
        Ok(self.data.get(index).map(|page| page.bytes.clone()))
    }

    fn write(&mut self, ptr: u64, data: &[u8]) -> io::Result<()> {
        assert!(
            ptr < data.len() as u64,
            "the pointer must always be allocated"
        );
        let data = &data[..PAGE_SIZE.min(data.len())];
        let index: usize = ptr
            .try_into()
            .expect("the number of pages should always fit in usize");

        self.data[index].bytes[..data.len()].copy_from_slice(data);
        Ok(())
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
    }

    #[test]
    fn test_empty_storage_get_should_return_none() {
        let storage = VecStorage::new();
        assert!(storage.read(1).unwrap().is_none());
        assert!(storage.read(100).unwrap().is_none());
        assert!(storage.read(12).unwrap().is_none());
        assert!(storage.read(2).unwrap().is_none());
        assert!(storage.read(3).unwrap().is_none());
    }

    #[test]
    fn test_allocated_pointer_can_be_read() {
        let mut storage = VecStorage::new();
        let ptr = storage.allocate(b"Hello world").unwrap();
        assert!(storage
            .read(ptr)
            .unwrap()
            .unwrap()
            .starts_with(b"Hello world"));
    }
}
