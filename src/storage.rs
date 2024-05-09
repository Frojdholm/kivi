//! Allocation and memory handling routines.
//!
//! This module contains the base building blocks for interacting directly
//! with memory. Storage backends defined in these modules are meant to be
//! wrapped in higher level APIs to handle memory efficiently.

use std::{
    fs::{File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::Path,
};

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

/// A file-backed storage.
#[derive(Debug)]
pub struct Disk {
    file: File,
    num_pages: u64,
}

impl Disk {
    /// Create a `Disk` from the path.
    ///
    /// # Errors
    ///
    /// If the file does not exist or if the file size is not a multiple
    /// of the page size this function returns an error.
    pub fn from_path(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .create(true)
            .write(true)
            .truncate(false)
            .open(path)?;
        let metadata = std::fs::metadata(path)?;
        Ok(Self {
            file,
            // Note that this constructs the storage over the aligned part of the
            // file, if there is extra space for some reason that is ignored.
            num_pages: metadata.len() / PAGE_SIZE as u64,
        })
    }

    fn seek_to(&self, ptr: u64) -> io::Result<u64> {
        assert!(ptr < self.num_pages);
        let mut file = &self.file;
        file.seek(SeekFrom::Start(Self::offset(ptr)))
    }

    fn allocate_next(&mut self) -> io::Result<u64> {
        let ptr = self.num_pages;
        self.num_pages += 1;
        self.file.set_len(self.size())?;
        Ok(ptr)
    }

    fn size(&self) -> u64 {
        self.num_pages * PAGE_SIZE as u64
    }

    fn offset(ptr: u64) -> u64 {
        ptr * PAGE_SIZE as u64
    }
}

impl PageStorage for Disk {
    fn sync(&mut self) -> io::Result<()> {
        self.file.sync_all()
    }

    fn allocated(&self) -> u64 {
        self.num_pages
    }

    fn allocate_ptr(&mut self, ptr: u64) -> io::Result<()> {
        if ptr < self.num_pages {
            return Ok(());
        }
        self.num_pages = ptr + 1;
        self.file.set_len(self.size())?;
        Ok(())
    }

    fn allocate(&mut self, data: &[u8]) -> io::Result<u64> {
        let ptr = self.allocate_next()?;
        self.write(ptr, data)?;
        Ok(ptr)
    }

    fn read(&self, ptr: u64) -> io::Result<Option<Vec<u8>>> {
        if ptr >= self.num_pages {
            return Ok(None);
        }
        self.seek_to(ptr)?;
        let mut file = &self.file;
        let mut buf = vec![0_u8; PAGE_SIZE];
        file.read_exact(&mut buf)?;
        Ok(Some(buf))
    }

    fn write(&mut self, ptr: u64, data: &[u8]) -> io::Result<()> {
        assert!(ptr < self.num_pages, "the pointer must be allocated");
        self.seek_to(ptr)?;
        self.file.write_all(data)?;
        Ok(())
    }
}

/// An in-memory storage.
#[derive(Debug, Default, Clone)]
pub struct Memory {
    data: Vec<Page>,
}

impl Memory {
    /// Create an empty in-memory storage
    #[must_use]
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
}

impl PageStorage for Memory {
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

    fn allocated_pointers_are_unique<S: PageStorage>(storage: S) {
        let mut storage = storage;
        let mut ptrs = HashSet::new();

        // Allocate a number of pointers
        for _ in 0..20 {
            let ptr = storage.allocate(&[]).unwrap();
            assert!(!ptrs.contains(&ptr));
            ptrs.insert(ptr);
        }
    }

    fn empty_storage_get_should_return_none<S: PageStorage>(storage: S) {
        let storage = storage;
        assert!(storage.read(1).unwrap().is_none());
        assert!(storage.read(100).unwrap().is_none());
        assert!(storage.read(12).unwrap().is_none());
        assert!(storage.read(2).unwrap().is_none());
        assert!(storage.read(3).unwrap().is_none());
    }

    fn allocated_pointer_can_be_read<S: PageStorage>(storage: S) {
        let mut storage = storage;
        let ptr = storage.allocate(b"Hello world").unwrap();
        assert!(storage
            .read(ptr)
            .unwrap()
            .unwrap()
            .starts_with(b"Hello world"));
    }

    #[test]
    fn test_allocated_pointers_are_unique() {
        allocated_pointers_are_unique(Memory::new());
        allocated_pointers_are_unique(Disk {
            file: tempfile::tempfile().unwrap(),
            num_pages: 0,
        });
    }

    #[test]
    fn test_empty_storage_get_should_return_none() {
        empty_storage_get_should_return_none(Memory::new());
        empty_storage_get_should_return_none(Disk {
            file: tempfile::tempfile().unwrap(),
            num_pages: 0,
        });
    }

    #[test]
    fn test_allocated_pointer_can_be_read() {
        allocated_pointer_can_be_read(Memory::new());
        allocated_pointer_can_be_read(Disk {
            file: tempfile::tempfile().unwrap(),
            num_pages: 0,
        });
    }
}
