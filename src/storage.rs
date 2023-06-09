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
