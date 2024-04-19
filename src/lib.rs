#![deny(missing_docs)]
#![deny(clippy::perf)]
#![warn(clippy::pedantic)]
#![warn(clippy::cargo)]

//! Kivi is a persistent key-value database.
//!
//! The database is based on a persistent B+-Tree.

pub mod btree;
pub mod storage;

/// The page size used in kivi.
pub const PAGE_SIZE: usize = 4096;
