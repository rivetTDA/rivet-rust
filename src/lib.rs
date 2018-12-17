// #![feature(alloc_system, global_allocator, allocator_api, try_trait, box_syntax, use_extern_macros)]
#[macro_use]
extern crate ndarray;
extern crate csv;
extern crate libc;
extern crate noisy_float;
extern crate num_rational;
extern crate num_traits;
extern crate rand;
// extern crate time;
extern crate num_iter;

#[macro_use]
extern crate serde_derive;
// extern crate serde_yaml;
// extern crate docopt;
// extern crate alloc_system;
extern crate itertools;

// use alloc_system::System;

// #[global_allocator]
// static A: System = System;

extern crate flexi_logger;
#[macro_use]
extern crate log;
extern crate core;
extern crate serde;

#[cfg(feature = "hera")]
pub mod hera;
pub mod hilbert_distance;
#[cfg(feature = "hera")]
pub mod matching_distance;
pub mod rivet;
