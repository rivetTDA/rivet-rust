// #![feature(alloc_system, global_allocator, allocator_api, try_trait, box_syntax, use_extern_macros)]
#[macro_use]
extern crate ndarray;
extern crate libc;
extern crate num_traits;
extern crate num_rational;
extern crate rand;
extern crate csv;
extern crate noisy_float;
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
extern crate serde;
extern crate core;

mod rivet;
#[cfg(feature="hera")]
mod hera;
#[cfg(feature="hera")]
mod matching_distance;
mod hilbert_distance;
