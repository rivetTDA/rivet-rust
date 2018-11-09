# rivet-rust
Rust API for RIVET

This package provides access to [RIVET](http://rivet.online), a tool for computing and visualizing two-parameter persistent homology.
It links directly to the RIVET static library, providing a high-performance interface through which you can compute persistence modules
from point cloud data or other input formats, analyze them, and compare them using a variety of distance metrics. Note that
because this library links directly with RIVET, it is GPL licensed. If you need a non-GPL API, consider the [rivet-python](https://github.com/rivettda/rivet-python) package for Python.

## Installation

In the future, installation via crates.io will be supported. Currently, the best process is:

1. Choose an install location for RIVET, (optionally Hera), and rivet-rust. Let's call this `$SRC`.
2. Clone and build RIVET in `$SRC/rivet` following the instructions for RIVET. This will ensure you have the 
   RIVET static library in `$SRC/rivet/build`.
3. Optionally, if you want to use the matching distance computations, clone our fork of Hera(https://bitbucket.org/xoltar/hera)
   and build it according to instructions. Then you will have the Hera static libary in `$SRC/hera/geom_bottleneck/build`.
4. Clone this repository, so you will have `$SRC/rivet-rust`.
5. Run `cargo build`, or if you installed Hera and want to use it, `cargo build --features hera`.
6. Assuming all goes well, you can now use this library in your projects by referring to it in your `Cargo.toml` like this:
   ```
   [dependencies]
   rivet-rust = {path = "$SRC/rivet-rust"}
   ```
   or
   ```
   [dependencies]
   rivet-rust = {path = "$SRC/rivet-rust" features="hera"}
   ```
   Remember to replace `$SRC` with the real path!

### Installation with RIVET or Hera in different locations
The build script will find RIVET and Hera if they are installed in the locations
described above, or they are on the normal library search path. If neither of 
those is the case, you can override the build script using [standard cargo techniques](https://doc.rust-lang.org/cargo/reference/build-scripts.html#overriding-build-scripts).