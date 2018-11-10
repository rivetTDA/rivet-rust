#[cfg(all(unix, not(target_os="macos")))]
fn main() {
    println!("cargo:rustc-link-lib=static=rivet");
    println!("cargo:rustc-link-search=../rivet/build");
    println!("cargo:rustc-flags=-l dylib=stdc++");
    println!("cargo:rustc-flags=-l dylib=boost_system");
    println!("cargo:rustc-flags=-l dylib=boost_serialization");
    println!("cargo:rustc-link-lib=static=bottleneck");
    println!("cargo:rustc-link-search=../hera/geom_bottleneck/build");
}

#[cfg(target_os="macos")]
fn main() {
    println!("cargo:rustc-link-lib=static=rivet");
    println!("cargo:rustc-link-search=../rivet/build");
    println!("cargo:rustc-flags=-l dylib=c++");
    println!("cargo:rustc-flags=-l dylib=boost_system");
    println!("cargo:rustc-flags=-l dylib=boost_serialization");
    println!("cargo:rustc-link-lib=static=bottleneck");
    println!("cargo:rustc-link-search=../hera/geom_bottleneck/build");
}
