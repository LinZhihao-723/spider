//! Links the `clp_search` TDL package against the `clp-s` C-API shared library (`libclp-s.so`).
//!
//! The library directory defaults to the handoff location and can be overridden with the
//! `CLP_S_LIB_DIR` environment variable at build time. The rpath entries let the loader resolve
//! `libclp-s.so` (and its `/usr/local/lib` dependencies: `libarchive`, `libzstd`) when the task
//! executor `dlopen`s this package.

fn main() {
    let lib_dir = std::env::var("CLP_S_LIB_DIR")
        .unwrap_or_else(|_| "/home/lzh/dev/clp/claude/clp-s-lib".to_owned());

    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rustc-link-lib=dylib=clp-s");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/local/lib");
    println!("cargo:rerun-if-env-changed=CLP_S_LIB_DIR");
}
