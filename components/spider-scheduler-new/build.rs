//! Build script that compiles the prototype scheduler's protobuf schema into `OUT_DIR`.

use std::env;
use std::path::PathBuf;

/// The root of the crate's protobuf sources, relative to the crate root.
const PROTO_ROOT: &str = "proto";

/// The protobuf source file to compile, relative to [`PROTO_ROOT`].
const PROTO_SOURCE_FILE: &str = "prototype_scheduler.proto";

/// Compiles the prototype scheduler's protobuf schema, emitting both the client and the server
/// stubs into `OUT_DIR`.
///
/// # Panics
///
/// Panics if:
///
/// * `CARGO_MANIFEST_DIR` is not set.
/// * The protobuf compilation fails.
fn main() {
    let crate_root = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("`CARGO_MANIFEST_DIR` env var not set"),
    );
    let proto_root = crate_root.join(PROTO_ROOT);
    let proto_source = proto_root.join(PROTO_SOURCE_FILE);
    println!("cargo:rerun-if-changed={}", proto_source.display());

    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        // `NextTaskRequest` uses proto3 `optional`, which `protoc` releases older than 3.15 gate
        // behind this flag. Newer releases accept it as a no-op.
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&[proto_source], &[proto_root])
        .expect("failed to compile the prototype scheduler protobuf schema");
}
