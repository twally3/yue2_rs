use std::{env, path::PathBuf, process::Command};
fn main() {
    println!("cargo:rerun-if-changed=src/metal.m");
    println!("cargo:rerun-if-changed=src/attention.metal");
    println!("cargo:rerun-if-changed=src/vae_graph.m");
    let target = env::var("TARGET").unwrap();
    if !target.contains("apple-darwin") {
        panic!("YuE2 supports macOS only");
    }
    let arch = if target.starts_with("aarch64") {
        "arm64"
    } else {
        "x86_64"
    };
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let shader = std::fs::read_to_string("src/attention.metal").unwrap();
    std::fs::write(
        output.join("yue2_shader.inc"),
        format!("static const char *yue2_shader = {shader:?};\n"),
    )
    .unwrap();
    let object = output.join("metal.o");
    let library = output.join("libyue2metal.a");
    let vae_object = output.join("vae_graph.o");
    assert!(
        Command::new("/usr/bin/clang")
            .args([
                "-arch",
                arch,
                "-mmacosx-version-min=11.0",
                "-O3",
                "-fobjc-arc",
                "-I",
            ])
            .arg(&output)
            .args(["-c", "src/metal.m", "-o"])
            .arg(&object)
            .status()
            .unwrap()
            .success(),
        "Native Metal bridge compilation failed"
    );
    assert!(
        Command::new("/usr/bin/clang")
            .args([
                "-arch",
                arch,
                "-mmacosx-version-min=11.0",
                "-O3",
                "-fobjc-arc",
                "-c",
                "src/vae_graph.m",
                "-o"
            ])
            .arg(&vae_object)
            .status()
            .unwrap()
            .success(),
        "Native VAE graph compilation failed"
    );
    assert!(
        Command::new("/usr/bin/ar")
            .arg("crs")
            .arg(&library)
            .arg(&object)
            .arg(&vae_object)
            .status()
            .unwrap()
            .success(),
        "Metal archive creation failed"
    );
    println!("cargo:rustc-link-search=native={}", output.display());
    println!("cargo:rustc-link-lib=static=yue2metal");
    for framework in [
        "Foundation",
        "Metal",
        "MetalPerformanceShaders",
        "MetalPerformanceShadersGraph",
    ] {
        println!("cargo:rustc-link-lib=framework={framework}");
    }
    println!("cargo:rustc-link-lib=objc");
    println!("cargo:rustc-link-arg=-mmacosx-version-min=11.0");
}
