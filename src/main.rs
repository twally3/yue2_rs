fn main() {
    match yue2_rs::cli::run("generate") {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("YuE2: {error}");
            std::process::exit(2);
        }
    }
}
