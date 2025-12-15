use prost_build::Config;
use std::io::Result;

fn main() -> Result<()> {
    println!(
        "cargo:warning=OUT_DIR={}",
        std::env::var("OUT_DIR").unwrap()
    );
    Ok(())
}

