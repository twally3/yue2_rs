//! Native macOS inference for the released YuE2 AR/NAR music model.
pub mod audio;
pub mod checkpoint;
pub mod cli;
pub mod json;
pub mod model;
pub mod pipeline;
pub mod protocol;
pub mod sampling;
pub mod tensor;
pub mod tokenizer;
pub mod vae;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;
pub(crate) fn invalid(message: impl Into<String>) -> Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
}
