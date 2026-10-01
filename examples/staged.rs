//! Equivalent of the reference's plan/save/restore/semantic/synthesize/decode example.
use yue2_rs::{
    Result, audio, json,
    pipeline::{Pipeline, SymbolicPlan},
    protocol::SongRequest,
};
fn main() -> Result<()> {
    let output = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "outputs/staged".into());
    let directory = std::path::Path::new(&output);
    if directory.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "Choose a fresh output directory",
        )
        .into());
    }
    let request = SongRequest::from_json(&json::parse(include_str!("song.json"))?)?;
    let mut pipe = Pipeline::from_pretrained("m-a-p/YuE2-3B", "m-a-p/YuE2-Vae")?;
    let plan = pipe.plan(request, None)?;
    plan.save(directory)?;
    let restored = SymbolicPlan::load(directory)?;
    let semantic = pipe.generate_semantic(restored, None)?;
    let latents = pipe.synthesize(&semantic)?;
    let audio = pipe.decode(&latents, false)?;
    audio::save_tensor(directory.join("latent.npy"), &latents)?;
    audio::save_ids(directory.join("semantic.npy"), &semantic.tokens)?;
    audio::save_flac(directory.join("audio.flac"), &audio, 48000)?;
    if semantic.truncated || semantic.plan.truncated {
        std::process::exit(1);
    }
    Ok(())
}
