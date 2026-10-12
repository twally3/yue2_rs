//! CPU-only helpers for the yue2-music-rs skill. No models or Python needed.
#[path = "music_tools/abc.rs"]
mod abc;
#[path = "music_tools/bundle.rs"]
mod bundle;
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};
use yue2_rs::{
    Result, audio,
    json::{self, Value},
};

fn write_new(path: &Path, text: &str) -> Result<()> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(text.as_bytes())?;
    Ok(())
}
fn voice(value: Option<&String>) -> Result<Option<usize>> {
    match value.map(String::as_str).unwrap_or("both") {
        "both" => Ok(None),
        "Vocal" => Ok(Some(0)),
        "Ins" => Ok(Some(1)),
        _ => Err("Voice must be both, Vocal or Ins".into()),
    }
}
fn run() -> Result<i32> {
    let mut args = std::env::args().skip(1);
    let action = args.next().unwrap_or_default();
    if action.is_empty() || action == "--help" || action == "-h" {
        println!(
            "Native YuE2 music helpers (CPU only)\n\n  inspect SCORE [--output REPORT.json]\n  compare BEFORE AFTER [--voices both|Vocal|Ins] [--allow-tempo-change] [--output REPORT.json]\n  strip-chords SOURCE OUTPUT [--keep-voice both|Vocal|Ins]\n  instrumentalize SOURCE --output DIRECTORY [--overlap vocal|error] [--drop-chords]\n  listen SONG_DIRECTORY... --output DIRECTORY\n\nOutputs must be fresh; source files are never overwritten."
        );
        return Ok(0);
    }
    let mut flags = BTreeMap::new();
    let mut positional = Vec::new();
    while let Some(arg) = args.next() {
        if arg.starts_with('-') {
            let value = if matches!(arg.as_str(), "--allow-tempo-change" | "--drop-chords") {
                String::new()
            } else {
                args.next().ok_or("Missing option value")?
            };
            if flags.insert(arg, value).is_some() {
                return Err("Duplicate option".into());
            }
        } else {
            positional.push(PathBuf::from(arg));
        }
    }
    let allowed: &[&str] = match action.as_str() {
        "inspect" | "listen" => &["--output"],
        "compare" => &["--output", "--voices", "--allow-tempo-change"],
        "strip-chords" => &["--keep-voice"],
        "instrumentalize" => &["--output", "--overlap", "--drop-chords"],
        _ => return Err("Unknown helper; use --help".into()),
    };
    if flags.keys().any(|k| !allowed.contains(&k.as_str())) {
        return Err("Unknown option for this helper".into());
    }
    let count = match action.as_str() {
        "compare" | "strip-chords" => 2,
        "listen" => positional.len(),
        _ => 1,
    };
    if positional.len() != count || count == 0 {
        return Err("Wrong number of input paths; use --help".into());
    }
    let output = flags.get("--output").map(PathBuf::from);
    if output.as_ref().is_some_and(|p| p.exists()) {
        return Err("Choose a fresh output path".into());
    }
    if action == "listen" {
        return Ok(i32::from(!bundle::build(
            &positional,
            output.as_deref().ok_or("listen requires --output")?,
        )?));
    }
    let source = fs::read_to_string(&positional[0])?;
    let score = abc::Score::parse(&source)?;
    let data = match action.as_str() {
        "inspect" => {
            let mut data = score.report().object()?.clone();
            data.insert("sha256".into(), audio::sha256_file(&positional[0])?.into());
            Value::Object(data)
        }
        "compare" => score.compare(
            &abc::Score::parse(&fs::read_to_string(&positional[1])?)?,
            voice(flags.get("--voices"))?,
            flags.contains_key("--allow-tempo-change"),
        ),
        "strip-chords" => {
            write_new(
                &positional[1],
                &score.strip_chords(voice(flags.get("--keep-voice"))?)?,
            )?;
            println!("{}", positional[1].display());
            return Ok(0);
        }
        "instrumentalize" => {
            let policy = flags
                .get("--overlap")
                .map(String::as_str)
                .unwrap_or("vocal");
            if !["vocal", "error"].contains(&policy) {
                return Err("Overlap must be vocal or error".into());
            }
            let (text, report) =
                score.instrumental(policy == "error", !flags.contains_key("--drop-chords"))?;
            let output = output.ok_or("instrumentalize requires --output")?;
            fs::create_dir_all(&output)?;
            write_new(&output.join("source.abc"), &source)?;
            write_new(&output.join("score.abc"), &text)?;
            let mut report = report.object()?.clone();
            report.insert(
                "source_sha256".into(),
                audio::sha256_file(output.join("source.abc"))?.into(),
            );
            report.insert(
                "output_sha256".into(),
                audio::sha256_file(output.join("score.abc"))?.into(),
            );
            write_new(
                &output.join("transfer.json"),
                &(json::stringify(&Value::Object(report)) + "\n"),
            )?;
            println!("{}", output.join("score.abc").display());
            return Ok(0);
        }
        _ => unreachable!(),
    };
    if let Some(path) = output {
        write_new(&path, &(json::stringify(&data) + "\n"))?;
    } else {
        println!("{}", json::stringify(&data));
    }
    Ok(i32::from(
        data.optional("match")
            .is_some_and(|v| v.boolean().ok() == Some(false)),
    ))
}
fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("Music helper failed: {error}");
            std::process::exit(2);
        }
    }
}
